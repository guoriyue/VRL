"""NVFP4 dynamic-quantization linear for the rollout (generation) forward.

The fp4 sibling of :mod:`vrl.nn.quantization.fp8` (SPRINT_nvfp4_rollout.md): the
rollout policy's selected MLP GEMMs run through ``torch._scaled_mm`` on
``float4_e2m1fn_x2`` operands while the train-dtype master weight stays for RL
weight-sync. Training/replay is untouched — same lossy-rollout +
TIS/drift-guard contract as fp8, with one recipe instead of three:

- ``nvfp4`` (the only recipe): two-level scaling per the NVIDIA recipe. A global
  fp32 tensor scale sized so block scales fit e4m3 (``amax / (448 * 6)``), then
  1x16 block scales stored as fp8-e4m3 in cuBLAS's blocked-swizzled layout, and
  values rounded to e2m1 ({0, 0.5, 1, 1.5, 2, 3, 4, 6} x sign). The GEMM output
  is multiplied back by the two tensor scales.

CUDA quantize + pack uses a fused Triton kernel; the CPU implementation below is
the bit-reference used for initialization and tests. Torch 2.11 has no native
cast into ``float4_e2m1fn_x2`` (``copy_kernel`` is not implemented), so both
paths write the two E2M1 nibbles explicitly. The CUDA bytes/scales are checked
against the independent CPU path before real ``_scaled_mm`` parity tests.

Alignment: the quantization block is 16 values, while the current packed
``_scaled_mm`` kernel additionally requires ``K % 32 == 0`` and ``N % 16 == 0``.
The swap skips Linears outside that executable contract (there is no coarser
fp4 recipe to fall back to, unlike blockwise->rowwise in fp8).
"""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.quantization.base import QuantizedLinear
from vrl.nn.quantization.formats import (
    E2M1_MAX,
    E2M1_MIDPOINTS,
    E2M1_VALUES,
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4_K_ALIGNMENT,
    NVFP4_N_ALIGNMENT,
    NVFP4_SCALE_COL_TILE,
    NVFP4_SCALE_INNER_ROWS,
    NVFP4_SCALE_ROW_TILE,
)
from vrl.nn.quantization.targeting import (
    DEFAULT_EXCLUDE,
    LinearTargetProfile,
    matches_linear_target,
)

# e2m1 representable magnitudes (3 exponent-ish levels x 1 mantissa bit + zero)
# and the midpoints used for round-to-nearest bucketing.
_E2M1_BOUNDS = torch.tensor(
    E2M1_MIDPOINTS,
)
# nvfp4 block size (1x16 along K, e4m3 scale per block).
FP4_BLOCK = NVFP4_BLOCK_SIZE
# ``float4_e2m1fn_x2`` packs two K values per byte. The current scaled-mm
# backend requires that packed trailing dimension to be divisible by 16.
FP4_K_ALIGNMENT = NVFP4_K_ALIGNMENT
FP4_N_ALIGNMENT = NVFP4_N_ALIGNMENT


def nvfp4_available(device: torch.device | str | None = None) -> bool:
    """Whether ``device`` can execute the packed NVFP4 scaled-mm path."""

    if not torch.cuda.is_available() or not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        return False
    return torch.cuda.get_device_capability(target)[0] >= 10


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def to_blocked_scale_layout(scales: torch.Tensor) -> torch.Tensor:
    """Swizzle a ``[rows, K/16]`` scale matrix into cuBLAS's blocked layout.

    ``torch._scaled_mm``'s nvfp4 path consumes block scales in the hardware's
    128-row x 4-col tiled arrangement, flattened. Rows/cols are zero-padded up
    to the tile multiples first.
    """
    rows, cols = scales.shape
    n_row_blocks = _ceil_div(rows, NVFP4_SCALE_ROW_TILE)
    n_col_blocks = _ceil_div(cols, NVFP4_SCALE_COL_TILE)
    padded = scales
    if (rows, cols) != (
        n_row_blocks * NVFP4_SCALE_ROW_TILE,
        n_col_blocks * NVFP4_SCALE_COL_TILE,
    ):
        padded = torch.zeros(
            n_row_blocks * NVFP4_SCALE_ROW_TILE,
            n_col_blocks * NVFP4_SCALE_COL_TILE,
            device=scales.device,
            dtype=scales.dtype,
        )
        padded[:rows, :cols] = scales
    blocks = padded.view(
        n_row_blocks,
        NVFP4_SCALE_ROW_TILE,
        n_col_blocks,
        NVFP4_SCALE_COL_TILE,
    ).permute(0, 2, 1, 3)
    return (
        blocks.reshape(
            -1,
            NVFP4_SCALE_ROW_TILE // NVFP4_SCALE_INNER_ROWS,
            NVFP4_SCALE_INNER_ROWS,
            NVFP4_SCALE_COL_TILE,
        )
        .transpose(1, 2)
        .reshape(
            -1,
            NVFP4_SCALE_INNER_ROWS,
            NVFP4_SCALE_ROW_TILE // NVFP4_SCALE_INNER_ROWS * NVFP4_SCALE_COL_TILE,
        )
        .flatten()
    )


def quantize_nvfp4(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to nvfp4: values, swizzled block scales, tensor scale.

    CUDA dispatches to the fused rollout kernel; CPU remains the readable
    bit-reference. Returns ``(packed_fp4x2 [R, K/2], swizzled e4m3 block scales (flat),
    tensor_scale fp32 scalar)``. Dequantization is deliberately not part of
    this hot-path API: materializing an ``[R, K]`` fp32 twin in every rollout
    layer defeats the memory and latency purpose of quantization.
    """
    r, k = t.shape
    if k % FP4_BLOCK:
        raise ValueError(f"nvfp4 needs K % {FP4_BLOCK} == 0, got K={k}")
    tensor_scale = (t.detach().abs().amax().float() / (FP8_E4M3_MAX * E2M1_MAX)).clamp_min(1e-12)
    if t.is_cuda:
        from vrl.nn.quantization.fp4_kernels import quantize_nvfp4_cuda

        packed, scales = quantize_nvfp4_cuda(t, tensor_scale)
        return packed, scales, tensor_scale

    tf = t.float()
    blocks = tf.view(r, k // FP4_BLOCK, FP4_BLOCK)
    block_scale = (blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX / tensor_scale).clamp(
        1e-12, FP8_E4M3_MAX
    )
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    effective = block_scale_e4m3.float() * tensor_scale
    scaled = torch.where(effective > 0, blocks / effective, torch.zeros_like(blocks)).clamp(
        -E2M1_MAX, E2M1_MAX
    )

    sign = scaled < 0
    magnitude = scaled.abs()
    bounds = _E2M1_BOUNDS.to(t.device)
    idx = torch.bucketize(magnitude, bounds)
    # ``bucketize(..., right=False)`` chooses the lower value at every exact
    # midpoint. IEEE nearest-even instead chooses the even E2M1 code, so odd
    # lower codes advance while even lower codes stay put.
    bounded_idx = idx.clamp_max(len(E2M1_VALUES) - 2)
    midpoint = (idx < len(bounds)) & (magnitude == bounds[bounded_idx])
    idx = idx + (midpoint & ((idx & 1) == 1))
    codes = (idx + torch.where(sign, 8, 0)).to(torch.uint8).view(r, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return (
        packed.view(torch.float4_e2m1fn_x2),
        to_blocked_scale_layout(block_scale_e4m3.view(r, k // FP4_BLOCK)),
        tensor_scale,
    )


class Fp4Linear(QuantizedLinear):
    """Drop-in ``nn.Linear`` replacement running the matmul in nvfp4.

    Mirrors :class:`Fp8Linear`'s ownership contract: a train-dtype master ``weight``
    stays under its original key (RL weight-sync loads normally; the fp4 cache
    re-derives after each load), the bias stays in the master dtype and is added
    after the GEMM, and ``drop_master`` frees the master for adapter-only /
    sync-free rollouts.
    """

    quantization_scheme = "fp4"
    cache_buffer_names = ("weight_fp4", "weight_scale", "weight_tensor_scale")

    def __init__(self, linear: nn.Linear, *, recipe: str = "nvfp4") -> None:
        super().__init__()
        if recipe != "nvfp4":
            raise ValueError(f"fp4 recipe must be nvfp4; got {recipe!r}")
        if linear.in_features % FP4_K_ALIGNMENT:
            raise ValueError(
                f"Fp4Linear needs in_features % {FP4_K_ALIGNMENT} == 0 for "
                "packed nvfp4 scaled_mm, "
                f"got {linear.in_features}",
            )
        if linear.out_features % FP4_N_ALIGNMENT:
            raise ValueError(
                f"Fp4Linear needs out_features % {FP4_N_ALIGNMENT} == 0 for "
                f"nvfp4 scaled_mm, got {linear.out_features}",
            )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.recipe = recipe
        self.weight = nn.Parameter(
            linear.weight.data.clone(),
            requires_grad=linear.weight.requires_grad,
        )
        self.bias = linear.bias
        # fp4 cache (non-persistent: derived from `weight`, never in the
        # state_dict so the trainable-key check sees exactly `weight`).
        self.register_buffer(
            "weight_fp4",
            torch.empty(0, dtype=torch.float4_e2m1fn_x2),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer("weight_tensor_scale", torch.empty(0), persistent=False)
        self._requantize_weight()

    def _requantize_weight(self) -> None:
        """Re-derive the fp4 weight + scales from the master (after a sync)."""
        packed, scales, tensor_scale = quantize_nvfp4(self.weight.data)
        self.weight_fp4 = packed
        self.weight_scale = scales
        self.weight_tensor_scale = tensor_scale

    def drop_master(self) -> int:
        """Free the train-dtype master, keeping only the fp4 cache (+scales).

        Valid whenever weight-sync never loads base weights into this module —
        see :meth:`Fp8Linear.drop_master`.
        """
        if self.weight is None:
            return 0
        freed = self.weight.numel() * self.weight.element_size()
        self.weight = None
        return freed

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        if self.weight is None and f"{prefix}weight" in state_dict:
            raise RuntimeError(
                "cannot load base weights into a master-free Fp4Linear "
                f"({prefix.rstrip('.')}): the train-dtype master was dropped "
                "(drop_master). Master-free is for adapter-only/frozen "
                "rollouts; full-finetune weight-sync must keep the master.",
            )
        super()._load_from_state_dict(state_dict, prefix, *args)
        if self.weight is not None:
            self._requantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        x_fp4, x_scale, x_tensor_scale = quantize_nvfp4(x_2d)
        out = torch._scaled_mm(
            x_fp4,
            self.weight_fp4.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )
        # Second scaling level: _scaled_mm only saw the e4m3 block scales, so
        # fold the two fp32 tensor scales back in (kept on-device — no sync).
        out = out * (x_tensor_scale * self.weight_tensor_scale).to(out.dtype)
        out = out.reshape(*shape[:-1], self.out_features)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, recipe={self.recipe}, fp4=e2m1"


def swap_linears_to_fp4(
    root: nn.Module,
    *,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    min_features: int = 1024,
    target_profile: LinearTargetProfile | str = LinearTargetProfile.MLP_ONLY,
) -> list[str]:
    """Replace big ``nn.Linear`` modules under ``root`` with :class:`Fp4Linear` in place.

    The conservative production profile targets feed-forward/MLP GEMMs only;
    ``attention_mlp`` is an experimental matched-scope profile for hardware and
    accuracy validation, not a rollout default. Candidates must also avoid
    ``exclude``, meet ``min_features``, and satisfy the executable kernel
    alignment (K multiple of 32, N multiple of 16). Returns the dotted paths
    swapped.
    """
    target_profile = LinearTargetProfile(target_profile)
    swapped: list[str] = []
    for parent_path, parent in root.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            path = f"{parent_path}.{child_name}" if parent_path else child_name
            if any(token in path for token in exclude):
                continue
            if not matches_linear_target(path, target_profile):
                continue
            if child.in_features < min_features or child.out_features < min_features:
                continue
            if child.in_features % FP4_K_ALIGNMENT or child.out_features % FP4_N_ALIGNMENT:
                continue
            setattr(parent, child_name, Fp4Linear(child))
            swapped.append(path)
    return swapped


__all__ = [
    "FP4_BLOCK",
    "FP4_K_ALIGNMENT",
    "FP4_N_ALIGNMENT",
    "Fp4Linear",
    "nvfp4_available",
    "quantize_nvfp4",
    "swap_linears_to_fp4",
    "to_blocked_scale_layout",
]
