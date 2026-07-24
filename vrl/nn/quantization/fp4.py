"""NVFP4 dynamic-quantization linear for rollout generation forwards.

Selected MLP GEMMs run through ``torch._scaled_mm`` on packed
``float4_e2m1fn_x2`` operands. A source-dtype master weight remains available
when full-parameter rollout weight sync needs it; the non-persistent NVFP4 cache
is rebuilt after each load. Training/replay remains unquantized.

The single ``nvfp4`` recipe uses two-level NVIDIA scaling: an FP32 tensor scale,
1x16 E4M3 block scales in cuBLAS's blocked layout, and signed E2M1 values. CUDA
quantization and packing use the fused Triton kernel in ``fp4_kernels.py``; the
CPU implementation here is the independent bit reference.
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

_E2M1_BOUNDS = torch.tensor(E2M1_MIDPOINTS)


def nvfp4_available(device: torch.device | str | None = None) -> bool:
    """Whether ``device`` satisfies the packed NVFP4 scaled-mm hardware gate."""

    if not torch.cuda.is_available() or not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        return False
    return torch.cuda.get_device_capability(target)[0] >= 10


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def to_blocked_scale_layout(scales: torch.Tensor) -> torch.Tensor:
    """Swizzle ``[rows, K/16]`` scales into cuBLAS's blocked layout."""

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
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed E2M1 values, swizzled E4M3 scales, and an FP32 tensor scale."""

    rows, k = tensor.shape
    if k % NVFP4_BLOCK_SIZE:
        raise ValueError(f"nvfp4 needs K % {NVFP4_BLOCK_SIZE} == 0, got K={k}")
    tensor_scale = (tensor.detach().abs().amax().float() / (FP8_E4M3_MAX * E2M1_MAX)).clamp_min(
        1e-12
    )
    if tensor.is_cuda:
        from vrl.nn.quantization.fp4_kernels import quantize_nvfp4_cuda

        packed, scales = quantize_nvfp4_cuda(tensor, tensor_scale)
        return packed, scales, tensor_scale

    values_fp32 = tensor.float()
    blocks = values_fp32.view(rows, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_scale = (blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX / tensor_scale).clamp(
        1e-12,
        FP8_E4M3_MAX,
    )
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    effective = block_scale_e4m3.float() * tensor_scale
    scaled = torch.where(effective > 0, blocks / effective, torch.zeros_like(blocks)).clamp(
        -E2M1_MAX,
        E2M1_MAX,
    )

    sign = scaled < 0
    magnitude = scaled.abs()
    bounds = _E2M1_BOUNDS.to(tensor.device)
    code = torch.bucketize(magnitude, bounds)
    # ``bucketize(..., right=False)`` chooses the lower midpoint value. IEEE
    # nearest-even advances only when that lower E2M1 code is odd.
    bounded_code = code.clamp_max(len(E2M1_VALUES) - 2)
    midpoint = (code < len(bounds)) & (magnitude == bounds[bounded_code])
    code = code + (midpoint & ((code & 1) == 1))
    code = (code + torch.where(sign, 8, 0)).to(torch.uint8).view(rows, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (
        packed.view(torch.float4_e2m1fn_x2),
        to_blocked_scale_layout(block_scale_e4m3.view(rows, k // NVFP4_BLOCK_SIZE)),
        tensor_scale,
    )


class Fp4Linear(QuantizedLinear):
    """Drop-in ``nn.Linear`` replacement whose selected GEMM runs in NVFP4."""

    quantization_scheme = "nvfp4"
    cache_buffer_names = ("weight_fp4", "weight_scale", "weight_tensor_scale")

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.in_features % NVFP4_K_ALIGNMENT:
            raise ValueError(
                f"Fp4Linear needs in_features % {NVFP4_K_ALIGNMENT} == 0 for "
                f"packed nvfp4 scaled_mm, got {linear.in_features}",
            )
        if linear.out_features % NVFP4_N_ALIGNMENT:
            raise ValueError(
                f"Fp4Linear needs out_features % {NVFP4_N_ALIGNMENT} == 0 for "
                f"nvfp4 scaled_mm, got {linear.out_features}",
            )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.recipe = "nvfp4"
        self.weight = nn.Parameter(
            linear.weight.data.clone(),
            requires_grad=linear.weight.requires_grad,
        )
        self.bias = linear.bias
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
        """Rebuild the packed NVFP4 weight and scales from the source master."""

        packed, scales, tensor_scale = quantize_nvfp4(self.weight.data)
        self.weight_fp4 = packed
        self.weight_scale = scales
        self.weight_tensor_scale = tensor_scale

    def drop_master(self) -> int:
        """Free the source master while preserving the derived NVFP4 cache."""

        if self.weight is None:
            return 0
        freed = self.weight.numel() * self.weight.element_size()
        self.weight = None
        return freed

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        if self.weight is None and f"{prefix}weight" in state_dict:
            raise RuntimeError(
                "cannot load base weights into a master-free Fp4Linear "
                f"({prefix.rstrip('.')}): the source master was dropped "
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
        # ``_scaled_mm`` consumes the per-block E4M3 scales. Fold the two FP32
        # tensor scales back into the output without a host synchronization.
        out = out * (x_tensor_scale * self.weight_tensor_scale).to(out.dtype)
        out = out.reshape(*shape[:-1], self.out_features)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, recipe={self.recipe}, fp4=e2m1"


def swap_linears_to_nvfp4(
    root: nn.Module,
    *,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    min_features: int = 1024,
    target_profile: LinearTargetProfile | str = LinearTargetProfile.MLP_ONLY,
) -> list[str]:
    """Replace eligible large policy Linears with :class:`Fp4Linear` in place.

    Production defaults to MLP-only targeting. Candidates must also avoid the
    shared exclusion taxonomy, meet the size threshold, and satisfy packed
    scaled-mm alignment.
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
            if child.in_features % NVFP4_K_ALIGNMENT or child.out_features % NVFP4_N_ALIGNMENT:
                continue
            setattr(parent, child_name, Fp4Linear(child))
            swapped.append(path)
    return swapped


__all__ = [
    "Fp4Linear",
    "nvfp4_available",
    "quantize_nvfp4",
    "swap_linears_to_nvfp4",
    "to_blocked_scale_layout",
]
