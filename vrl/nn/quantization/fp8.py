"""fp8 dynamic-quantization linear for the rollout (generation) forward.

The real ``torch._scaled_mm`` path that lets the rollout DiT run its big policy
GEMMs (attention QKV/out, MLP) in fp8-e4m3 while keeping a bf16 master weight and
bf16-accumulated output. This is the kernel the rollout engine swaps in when
``precision.rollout=fp8`` (SPRINT_fp8_rollout_gemm_kernel.md); training/replay is
untouched (still bf16/fp32).

Why a module swap (not a dtype cast): fp8 is not a drop-in dtype — a plain
``nn.Linear`` on float8 storage raises ``"addmm_cuda" not implemented for
Float8_e4m3fn``. The GEMM must explicitly quantize both operands with a scale,
call ``_scaled_mm``, and accumulate in bf16. :class:`Fp8Linear` wraps exactly that
and is a drop-in replacement for ``nn.Linear`` in the forward.

Recipe: a bf16 master weight is kept and quantized to the fp8 cache at
construction and again after each weight-sync (full fine-tune overwrites it every
step); the activation is quantized per forward.

- ``rowwise`` (default): per-token activation + per-output-channel weight scales,
  torch ``_scaled_mm`` — dep-free, robust to activation outliers.
- ``tensorwise``: one scalar scale per tensor, torch ``_scaled_mm`` — cheapest.
- ``blockwise``: 1x128 activation + 128x128 weight scales (DeepSeek recipe),
  **delegated to vLLM's triton kernel** (``w8a8_triton_block_scaled_mm``) instead
  of hand-rolling — best accuracy on outliers, and runs on CUDA 12.8 (the torch
  ``_scaled_mm`` block path needs CUDA>=12.9). Requires vLLM installed.

All accumulate in bf16.
"""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.quantization.base import QuantizedLinear

# fp8-e4m3 max representable magnitude; the amax scale maps a tensor's peak onto it.
FP8_E4M3_MAX = 448.0
# Block size for the ``blockwise`` recipe (DeepSeek/vLLM standard 128).
FP8_BLOCK = 128

# Module-path substrings whose nn.Linear must stay in bf16: the small,
# numerically sensitive layers (embeddings, the final noise-pred head, anything
# feeding a norm / AdaLN modulation, timestep projections). Quantizing these
# trades accuracy for ~no speed (they are tiny). Matched against the dotted path
# within the transformer; size filtering (min_features) skips the rest.
# AdaLN modulation is named per-backbone: flux/sd3.5 fold it into ``norm*.linear``
# (caught by ``norm``), but qwen-image exposes it as ``img_mod``/``txt_mod`` and
# its text-conditioning input as ``txt_in`` — none of which contain ``norm``, so
# they must be listed explicitly or they would be silently quantized (the
# modulation drives every block's scale/shift/gate — the most precision-sensitive
# GEMM in the stack).
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "norm",
    "embed",
    "proj_out",
    "time_",
    "_mod",
    "txt_in",
)


def _amax_scale(t: torch.Tensor, dim: int | None) -> torch.Tensor:
    """e4m3 scale mapping ``t``'s amax onto FP8_E4M3_MAX (fp32, per-``dim`` or scalar)."""
    amax = t.abs().amax() if dim is None else t.abs().amax(dim=dim, keepdim=True)
    return (amax / FP8_E4M3_MAX).clamp_min(1e-12).to(torch.float32)


def vllm_block_fp8_available() -> bool:
    """Whether the opt-in ``blockwise`` recipe (vLLM triton kernel) can be used."""
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


class Fp8Linear(QuantizedLinear):
    """Drop-in ``nn.Linear`` replacement running the matmul in fp8-e4m3.

    Keeps a bf16 master ``weight`` (so RL weight-sync loads normally) plus a
    derived fp8 cache; quantizes the activation dynamically each forward and runs
    ``torch._scaled_mm`` with bf16 accumulation. The bias (if any) stays in the
    original dtype and is added after the GEMM.
    """

    def __init__(self, linear: nn.Linear, *, recipe: str = "rowwise") -> None:
        super().__init__()
        if recipe not in ("rowwise", "tensorwise", "blockwise"):
            raise ValueError(f"fp8 recipe must be rowwise/tensorwise/blockwise; got {recipe!r}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # blockwise needs 128-aligned dims (vLLM block kernel); fall back to rowwise
        # for the rare non-aligned linear so the swap never breaks.
        if recipe == "blockwise" and (
            self.in_features % FP8_BLOCK or self.out_features % FP8_BLOCK
        ):
            recipe = "rowwise"
        self.recipe = recipe
        # Keep the bf16 master weight under its original ``.weight`` key so RL
        # weight-sync (full fine-tune syncs the base weights every step) can load
        # updated weights normally; the fp8 cache is re-derived after each load
        # (_load_from_state_dict). LoRA rollouts sync adapters, not the base, so
        # this only matters for full fine-tune — but it makes the swap correct for
        # both. Inference-only use just pays the one init-time quantization.
        self.weight = nn.Parameter(
            linear.weight.data.clone(), requires_grad=linear.weight.requires_grad,
        )
        # Bias is cheap and sensitive; keep it in the master dtype, add post-GEMM.
        self.bias = linear.bias
        # fp8 cache (non-persistent: derived from `weight`, never in the state_dict
        # so the trainable-key check sees exactly `weight`).
        self.register_buffer("weight_fp8", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0), persistent=False)
        self._requantize_weight()

    def _requantize_weight(self) -> None:
        """Re-derive the fp8 weight + scale from the bf16 master (after a sync)."""
        w = self.weight.data
        if self.recipe == "blockwise":
            n, k = w.shape
            wb = w.reshape(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
            scale = (wb.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX).clamp_min(1e-12).float()
            self.weight_fp8 = (wb / scale[:, None, :, None]).reshape(n, k).to(torch.float8_e4m3fn)
            self.weight_scale = scale  # [N/128, K/128]
            return
        scale = _amax_scale(w, dim=1 if self.recipe == "rowwise" else None)
        self.weight_fp8 = (w / scale).to(torch.float8_e4m3fn)
        self.weight_scale = scale

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        super()._load_from_state_dict(state_dict, prefix, *args)
        # A weight-sync just overwrote `weight`; refresh the fp8 cache from it.
        self._requantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        if self.recipe == "blockwise":
            out = self._blockwise_gemm(x_2d.to(torch.bfloat16))
            out = out.reshape(*shape[:-1], self.out_features)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
            return out if self.bias is None else out + self.bias
        if self.recipe == "rowwise":
            x_scale = _amax_scale(x_2d, dim=1)  # [M, 1] per token
            weight_scale = self.weight_scale.reshape(1, self.out_features)  # [1, N] per channel
        else:
            x_scale = _amax_scale(x_2d, dim=None)  # scalar
            weight_scale = self.weight_scale
        x_fp8 = (x_2d.to(torch.bfloat16) / x_scale).to(torch.float8_e4m3fn)
        # _scaled_mm rowwise only emits bf16/fp16; DiT layers fed fp32 activations
        # (adaLN / pooled-projection paths) would otherwise be rejected. Accumulate
        # in bf16 and cast back to the input dtype so the swap is transparent.
        gemm_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # column-major [K, N] operand for _scaled_mm
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=gemm_dtype,
        )
        out = out.reshape(*shape[:-1], self.out_features)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _blockwise_gemm(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """1x128 fp8 block GEMM via vLLM's triton kernel (not hand-rolled).

        Reuses vLLM's ``per_token_group_quant_fp8`` (1x128 activation quant) +
        ``w8a8_triton_block_scaled_mm`` (128x128 weight blocks), so we don't
        reimplement DeepSeek-style block fp8 and it runs on CUDA 12.8.
        """
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8,
                w8a8_triton_block_scaled_mm,
            )
        except ImportError as exc:  # pragma: no cover - dep guard
            raise RuntimeError(
                "fp8 recipe='blockwise' reuses vLLM's block kernel but vLLM is not "
                "installed. Install the vllm extra, or use recipe='rowwise' (torch, "
                "dep-free).",
            ) from exc
        x_fp8, x_scale = per_token_group_quant_fp8(x_bf16, FP8_BLOCK)
        return w8a8_triton_block_scaled_mm(
            x_fp8, self.weight_fp8, x_scale, self.weight_scale,
            [FP8_BLOCK, FP8_BLOCK], output_dtype=torch.bfloat16,
        )

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, recipe={self.recipe}, fp8=e4m3"


def swap_linears_to_fp8(
    root: nn.Module,
    *,
    recipe: str = "rowwise",
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    min_features: int = 1024,
) -> list[str]:
    """Replace big ``nn.Linear`` modules under ``root`` with :class:`Fp8Linear` in place.

    Quantizes a Linear only when its dotted path matches no ``exclude`` substring
    AND both in/out features are ``>= min_features`` (small Linears are cheap and
    quantizing them only adds drift). Default ``rowwise`` (torch, dep-free, modest
    memory — the validated path); pass ``recipe="blockwise"`` to opt into vLLM's
    faster/more-accurate block kernel (needs more GPU headroom). Returns the dotted
    paths swapped.
    """

    swapped: list[str] = []
    for parent_path, parent in root.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            path = f"{parent_path}.{child_name}" if parent_path else child_name
            if any(token in path for token in exclude):
                continue
            if child.in_features < min_features or child.out_features < min_features:
                continue
            setattr(parent, child_name, Fp8Linear(child, recipe=recipe))
            swapped.append(path)
    return swapped


__all__ = [
    "DEFAULT_EXCLUDE",
    "Fp8Linear",
    "swap_linears_to_fp8",
    "vllm_block_fp8_available",
]
