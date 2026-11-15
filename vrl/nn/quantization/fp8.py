"""fp8 dynamic-quantization linear for the rollout (generation) forward.

The real ``torch._scaled_mm`` path that lets the rollout DiT run its big policy
GEMMs (attention QKV/out, MLP) in fp8-e4m3 while keeping a source-dtype master
weight and a bf16/fp16 output matching the surrounding activation. This is the kernel the rollout engine swaps
in when ``precision.rollout.quantization.format=fp8``
(SPRINT_fp8_rollout_gemm_kernel.md), and the replay too under
``precision.training.quantization``.

Why a module swap (not a dtype cast): fp8 is not a drop-in dtype — a plain
``nn.Linear`` on float8 storage raises ``"addmm_cuda" not implemented for
Float8_e4m3fn``. The GEMM must explicitly quantize both operands with a scale,
call ``_scaled_mm``, and request a supported bf16/fp16 output dtype.
:class:`Fp8Linear` wraps exactly that and is a drop-in replacement for
``nn.Linear`` in the forward.

Recipe: a source-dtype master weight is kept and quantized to the fp8 cache at
construction and again after each weight-sync (full fine-tune overwrites it every
step); the activation is quantized per forward.

- ``rowwise`` (default): per-token activation + per-output-channel weight scales,
  torch ``_scaled_mm`` — dep-free, robust to activation outliers.
- ``tensorwise``: one scalar scale per tensor, torch ``_scaled_mm`` — cheapest.

"""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.quantization.base import QuantizedLinear
from vrl.nn.quantization.formats import FP8_E4M3_MAX


class Fp8Linear(QuantizedLinear):
    """Drop-in ``nn.Linear`` replacement running the matmul in fp8-e4m3.

    Keeps a source-dtype master ``weight`` (so RL weight-sync loads normally)
    plus a derived fp8 cache; quantizes the activation dynamically each forward
    and runs ``torch._scaled_mm`` with a bf16/fp16 output. The bias (if any)
    keeps its original dtype in storage and is fused into the GEMM epilogue.
    """

    quantization_scheme = "fp8"
    cache_buffer_names = ("weight_fp8", "weight_scale")

    def __init__(
        self,
        linear: nn.Linear,
        *,
        keep_master: bool = True,
        device: torch.device | str | None = None,
        recipe: str = "rowwise",
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.recipe = recipe
        # A kept master sits under the original ``.weight`` key so a
        # full-finetune weight sync loads it normally; the fp8 cache is
        # re-derived after each load (_load_from_state_dict).
        # Bias keeps its master dtype in storage; forward fuses it into the GEMM.
        self.bias = linear.bias
        # fp8 cache (non-persistent: derived from `weight`, never in the state_dict
        # so the trainable-key check sees exactly `weight`).
        self.register_buffer(
            "weight_fp8", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False
        )
        self.register_buffer("weight_scale", torch.empty(0), persistent=False)
        # The cache is quantized on ``device``, where the forward will run: the
        # CPU and CUDA quantizers do not agree bit for bit on real weights, so
        # a rollout and a replay match only if both quantize on the same kind
        # of device. A frozen base is never written again and keeps no master;
        # the source Linear's storage is released with the Linear.
        source = linear.weight.data if device is None else linear.weight.data.to(device)
        self.weight = nn.Parameter(source, requires_grad=False)
        self._requantize_weight()
        self.weight = (
            nn.Parameter(linear.weight.data.clone(), requires_grad=linear.weight.requires_grad)
            if keep_master
            else None
        )

    @staticmethod
    def _amax_scale(t: torch.Tensor, dim: int | None) -> torch.Tensor:
        """e4m3 scale mapping ``t``'s amax onto FP8_E4M3_MAX (fp32, per-``dim`` or scalar)."""
        amax = t.abs().amax() if dim is None else t.abs().amax(dim=dim, keepdim=True)
        return (amax / FP8_E4M3_MAX).clamp_min(1e-12).to(torch.float32)

    def _requantize_weight(self) -> None:
        """Re-derive the fp8 weight + scale from the source master after a sync."""
        w = self.weight.data
        scale = self._amax_scale(w, dim=1 if self.recipe == "rowwise" else None)
        self.weight_fp8 = (w / scale).to(torch.float8_e4m3fn)
        self.weight_scale = scale

    def dequantized_weight(self) -> torch.Tensor:
        return (self.weight_fp8.float() * self.weight_scale).to(torch.bfloat16)

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        if self.recipe == "rowwise":
            x_scale = self._amax_scale(x_2d, dim=1)  # [M, 1] per token
            weight_scale = self.weight_scale.reshape(1, self.out_features)  # [1, N] per channel
        else:
            x_scale = self._amax_scale(x_2d, dim=None)  # scalar
            weight_scale = self.weight_scale
        x_fp8 = (x_2d.to(torch.bfloat16) / x_scale).to(torch.float8_e4m3fn)
        # _scaled_mm rowwise only emits bf16/fp16; DiT layers fed fp32 activations
        # (adaLN / pooled-projection paths) would otherwise be rejected. Use bf16
        # output there and cast back to the input dtype so the swap is transparent.
        gemm_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        # The epilogue adds the bias to the fp32 accumulator before the single
        # rounding to ``gemm_dtype``; adding it after the bf16 output rounds
        # twice (relative error 2.34e-3 -> 1.66e-3 on an RTX 5090). The kernel
        # requires the bias in the output dtype.
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # column-major [K, N] operand for _scaled_mm
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=None if self.bias is None else self.bias.to(gemm_dtype),
            out_dtype=gemm_dtype,
        )
        out = out.reshape(*shape[:-1], self.out_features)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, recipe={self.recipe}, fp8=e4m3"


__all__ = ["Fp8Linear"]
