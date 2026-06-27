"""Small fp8 math helpers shared by fp8 perf probes."""

from __future__ import annotations

import torch

from vrl.nn.quantization.fp8 import FP8_E4M3_MAX


def amax_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Per-tensor e4m3 scale mapping a tensor's amax onto the fp8 max value."""

    return (tensor.abs().amax() / FP8_E4M3_MAX).clamp_min(1e-12).to(torch.float32)


def tensorwise_fp8_matmul(
    x_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` through tensorwise fp8-e4m3 ``_scaled_mm``."""

    x_scale = amax_scale(x_bf16)
    weight_scale = amax_scale(weight_bf16)
    x_fp8 = (x_bf16 / x_scale).to(torch.float8_e4m3fn)
    weight_fp8 = (weight_bf16 / weight_scale).to(torch.float8_e4m3fn)
    return torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=x_scale,
        scale_b=weight_scale,
        out_dtype=out_dtype,
    )


def relative_l1_drift(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean absolute relative drift, using fp32 reductions."""

    return float((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-12))
