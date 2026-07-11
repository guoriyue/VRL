"""Drift metric shared by the fp8 perf probes.

The fp8 GEMM itself is NOT here: probes exercise the production
``vrl.nn.quantization.fp8.Fp8Linear`` directly so they measure the kernel the
rollout engine actually runs (the hand-copied amax-scale/_scaled_mm sequence
that used to live here was form-4 duplication of that module).
"""

from __future__ import annotations

import torch


def relative_l1_drift(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean absolute relative drift, using fp32 reductions."""

    return float((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-12))
