"""Project resolved forward precision onto PyTorch execution boundaries."""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from typing import Any

import torch

from vrl.models.interfaces.runtime import Float32Precision, ForwardPrecision


def forward_autocast(
    precision: ForwardPrecision,
    device: torch.device | str,
) -> AbstractContextManager[Any]:
    """Return the transformer's resolved outer-autocast context."""

    if precision.autocast == "off":
        return contextlib.nullcontext()
    device_type = torch.device(device).type
    if device_type == "cpu" and precision.autocast == "fp16":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision.autocast == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def apply_float32_precision(mode: Float32Precision) -> None:
    """Apply one concrete FP32 matmul mode to the current PyTorch process.

    PyTorch 2.9 introduced the string-valued ``fp32_precision`` API. The
    fallback keeps supported older releases on the legacy bool API without
    mixing both mechanisms in one process.
    """

    if mode not in ("ieee", "tf32"):
        raise ValueError(f"unsupported float32 precision mode: {mode!r}")
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision"):
        matmul.fp32_precision = mode
        cudnn.fp32_precision = mode
        return
    enabled = mode == "tf32"
    matmul.allow_tf32 = enabled
    cudnn.allow_tf32 = enabled


def float32_precision_state() -> dict[str, str]:
    """Return the effective PyTorch FP32 backend modes for diagnostics/gates."""

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision"):
        return {
            "matmul": str(matmul.fp32_precision),
            "cudnn": str(cudnn.fp32_precision),
        }
    return {
        "matmul": "tf32" if bool(matmul.allow_tf32) else "ieee",
        "cudnn": "tf32" if bool(cudnn.allow_tf32) else "ieee",
    }


__all__ = [
    "apply_float32_precision",
    "float32_precision_state",
    "forward_autocast",
]
