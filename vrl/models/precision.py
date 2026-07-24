"""Project resolved role precision onto PyTorch execution boundaries."""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from vrl.config.precision import Float32Precision, RolePrecision

if TYPE_CHECKING:
    import torch


def model_precision(model: Any) -> RolePrecision:
    """Return the role precision stamped on a runtime or replay model."""

    precision = getattr(model, "precision", None)
    if not isinstance(precision, RolePrecision):
        raise TypeError(
            f"{type(model).__name__}.precision is missing or not a RolePrecision; "
            "models receive it from RuntimeBundle assembly",
        )
    return precision


def model_autocast(
    model: Any,
    device: torch.device | str,
) -> AbstractContextManager[Any]:
    """Apply a model's resolved role dtype and outer-autocast policy."""

    import torch

    precision = model_precision(model)
    dtype = precision.dtype
    if not precision.outer_autocast or dtype == "fp32":
        return contextlib.nullcontext()
    if dtype not in ("fp16", "bf16"):
        raise ValueError(f"unsupported autocast dtype: {dtype!r}")
    device_type = torch.device(device).type
    if device_type == "cpu" and dtype == "fp16":
        return contextlib.nullcontext()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device_type, dtype=torch_dtype)


def apply_float32_precision(mode: Float32Precision) -> None:
    """Apply one concrete FP32 matmul mode to the current PyTorch process.

    PyTorch 2.9 introduced the string-valued ``fp32_precision`` API. The
    fallback keeps supported older releases on the legacy bool API without
    mixing both mechanisms in one process.
    """

    import torch

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
    """Return the effective PyTorch FP32 backend modes for diagnostics."""

    import torch

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
    "model_autocast",
    "model_precision",
]
