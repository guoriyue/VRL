"""Low-overhead training diagnostics for parity checks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def trainable_state_digest(module: Any) -> dict[str, Any]:
    """Return a stable digest and basic stats for trainable tensors only.

    The digest intentionally uses only tensors with ``requires_grad=True`` when
    ``named_parameters`` is available. That keeps SD3 LoRA parity checks focused
    on adapter state instead of hashing the frozen base checkpoint.
    """

    import torch

    tensors: list[tuple[str, Any]] = []
    named_parameters = getattr(module, "named_parameters", None)
    if callable(named_parameters):
        tensors = [
            (name, parameter)
            for name, parameter in named_parameters()
            if getattr(parameter, "requires_grad", False)
        ]

    if not tensors:
        state_dict = getattr(module, "state_dict", None)
        if callable(state_dict):
            state = state_dict()
            lora_items = [
                (name, tensor)
                for name, tensor in state.items()
                if isinstance(tensor, torch.Tensor) and "lora_" in name
            ]
            tensors = lora_items or [
                (name, tensor)
                for name, tensor in state.items()
                if isinstance(tensor, torch.Tensor)
            ]

    digest = hashlib.sha256()
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    tensor_count = 0
    numel = 0
    square_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0

    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        if not isinstance(tensor, torch.Tensor):
            continue
        detached = tensor.detach()
        dtype = str(detached.dtype)
        device = str(detached.device)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1

        cpu = detached.float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(detached.shape)).encode("utf-8"))
        digest.update(dtype.encode("utf-8"))
        digest.update(cpu.numpy().tobytes())

        tensor_count += 1
        tensor_numel = int(cpu.numel())
        numel += tensor_numel
        if tensor_numel:
            square_sum += float(cpu.pow(2).sum().item())
            abs_sum += float(cpu.abs().sum().item())
            max_abs = max(max_abs, float(cpu.abs().max().item()))

    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "numel": numel,
        "l2_norm": math.sqrt(square_sum),
        "mean_abs": abs_sum / numel if numel else 0.0,
        "max_abs": max_abs,
        "dtypes": dtype_counts,
        "devices": device_counts,
    }


def parameter_state_summary(module: Any) -> dict[str, Any]:
    """Return dtype/device counts for all registered parameters."""

    import torch

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return {"parameter_count": 0, "numel": 0, "dtypes": {}, "devices": {}}

    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    trainable_dtype_counts: dict[str, int] = {}
    trainable_device_counts: dict[str, int] = {}
    parameter_count = 0
    trainable_count = 0
    numel = 0
    trainable_numel = 0

    for _, parameter in named_parameters():
        if not isinstance(parameter, torch.Tensor):
            continue
        dtype = str(parameter.dtype)
        device = str(parameter.device)
        count = int(parameter.numel())
        parameter_count += 1
        numel += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1
        if getattr(parameter, "requires_grad", False):
            trainable_count += 1
            trainable_numel += count
            trainable_dtype_counts[dtype] = trainable_dtype_counts.get(dtype, 0) + 1
            trainable_device_counts[device] = trainable_device_counts.get(device, 0) + 1

    return {
        "parameter_count": parameter_count,
        "numel": numel,
        "dtypes": dtype_counts,
        "devices": device_counts,
        "trainable_parameter_count": trainable_count,
        "trainable_numel": trainable_numel,
        "trainable_dtypes": trainable_dtype_counts,
        "trainable_devices": trainable_device_counts,
    }


def tensor_stats(value: Any) -> dict[str, Any]:
    """Return JSON-safe summary stats for a tensor-like value."""

    import torch

    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    detached = value.detach().float().cpu()
    numel = int(detached.numel())
    if numel == 0:
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": 0,
        }
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": numel,
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()) if numel > 1 else 0.0,
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def write_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "parameter_state_summary",
    "tensor_stats",
    "trainable_state_digest",
    "write_jsonl",
]
