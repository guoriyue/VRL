"""Shared trainer precision normalization."""

from __future__ import annotations

from typing import Any

from vrl.config.precision import QuantizationPolicy


def normalize_mixed_precision(mixed_precision: Any) -> str:
    """Extract the base dtype from a trainer role label for legacy consumers."""

    if isinstance(mixed_precision, bool):
        if mixed_precision:
            raise ValueError(
                "AMP mixed precision value true is ambiguous; expected 'no', 'fp16', or 'bf16'",
            )
        return "no"

    token = str(mixed_precision or "").lower().strip()
    if not token:
        return "no"

    parts = token.split("+")
    token = parts[0]
    suffixes = parts[1:]
    quantization_suffixes = [suffix for suffix in suffixes if suffix != "no-autocast"]
    valid_suffixes = len(suffixes) == len(set(suffixes)) and len(quantization_suffixes) <= 1
    if valid_suffixes and quantization_suffixes:
        try:
            QuantizationPolicy(format=quantization_suffixes[0])
        except ValueError:
            valid_suffixes = False
    if not valid_suffixes:
        raise ValueError(
            "trainer precision label has an unknown or duplicate execution policy; "
            f"got {mixed_precision!r}",
        )

    aliases = {
        "none": "no",
        "off": "no",
        "false": "no",
        "0": "no",
        "fp32": "no",
        "float32": "no",
        "float": "no",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
    }
    token = aliases.get(token, token)
    if token not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "AMP mixed precision must resolve to 'no', 'fp16', or 'bf16'; "
            f"got {mixed_precision!r}",
        )
    return token


def torch_dtype_for_mixed_precision(
    mixed_precision: Any,
    *,
    torch: Any,
) -> Any:
    """Return the model weight dtype encoded by a trainer role label."""

    precision = normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise AssertionError(f"unreachable mixed precision: {precision!r}")


def torch_dtype_for_trainer_precision(config: Any, torch: Any) -> Any:
    """Return the model weight dtype for a TrainerConfig-like object."""

    return torch_dtype_for_mixed_precision(
        getattr(config, "train_precision", ""),
        torch=torch,
    )


__all__ = [
    "normalize_mixed_precision",
    "torch_dtype_for_mixed_precision",
    "torch_dtype_for_trainer_precision",
]
