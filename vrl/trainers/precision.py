"""Shared trainer precision normalization."""

from __future__ import annotations

from typing import Any


def normalize_mixed_precision(mixed_precision: Any) -> str:
    """Normalize a value to the AMP ``no``/``fp16``/``bf16`` protocol."""

    if isinstance(mixed_precision, bool):
        if mixed_precision:
            raise ValueError(
                "AMP mixed precision value true is ambiguous; expected 'no', 'fp16', or 'bf16'",
            )
        return "no"

    token = str(mixed_precision or "").lower().strip()
    if not token:
        return "no"

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
    """Return the model weight dtype implied by normalized precision config."""

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
