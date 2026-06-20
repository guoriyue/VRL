"""Shared trainer precision normalization."""

from __future__ import annotations

from typing import Any


def normalize_mixed_precision(mixed_precision: Any) -> str:
    """Normalize user precision config to ``"no"``, ``"fp16"``, or ``"bf16"``."""

    if isinstance(mixed_precision, bool):
        if mixed_precision:
            raise ValueError(
                "actor.mixed_precision=true is ambiguous; use 'fp16' or 'bf16'",
            )
        return "no"

    precision = str(mixed_precision or "").lower().strip()
    if not precision:
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
    precision = aliases.get(precision, precision)
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "actor.mixed_precision must be one of 'no', 'fp16', or 'bf16', "
            f"got {mixed_precision!r}",
        )
    return precision


def trainer_mixed_precision(config: Any) -> str:
    """Normalize precision from a TrainerConfig-like object."""

    return normalize_mixed_precision(getattr(config, "mixed_precision", ""))


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
        getattr(config, "mixed_precision", ""),
        torch=torch,
    )


__all__ = [
    "normalize_mixed_precision",
    "torch_dtype_for_mixed_precision",
    "torch_dtype_for_trainer_precision",
    "trainer_mixed_precision",
]
