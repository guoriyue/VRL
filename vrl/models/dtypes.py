"""Canonical torch dtype parsing for the model-loading boundary.

One place that maps the tolerant set of dtype spellings the codebase accepts at
config/checkpoint boundaries — ``bf16``/``bfloat16``/``half``/``torch.float16``
etc. — onto ``torch.dtype`` values, and back to canonical config strings. The
config layer (:mod:`vrl.config.precision`) speaks only the three canonical axis
names; this module is the lenient parser used where raw user/checkpoint strings
enter the model layer, so the alias table lives in exactly one place instead of
being re-hand-rolled in every model/runtime file.

``torch`` is imported lazily so this module stays importable in torch-free
contexts (config resolution, Ray launcher serialization).
"""

from __future__ import annotations

from typing import Any

# Tolerant spelling -> canonical config string. Single source of truth for both
# the forward (string -> torch.dtype) and reverse (-> config string) directions.
_CANONICAL_STRING: dict[str, str] = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def resolve_torch_dtype(value: Any) -> Any:
    """Resolve a dtype spelling (or an existing ``torch.dtype``) into a ``torch.dtype``."""

    import torch

    if isinstance(value, torch.dtype):
        return value
    key = str(value).removeprefix("torch.").lower()
    try:
        name = _CANONICAL_STRING[key]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {value!r}") from exc
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def dtype_to_config_string(value: Any) -> str:
    """Normalize a dtype spelling (or ``torch.dtype``) to its canonical config string.

    Unknown values pass through unchanged so callers can forward already-canonical
    or non-dtype tokens without losing them.
    """

    text = str(value).removeprefix("torch.")
    return _CANONICAL_STRING.get(text.lower(), text)


__all__ = ["dtype_to_config_string", "resolve_torch_dtype"]
