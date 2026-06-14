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

# Canonical dtype name -> tolerant input spellings accepted at config/checkpoint
# boundaries. Source of truth; the lenient lookup below is derived from it so the
# identity entry (a canonical name accepts itself) is generated, not hand-listed.
_DTYPE_SPELLINGS: dict[str, tuple[str, ...]] = {
    "bfloat16": ("bf16",),
    "float16": ("fp16", "half"),
    "float32": ("fp32", "float"),
}
# Tolerant spelling -> canonical config string. Single source of truth for both
# the forward (string -> torch.dtype) and reverse (-> config string) directions.
_DTYPE_NAME_BY_INPUT: dict[str, str] = {
    spelling: canonical
    for canonical, spellings in _DTYPE_SPELLINGS.items()
    for spelling in (canonical, *spellings)
}


def resolve_torch_dtype(value: Any) -> Any:
    """Resolve a dtype spelling (or an existing ``torch.dtype``) into a ``torch.dtype``."""

    import torch

    if isinstance(value, torch.dtype):
        return value
    key = str(value).removeprefix("torch.").lower()
    try:
        name = _DTYPE_NAME_BY_INPUT[key]
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
    return _DTYPE_NAME_BY_INPUT.get(text.lower(), text)


__all__ = ["dtype_to_config_string", "resolve_torch_dtype"]
