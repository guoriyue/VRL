"""Prompt-hint vocabulary and helpers for pose-structure scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_BOTH_HAND_HINTS = ("both hands",)
_HAND_HINTS = ("hand", "hands", "finger", "fingers", "holding", "gesture", "clenched")
_FEET_HINTS = (
    "feet",
    "foot",
    "shoe",
    "shoes",
    "boot",
    "boots",
    "full body",
    "standing",
    "walking",
    "running",
    "jumping",
    "kicking",
)
_HINT_TRANSLATION = str.maketrans({char: " " for char in "_-/.,;:()[]{}!?"})


def _constraint_texts(metadata: Any) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    value = _first_constraint_value(metadata)
    if not value:
        return ()
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    return tuple(
        dict.fromkeys(
            item.strip().lower() for item in value if isinstance(item, str) and item.strip()
        )
    )


def _first_constraint_value(metadata: Mapping[str, Any]) -> Any:
    if metadata.get("constraints"):
        return metadata["constraints"]
    manifest_row = metadata.get("manifest_row")
    if isinstance(manifest_row, Mapping):
        if manifest_row.get("constraints"):
            return manifest_row["constraints"]
        nested = manifest_row.get("metadata")
        if isinstance(nested, Mapping) and nested.get("constraints"):
            return nested["constraints"]
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, Mapping) and nested_metadata.get("constraints"):
        return nested_metadata["constraints"]
    return None


def _contains_hint(texts: Sequence[str], hints: Sequence[str]) -> bool:
    for raw_text in texts:
        text = f" {_normalize_hint_text(raw_text)} "
        for raw_hint in hints:
            hint = f" {_normalize_hint_text(raw_hint)} "
            if hint in text:
                return True
    return False


def _normalize_hint_text(text: str) -> str:
    return " ".join(str(text).lower().translate(_HINT_TRANSLATION).split())
