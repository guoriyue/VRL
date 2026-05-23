"""Device movement helpers for trajectory tensor trees."""

from __future__ import annotations

from typing import Any


def move_value_to_device(value: Any, device: Any | None) -> Any:
    """Move tensor-like leaves in a nested value without requiring torch types."""

    if device is None:
        return value
    mover = getattr(value, "to", None)
    if callable(mover):
        try:
            return mover(device)
        except TypeError:
            return value
    if isinstance(value, dict):
        return {key: move_value_to_device(inner, device) for key, inner in value.items()}
    if isinstance(value, list):
        return [move_value_to_device(inner, device) for inner in value]
    if isinstance(value, tuple):
        return tuple(move_value_to_device(inner, device) for inner in value)
    return value


__all__ = ["move_value_to_device"]
