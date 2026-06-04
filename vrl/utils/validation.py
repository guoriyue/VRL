"""Shared lightweight validation helpers."""

from __future__ import annotations

__all__ = ["require_string_tuple"]


def require_string_tuple(name: str, values: tuple[str, ...]) -> None:
    """Raise ValueError if any element of `values` is not a non-empty string."""
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain non-empty strings")
