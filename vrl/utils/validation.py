"""Dependency-free value validation shared across VRL subsystems."""

from __future__ import annotations


def require_int(value: object, *, path: str, minimum: int | None = None) -> int:
    """Validate an integer value without coercion and return it.

    Rejects ``bool`` (Python's ``bool`` is an ``int`` subclass) and any non-int,
    then an optional lower bound. ``path`` names the offending field in the error.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer (got {value!r})")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum} (got {value})")
    return value


__all__ = ["require_int"]
