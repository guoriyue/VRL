"""Dependency-free value validation shared across VRL subsystems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


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


__all__ = ["require_int", "require_mapping_keys"]


def require_mapping_keys(
    value: Any,
    allowed: Iterable[str],
    *,
    what: str,
    complete: bool = True,
) -> dict[str, Any]:
    """Return ``value`` as a dict whose keys are drawn from ``allowed``.

    The fail-closed key check shared by persisted records and configuration
    blocks: a non-mapping is a ``TypeError``; an unknown key is a ``ValueError``
    naming the offending sets, and with ``complete=True`` (records) so is a
    missing one, so a schema drift between writer and reader surfaces at the
    read instead of as a default silently applied.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    expected = set(allowed)
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value)) if complete else []
    if missing or unknown:
        raise ValueError(f"invalid {what} fields: missing={missing} unknown={unknown}")
    return {name: value[name] for name in value}
