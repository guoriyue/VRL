"""Shared config-conversion helpers (leaf: no domain imports)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def plain_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Deep-convert a typed/OmegaConf config (or Mapping) into a plain ``dict``.

    OmegaConf is checked first: a ``DictConfig`` is itself a ``Mapping``, so a
    shallow ``dict(value)`` would leave nested ``ListConfig``/``DictConfig``
    values intact, which downstream serialization (the Ray launch contract)
    rejects. ``to_container`` recurses to plain list/dict. Pydantic models use
    ``exclude_unset`` so omitted defaults stay absent while explicit false,
    zero, and null values retain their presence semantics.
    """

    try:
        from omegaconf import OmegaConf
    except Exception:
        OmegaConf = None  # type: ignore[assignment]

    if OmegaConf is not None and OmegaConf.is_config(value):
        raw = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
        if isinstance(raw, Mapping):
            return dict(raw)
        raise TypeError(f"{field_name} must be a mapping")

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        raw = model_dump(mode="python", exclude_unset=True)
        if isinstance(raw, Mapping):
            return dict(raw)
        raise TypeError(f"{field_name} must be a mapping")

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(f"{field_name} must be a mapping")


def require_exact_int(value: object, *, path: str, minimum: int | None = None) -> int:
    """Validate an exact-integer config boundary and return the value.

    Rejects ``bool`` (Python's ``bool`` is an ``int`` subclass) and any non-int,
    then an optional lower bound. ``path`` names the offending key in the error.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer (got {value!r})")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum} (got {value})")
    return value


def to_builtin_deep(value: Any) -> Any:
    """Deep-convert OmegaConf configs and nested Mapping/list/tuple to plain types.

    Use for config payloads that must serialize cleanly (e.g. the Ray launch
    contract). Tuples become lists; YAML-sourced configs never carry tuples, so
    this only matters for hand-built test values.
    """

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except Exception:
        return value

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): to_builtin_deep(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin_deep(inner) for inner in value]
    return value


def import_from_path(path: str) -> Any:
    """Load ``module:attribute`` or ``module.attribute`` import paths."""

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


__all__ = [
    "import_from_path",
    "plain_mapping",
    "require_exact_int",
    "to_builtin_deep",
]
