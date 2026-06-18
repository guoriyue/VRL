"""Shared config-conversion helpers (leaf: no domain imports)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def plain_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Deep-convert an OmegaConf config (or Mapping) into a plain ``dict``.

    OmegaConf is checked first: a ``DictConfig`` is itself a ``Mapping``, so a
    shallow ``dict(value)`` would leave nested ``ListConfig``/``DictConfig``
    values intact, which downstream serialization (the Ray launch contract)
    rejects. ``to_container`` recurses to plain list/dict.
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

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(f"{field_name} must be a mapping")


_MISSING = object()


def cfg_get(node: Any, key: str, default: Any = None) -> Any:
    """Read one key from a config node of unknown shape.

    Works across the mix of node types the config layer passes around — a
    ``Mapping``/``DictConfig`` (has ``.get``), a plain object/dataclass/namespace
    (attribute access), or a subscriptable. Tries ``.get`` -> ``[]`` -> attribute,
    returning ``default`` when ``node`` is ``None`` or the key is absent.
    """

    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


def cfg_path(node: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted ``a.b.c`` path through nested config nodes via :func:`cfg_get`."""

    for key in path.split("."):
        node = cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def to_builtin(value: Any) -> Any:
    """Shallow-unwrap an OmegaConf ``DictConfig``/``ListConfig`` to a plain dict/list.

    Anything else (including already-plain values) passes through unchanged. Use
    :func:`plain_mapping` instead when a deep, fully-recursive conversion is needed.
    """

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except Exception:
        return value

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
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
    "cfg_get",
    "cfg_path",
    "import_from_path",
    "plain_mapping",
    "to_builtin",
    "to_builtin_deep",
]
