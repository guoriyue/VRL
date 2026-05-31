"""Shared config-conversion helpers (leaf: no domain imports)."""

from __future__ import annotations

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


__all__ = ["plain_mapping"]
