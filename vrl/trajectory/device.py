"""Tensor-tree traversal and device-movement helpers.

One walker serves every "apply f to tensor leaves of a nested payload" need
(device moves, CPU offload, pinned copies, masked selection). The codebase
carried four hand-rolled copies of this recursion that had already diverged
on container coverage — a missed container type here means tensors silently
skip the leaf op.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any


def map_tensor_tree(
    value: Any,
    leaf_fn: Callable[[Any], Any],
    *,
    is_leaf: Callable[[Any], bool],
) -> Any:
    """Apply ``leaf_fn`` to every leaf of a nested payload.

    Recurses through dicts, lists, tuples, and (non-type) dataclasses;
    anything else that is not a leaf passes through unchanged. ``is_leaf``
    is explicit because callers genuinely differ: torch-importing call
    sites match ``isinstance(_, Tensor)`` while torch-free modules duck-type.
    """

    if is_leaf(value):
        return leaf_fn(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: map_tensor_tree(
                getattr(value, field.name),
                leaf_fn,
                is_leaf=is_leaf,
            )
            for field in dataclasses.fields(value)
        }
        return type(value)(**payload)
    if isinstance(value, dict):
        return {
            key: map_tensor_tree(inner, leaf_fn, is_leaf=is_leaf) for key, inner in value.items()
        }
    if isinstance(value, list):
        return [map_tensor_tree(inner, leaf_fn, is_leaf=is_leaf) for inner in value]
    if isinstance(value, tuple):
        return tuple(map_tensor_tree(inner, leaf_fn, is_leaf=is_leaf) for inner in value)
    return value


def move_value_to_device(value: Any, device: Any | None) -> Any:
    """Move tensor-like leaves in a nested value without requiring torch types."""

    if device is None:
        return value

    def _move(leaf: Any) -> Any:
        try:
            return leaf.to(device)
        except TypeError:
            return leaf

    return map_tensor_tree(
        value,
        _move,
        is_leaf=lambda v: (
            callable(getattr(v, "to", None)) and not isinstance(v, (dict, list, tuple))
        ),
    )


__all__ = ["map_tensor_tree", "move_value_to_device"]
