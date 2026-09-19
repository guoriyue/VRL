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
    """Move tensor-like leaves, propagating failures from their ``to`` method.

    ``device=None`` leaves the tree untouched. Values without a callable ``to``
    are metadata and pass through; a failed device move is not a no-op.
    """

    if device is None:
        return value

    return map_tensor_tree(
        value,
        lambda leaf: leaf.to(device),
        is_leaf=lambda v: (
            callable(getattr(v, "to", None)) and not isinstance(v, (dict, list, tuple))
        ),
    )


def copy_tensor_tree_to_pinned_cpu(value: Any) -> Any:
    """Copy every CUDA tensor leaf into a pinned CPU buffer and wait once.

    The copies are queued ``non_blocking`` on the current stream and joined with a
    single ``torch.cuda.synchronize()`` before the tree is returned, so callers
    always receive readable host tensors. Non-CUDA tensors are detached and moved
    with ``Tensor.cpu()``. The generation worker uses this to hand a batch result
    across the Ray wire, and the per-request batch loop uses it to release each
    batch's GPU payload before producing the next.
    """

    import torch

    cuda_copies_pending = False

    def copy_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal cuda_copies_pending
        tensor = tensor.detach()
        if not tensor.is_cuda:
            return tensor.cpu()

        cpu_buffer = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        cpu_buffer.copy_(tensor, non_blocking=True)
        cuda_copies_pending = True
        return cpu_buffer

    copied = map_tensor_tree(
        value,
        copy_tensor_to_cpu,
        is_leaf=lambda item: isinstance(item, torch.Tensor),
    )
    if cuda_copies_pending:
        torch.cuda.synchronize()
    return copied


__all__ = ["copy_tensor_tree_to_pinned_cpu", "map_tensor_tree", "move_value_to_device"]
