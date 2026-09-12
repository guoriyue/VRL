"""Shared tensor helpers for diffusion family runners and replay paths."""

from __future__ import annotations

from typing import Any

import torch


def expand_tensor_to_batch(
    value: torch.Tensor,
    batch_size: int,
    *,
    materialize: bool = False,
) -> torch.Tensor:
    """Match the leading batch dimension, broadcasting only singleton inputs.

    Matching inputs are returned unchanged. On expansion, materialize=True
    allocates independent contiguous rows; otherwise return a shared view.
    """
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            f"cannot broadcast tensor batch={value.shape[0]} to batch_size={batch_size}"
        )
    expanded = value.expand(batch_size, *value.shape[1:])
    return expanded.clone(memory_format=torch.contiguous_format) if materialize else expanded


def broadcast_singleton_replay_tensor(value: Any, batch_size: int) -> Any:
    """Broadcast a leading-1 batch dim up to ``batch_size`` (contiguous)."""
    if not isinstance(value, torch.Tensor) or value.shape[:1] != (1,) or batch_size == 1:
        return value
    return expand_tensor_to_batch(value, batch_size, materialize=True)


def replay_tensor(
    replay_tensors: dict[str, Any],
    batch_context: dict[str, Any],
    name: str,
) -> Any:
    """Prefer the recorded replay tensor; fall back to the batch context."""
    if name in replay_tensors:
        return replay_tensors[name]
    return batch_context[name]


def shared_replay_tensor(
    replay_tensors: dict[str, Any],
    batch_context: dict[str, Any],
    name: str,
) -> Any:
    """Resolve a replay tensor and slice its shared (first) row when batched."""
    value = replay_tensor(replay_tensors, batch_context, name)
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return value[:1]
    return value


__all__ = [
    "broadcast_singleton_replay_tensor",
    "expand_tensor_to_batch",
    "replay_tensor",
    "shared_replay_tensor",
]
