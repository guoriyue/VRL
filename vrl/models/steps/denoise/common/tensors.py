"""Shared tensor helpers for diffusion family runners and replay paths."""

from __future__ import annotations

from typing import Any

import torch

# The batch expansion is layer-neutral (the generation executor applies it to
# encoded prompt fields); it lives in vrl.utils and is re-exported here for the
# family runners that read it next to the replay helpers.
from vrl.utils.tensors import expand_tensor_to_batch


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
    "expand_tensor_to_batch",
    "replay_tensor",
    "shared_replay_tensor",
]
