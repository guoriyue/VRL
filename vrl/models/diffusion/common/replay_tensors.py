"""Replay-tensor batch helpers shared by Cosmos diffusion families.

These resolve per-sample replay tensors during eval/replay reconstruction:
broadcast a leading-1 batch dim, fall back to batch_context, and slice the
shared (CFG-invariant) row. They are byte-identical across the Cosmos
Predict2 and Predict2.5 ``restore_eval_state`` paths.

Note: the Anima family deliberately keeps its own variants — its
``_align_replay_tensor`` omits ``.contiguous()`` and its shared-tensor helper
has a different signature (no ``batch_context``). Those are not the same
function and are intentionally NOT consolidated here.
"""

from __future__ import annotations

from typing import Any

import torch


def align_replay_tensor(value: Any, batch_size: int) -> Any:
    """Broadcast a leading-1 batch dim up to ``batch_size`` (contiguous)."""
    if not isinstance(value, torch.Tensor) or value.shape[:1] != (1,) or batch_size == 1:
        return value
    return value.expand(batch_size, *value.shape[1:]).contiguous()


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
    "align_replay_tensor",
    "replay_tensor",
    "shared_replay_tensor",
]
