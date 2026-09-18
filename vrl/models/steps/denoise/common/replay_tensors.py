"""Replay-tensor lookups shared by the diffusion families' ``restore_eval_state``."""

from __future__ import annotations

from typing import Any

import torch


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


__all__ = ["replay_tensor", "shared_replay_tensor"]
