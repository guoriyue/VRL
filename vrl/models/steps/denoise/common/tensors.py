"""Shared tensor helpers for diffusion family runners and replay paths."""

from __future__ import annotations

from typing import Any

import torch


def require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    """Fail fast when a CFG branch input that must exist is ``None``."""
    if value is None:
        raise ValueError(f"CFG branch requires {name}")
    return value


# -- replay-tensor batch helpers (Cosmos Predict2 / Predict2.5 / Anima) ------
#
# These resolve per-sample replay tensors during eval/replay reconstruction:
# broadcast a leading-1 batch dim, fall back to batch_context, and slice the
# shared (CFG-invariant) row, shared across the Cosmos-family
# ``restore_eval_state`` paths.
#
# Note: Wan deliberately keeps its own ``_batch_align_tensor`` — it is the
# STRICT variant that raises on an incompatible batch dim instead of passing
# the tensor through, guarding I2V conditioning shapes. That is a different
# contract and is intentionally NOT consolidated here.


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
    "require_tensor",
    "shared_replay_tensor",
]
