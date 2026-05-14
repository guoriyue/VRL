"""Trainer replay policy protocol.

This contract is intentionally smaller than a rollout policy. Rollout policies
may own prompt encoders, decoders, schedulers, and sampling helpers. Trainer
replay only needs the trainable forward path plus adapter state handling.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReplayPolicy(Protocol):
    """Minimal protocol consumed by trainer-side replay and weight sync."""

    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        """Recompute model outputs for one recorded trajectory slice."""
        ...

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Temporarily disable LoRA/adapters for a reference-policy pass."""
        ...

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load the flattened trainable-state payload pushed to rollout workers."""
        ...


__all__ = ["ReplayPolicy"]
