"""Trainer Protocol — training loop contract."""

from __future__ import annotations

from typing import Protocol

from vrl.algorithms.types import TrainStepMetrics


class Trainer(Protocol):
    """Structural interface for RL trainers."""

    async def step(self) -> TrainStepMetrics:
        """Run one training step (collect → reward → advantage → update)."""
        ...

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of trainer state."""
        ...

    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        """Restore trainer state from a snapshot."""
        ...
