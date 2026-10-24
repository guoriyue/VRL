"""Rollout schedule factory and protocol."""

from __future__ import annotations

from typing import Any, Protocol

from vrl.rollouts.orchestration.types import (
    RolloutIteration,
)


class RolloutSchedule(Protocol):
    """Interface consumed by ``OnlineTrainer``.

    Phase timings, counters and gauges are written to the caller's bound
    stats (``vrl.utils.stats``), never returned.
    """

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration: ...

    async def after_train_step(self) -> None: ...

    def reset(self) -> None: ...

    async def shutdown(self) -> None: ...


__all__ = [
    "RolloutSchedule",
]
