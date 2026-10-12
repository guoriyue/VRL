"""Rollout schedule factory and protocol."""

from __future__ import annotations

from typing import Any, Protocol

from vrl.rollouts.orchestration.types import (
    RolloutIteration,
)
from vrl.rollouts.stats import RolloutStats


class RolloutSchedule(Protocol):
    """Interface consumed by ``OnlineTrainer``."""

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration: ...

    async def after_train_step(self) -> RolloutStats: ...

    def reset(self) -> None: ...

    async def shutdown(self) -> None: ...


__all__ = [
    "RolloutSchedule",
]
