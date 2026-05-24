"""Rollout schedule factory and protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle
from vrl.rollouts.orchestration.one_batch_overlap import OneBatchOverlapRolloutSchedule
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.orchestration.types import RolloutIteration, RolloutScheduleMode


class RolloutSchedule(Protocol):
    """Interface consumed by ``OnlineTrainer``."""

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration: ...

    async def after_train_step(self) -> dict[str, float]: ...

    def reset(self) -> None: ...


def build_rollout_schedule(
    config: Any,
    *,
    collector: Any,
    model: nn.Module,
    device: torch.device,
    weight_syncer: Any | None,
    sync_state_getter: Callable[[], dict[str, Any]] | None,
    weights_initialized: Callable[[], bool],
    set_weights_initialized: Callable[[bool], None],
) -> RolloutSchedule:
    """Build the RL rollout schedule selected by trainer config."""

    mode = RolloutScheduleMode(
        getattr(config, "mode", RolloutScheduleMode.STRICT_ON_POLICY.value),
    )
    max_pending = int(getattr(config, "max_pending_rollouts", 1))
    if max_pending != 1:
        raise ValueError(
            "rollout_orchestration.max_pending_rollouts must be 1 in this sprint",
        )

    lifecycle = RolloutLifecycle(
        collector=collector,
        model=model,
        device=device,
        weight_syncer=weight_syncer,
        sync_state_getter=sync_state_getter,
        weights_initialized=weights_initialized,
        set_weights_initialized=set_weights_initialized,
    )

    if mode is RolloutScheduleMode.STRICT_ON_POLICY:
        return StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)
    if mode is RolloutScheduleMode.ONE_BATCH_OVERLAP:
        return OneBatchOverlapRolloutSchedule(
            lifecycle=lifecycle,
            require_separate_gpus=bool(getattr(config, "require_separate_gpus", True)),
        )
    raise AssertionError(f"unreachable rollout schedule mode: {mode}")


__all__ = ["RolloutSchedule", "build_rollout_schedule"]
