"""Rollout schedule factory and protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from vrl.rollouts.orchestration.continuous import ContinuousRolloutSchedule
from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle
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
    if mode is not RolloutScheduleMode.CONTINUOUS and max_pending != 1:
        raise ValueError(
            "rollout_orchestration.max_pending_rollouts must be 1 unless "
            "mode='continuous'",
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
    if mode is RolloutScheduleMode.CONTINUOUS:
        return _build_continuous_schedule(config, lifecycle=lifecycle)
    raise AssertionError(f"unreachable rollout schedule mode: {mode}")


def _build_continuous_schedule(
    config: Any,
    *,
    lifecycle: RolloutLifecycle,
) -> ContinuousRolloutSchedule:
    """Translate ``rollout_orchestration.continuous`` config into the schedule.

    Reads fields via ``getattr`` to keep the rollout layer free of any
    ``vrl.trainers`` import (architecture boundary).
    """

    cont = getattr(config, "continuous", None)

    def field(name: str, default: Any) -> Any:
        return getattr(cont, name, default) if cont is not None else default

    return ContinuousRolloutSchedule(
        lifecycle=lifecycle,
        require_separate_gpus=bool(getattr(config, "require_separate_gpus", True)),
        max_inflight_groups=int(field("max_inflight_groups", 1)),
        max_ready_groups=int(field("max_ready_groups", 2)),
        max_ready_bytes_mb=int(field("max_ready_bytes_mb", 8192)),
        max_stale_policy_versions=int(field("max_stale_policy_versions", 0)),
        drop_policy=str(field("drop_policy", "drop_oldest_stale")),
        wait_timeout_s=float(field("wait_timeout_s", 300.0)),
        queue_poll_interval_s=float(field("queue_poll_interval_s", 0.05)),
        fail_fast_errors=int(field("fail_fast_errors", 3)),
    )


__all__ = ["RolloutSchedule", "build_rollout_schedule"]
