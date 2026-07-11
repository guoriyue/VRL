"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.rollout_runtime import (
    RolloutRuntimeCoordinator,
    record_phase,
)
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
    annotate_batch_context,
    build_rollout_iteration,
)
from vrl.utils.stats import RolloutStats


class StrictOnPolicyRolloutSchedule:
    """Collect one rollout, train it, then sync weights after training."""

    mode = RolloutScheduleMode.STRICT_ON_POLICY

    def __init__(self, *, lifecycle: RolloutRuntimeCoordinator) -> None:
        self.lifecycle = lifecycle
        self.state = RolloutScheduleState()

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration:
        # Schedule-level phases (weight init / driver offload / activate / collect /
        # sync) are timed into a local dict via the lifecycle's record_phase;
        # the per-request collect stats accumulate into a typed RolloutStats.
        # Both are merged onto the iteration so nothing about the reported
        # breakdown changes.
        schedule_phases: dict[str, float] = {}
        stats = RolloutStats()
        await self.lifecycle.ensure_initial_weights(schedule_phases)
        rollout_id = self.state.rollout_id
        self.state.rollout_id += 1
        policy_version = self.lifecycle.current_policy_version()

        offloaded = self.lifecycle.offload_driver_model_for_rollout(schedule_phases)
        try:
            await self.lifecycle.activate_rollout_runtime(schedule_phases)
            with record_phase(schedule_phases, "rollout.collect_s"):
                batches = await collect_prompt_batches(
                    collector=self.lifecycle.collector,
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                    stats=stats,
                )
        finally:
            await self.lifecycle.offload_rollout_runtime_memory(schedule_phases)
            if offloaded:
                self.lifecycle.restore_driver_model_after_rollout(schedule_phases)

        stats.add_phases(schedule_phases)
        return annotate_batch_context(
            build_rollout_iteration(
                rollout_id=rollout_id,
                policy_version=policy_version,
                mode=self.mode,
                batches=batches,
                prompt_count=len(prompts),
                stats=stats,
            )
        )

    async def after_train_step(self) -> dict[str, float]:
        phase_times: dict[str, float] = {}
        await self.lifecycle.sync_weights_after_train(phase_times)
        return phase_times

    def reset(self) -> None:
        """No-op reset; the schedule holds no resume-sensitive state."""

    async def shutdown(self) -> None:
        """No-op shutdown; strict scheduling owns no background task."""


__all__ = ["StrictOnPolicyRolloutSchedule"]
