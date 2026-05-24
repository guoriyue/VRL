"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle, record_phase
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
    annotate_batch_context,
    build_rollout_iteration,
)


class StrictOnPolicyRolloutSchedule:
    """Collect one rollout, train it, then sync weights after training."""

    mode = RolloutScheduleMode.STRICT_ON_POLICY

    def __init__(self, *, lifecycle: RolloutLifecycle) -> None:
        self.lifecycle = lifecycle
        self.state = RolloutScheduleState()

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration:
        phase_times: dict[str, float] = {}
        await self.lifecycle.ensure_initial_weights(phase_times)
        rollout_id = self.state.rollout_id
        self.state.rollout_id += 1
        policy_version = self.lifecycle.current_policy_version()

        offloaded = self.lifecycle.offload_driver_model_for_rollout(phase_times)
        try:
            with record_phase(phase_times, "rollout.collect_s"):
                batches = await collect_prompt_batches(
                    collector=self.lifecycle.collector,
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                )
        finally:
            await self.lifecycle.release_rollout_runtime_memory(phase_times)
            if offloaded:
                self.lifecycle.restore_driver_model_after_rollout(phase_times)

        return annotate_batch_context(
            build_rollout_iteration(
                rollout_id=rollout_id,
                policy_version=policy_version,
                mode=self.mode,
                batches=batches,
                prompt_count=len(prompts),
                phase_times=phase_times,
            )
        )

    async def after_train_step(self) -> dict[str, float]:
        phase_times: dict[str, float] = {}
        await self.lifecycle.sync_weights_after_train(phase_times)
        return phase_times

    def reset(self) -> None:
        self.state.initialized = False
        self.state.pending_rollout = None


__all__ = ["StrictOnPolicyRolloutSchedule"]
