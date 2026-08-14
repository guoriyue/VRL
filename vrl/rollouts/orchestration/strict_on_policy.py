"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.prompt_collection import collect_prompt_groups
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import (
    RewardCollectionMode,
    RolloutIteration,
)
from vrl.rollouts.stats import RolloutStats


class StrictOnPolicyRolloutSchedule:
    """Collect one rollout, train it, then sync weights after training."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        reward_mode: RewardCollectionMode | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        # None = derive the arm from the collector capability. A forced arm is an
        # acceptance-measurement control; prompt collection checks that it cannot
        # grant per-group execution to an incapable collector.
        self.reward_mode = reward_mode

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        del next_prompts
        # One accumulator for the whole iteration: schedule-level phases (weight
        # init / driver offload / activate / collect / sync) and the per-request
        # collect stats land in the same typed object.
        stats = RolloutStats()
        # Capability validation happens before weight export or collection so a
        # distributed strategy cannot enter a shared-GPU phase by pretending the
        # single-process parking implementation applies to it.
        self.lifecycle.validate_training_state_parking()
        await self.lifecycle.ensure_initial_weights(stats)
        policy_version = self.lifecycle.current_policy_version()

        # The schedule only announces the phase; who parks, activates,
        # releases, and restores (and in which order, under which failures)
        # is owned by the coordinator's phase manager.
        batches: list[Any] | None = None
        async with self.lifecycle.rollout_phase(stats):
            with stats.phase("rollout.collect_s"):
                batches = await collect_prompt_groups(
                    collector=self.lifecycle.collector,
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                    stats=stats,
                    reward_mode=self.reward_mode,
                )
        assert batches is not None

        return RolloutIteration(
            batches=batches,
            stats=stats,
        )

    async def after_train_step(self) -> RolloutStats:
        stats = RolloutStats()
        await self.lifecycle.sync_weights_after_train(stats)
        return stats

    def reset(self) -> None:
        """No-op reset; the schedule holds no resume-sensitive state."""

    async def shutdown(self) -> None:
        """Release the rollout pipeline; the coordinator owns the parking order."""

        await self.lifecycle.shutdown_collector_runtime()


__all__ = ["StrictOnPolicyRolloutSchedule"]
