"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
)
from vrl.utils.stats import timed


class StrictOnPolicyRolloutSchedule:
    """Collect one rollout, train it, then sync weights after training."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
    ) -> None:
        self.lifecycle = lifecycle

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        del next_prompts
        # Schedule-level phases (weight init / driver offload / activate /
        # collect) and the collect call's own timings land in the caller's
        # bound stats. Whether the strategy can park at all was checked before
        # launch, and its park re-checks; the schedule does not ask again each
        # iteration.
        await self.lifecycle.ensure_initial_weights()
        policy_version = self.lifecycle.current_policy_version()

        # The schedule only announces the phase; who parks, activates,
        # releases, and restores (and in which order, under which failures)
        # is owned by the coordinator's phase manager.
        async with self.lifecycle.rollout_phase():
            with timed("rollout.collect_s"):
                batches = await self.lifecycle.collector.prepare_training_batches(
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                )

        return RolloutIteration(batches=batches)

    async def after_train_step(self) -> None:
        await self.lifecycle.sync_weights_after_train()

    def reset(self) -> None:
        """No-op reset; the schedule holds no resume-sensitive state."""

    async def shutdown(self) -> None:
        """Release the rollout pipeline; the coordinator owns the parking order."""

        await self.lifecycle.shutdown_collector_runtime()


__all__ = ["StrictOnPolicyRolloutSchedule"]
