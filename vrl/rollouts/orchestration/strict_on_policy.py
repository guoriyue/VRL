"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import (
    RewardCollectionMode,
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
    build_rollout_iteration,
)
from vrl.rollouts.stats import RolloutStats


class RolloutPhaseCleanupError(RuntimeError):
    """A rollout phase failed and one or more terminal cleanup steps also failed."""

    def __init__(
        self,
        root_cause: BaseException,
        cleanup_errors: list[BaseException],
    ) -> None:
        self.root_cause = root_cause
        self.cleanup_errors = tuple(cleanup_errors)
        cleanup = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
        super().__init__(
            f"rollout phase root cause: {type(root_cause).__name__}: {root_cause}; "
            f"terminal cleanup failures: {cleanup}",
        )


class StrictOnPolicyRolloutSchedule:
    """Collect one rollout, train it, then sync weights after training."""

    mode = RolloutScheduleMode.STRICT_ON_POLICY

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
        self.state = RolloutScheduleState()

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
        rollout_id = self.state.rollout_id
        self.state.rollout_id += 1
        policy_version = self.lifecycle.current_policy_version()

        parked = self.lifecycle.park_training_state_for_rollout(stats)
        batches: list[Any] | None = None
        phase_error: BaseException | None = None
        try:
            await self.lifecycle.activate_rollout_runtime(stats)
            with stats.phase("rollout.collect_s"):
                batches = await collect_prompt_batches(
                    collector=self.lifecycle.collector,
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                    stats=stats,
                    reward_mode=self.reward_mode,
                )
        except BaseException as error:
            phase_error = error

        cleanup_errors: list[BaseException] = []
        rollout_memory_released = False
        try:
            await self.lifecycle.offload_rollout_runtime_memory(stats)
            rollout_memory_released = True
        except BaseException as error:
            cleanup_errors.append(error)
        # Restoring the trainer is safe only after rollout memory ownership was
        # handed off successfully. If parking or its residual-memory check
        # fails, the rollout GPU state is unknown; terminal shutdown must tear
        # down the rollout runtime before the strategy restores training state.
        if parked and rollout_memory_released:
            try:
                self.lifecycle.restore_training_state_after_rollout(stats)
            except BaseException as error:
                cleanup_errors.append(error)

        if phase_error is not None:
            if cleanup_errors:
                raise RolloutPhaseCleanupError(phase_error, cleanup_errors) from phase_error
            raise phase_error
        if cleanup_errors:
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            raise RolloutPhaseCleanupError(
                cleanup_errors[0], cleanup_errors[1:]
            ) from cleanup_errors[0]
        assert batches is not None

        return build_rollout_iteration(
            rollout_id=rollout_id,
            policy_version=policy_version,
            mode=self.mode,
            batches=batches,
            prompt_count=len(prompts),
            stats=stats,
        ).annotate_batch_context()

    async def after_train_step(self) -> RolloutStats:
        stats = RolloutStats()
        await self.lifecycle.sync_weights_after_train(stats)
        return stats

    def reset(self) -> None:
        """No-op reset; the schedule holds no resume-sensitive state."""

    async def shutdown(self) -> None:
        """Park shared trainer state before releasing the rollout pipeline."""

        # A shared in-process reward may be asleep in a CuMem pool. Its terminal
        # shutdown wakes those pages before dropping the model, so the trainer
        # must yield the physical GPU first just as it does for a rollout phase.
        # The top-level strategy owner restores only after this shutdown proves
        # that every shared rollout/reward owner was released.
        self.lifecycle.validate_training_state_parking()
        # Shutdown reports no timings, but parking stays unconditional; a
        # throwaway accumulator keeps the five recording sites branch-free.
        self.lifecycle.park_training_state_for_rollout(RolloutStats())
        await self.lifecycle.shutdown_collector_runtime()


__all__ = ["RolloutPhaseCleanupError", "StrictOnPolicyRolloutSchedule"]
