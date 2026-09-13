"""Strict on-policy rollout schedule."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.collector.core import RewardCollectionMode
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import (
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
        async with self.lifecycle.rollout_phase(stats):
            with stats.phase("rollout.collect_s"):
                batches = await self.lifecycle.collector.prepare_training_batches(
                    prompts=list(prompts),
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                    stats=stats,
                    reward_mode=self.reward_mode,
                )

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


class ContextParallelStrictRolloutSchedule:
    """Explicit disjoint-GPU strict owner adapter; not selected by the factory.

    All training ranks call every method in order, before strategy shutdown.
    Only CP leaders have an owner schedule. Owner weight snapshot getters must
    be rank-local (CP parameters are replicated), never DDP/FSDP collectives.
    """

    def __init__(
        self, *, owner: StrictOnPolicyRolloutSchedule | None, groups: Any, spool_dir: Any
    ):
        if (owner is not None) != (groups.cp_rank == 0):
            raise ValueError("only CP leaders may own a strict rollout schedule")
        self.owner, self.groups, self.spool_dir = owner, groups, spool_dir

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        from vrl.rollouts.orchestration.schedule import collect_context_parallel_iteration

        async def collect():
            if self.owner.lifecycle.requires_training_state_parking():
                raise ValueError("CP strict owner requires disjoint rollout/reward GPUs")
            return await self.owner.next_iteration(
                prompts,
                group_size=group_size,
                runtime_debug=runtime_debug,
                next_prompts=next_prompts,
            )

        return await collect_context_parallel_iteration(
            collect if self.owner is not None else None,
            groups=self.groups,
            spool_dir=self.spool_dir,
        )

    def _agree(self, result: Any, error: str | None) -> Any:
        import torch.distributed as dist

        message = [(result, error)]
        dist.broadcast_object_list(
            message, src=dist.get_global_rank(self.groups.cp_group, 0), group=self.groups.cp_group
        )
        errors = [None] * (self.groups.cp_size * self.groups.dp_size)
        dist.all_gather_object(errors, message[0][1])
        if any(item is not None for item in errors):
            raise RuntimeError(f"CP strict owner operation failed: {errors}")
        return message[0][0]

    async def _owner_event(self, name: str) -> Any:
        result = error = None
        if self.owner is not None:
            try:
                result = await getattr(self.owner, name)()
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
        return self._agree(result, error)

    async def after_train_step(self) -> RolloutStats:
        return await self._owner_event("after_train_step")

    def reset(self) -> None:
        error = None
        if self.owner is not None:
            try:
                self.owner.reset()
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
        self._agree(None, error)

    async def shutdown(self) -> None:
        await self._owner_event("shutdown")


__all__ = ["ContextParallelStrictRolloutSchedule", "StrictOnPolicyRolloutSchedule"]
