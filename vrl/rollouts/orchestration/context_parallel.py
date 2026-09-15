"""Rollout collection for context-parallel training: collect once, share within the CP group."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.schedule import RolloutSchedule
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.distributed import ContextParallelGroups


class ContextParallelRolloutSchedule:
    """Wrap a rank-local schedule so one rank of each CP group collects for all.

    Every rank still drives the wrapped schedule (its rollout phase parks and
    restores the trainer shards, exports weights, and syncs its rollout engine,
    all of which are FSDP collectives every rank must join); followers merely
    pass no prompts, so they generate nothing. The leader's batches are then
    broadcast over the CP group's CPU-capable group, and each rank returns the
    same iteration, which is what sequence sharding inside the model assumes.
    """

    def __init__(self, inner: RolloutSchedule, *, groups: ContextParallelGroups) -> None:
        self.inner = inner
        self.groups = groups

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        import torch.distributed as dist

        leader = self.groups.cp_rank == 0
        iteration = await self.inner.next_iteration(
            list(prompts) if leader else [],
            group_size=group_size,
            runtime_debug=runtime_debug,
            next_prompts=next_prompts if leader else None,
        )
        payload: list[Any] = [iteration.batches if leader else None]
        dist.broadcast_object_list(
            payload, src=self.groups.leader_rank, group=self.groups.object_group
        )
        if leader:
            return iteration
        return RolloutIteration(batches=list(payload[0]), stats=iteration.stats)

    async def after_train_step(self) -> RolloutStats:
        return await self.inner.after_train_step()

    def reset(self) -> None:
        self.inner.reset()

    async def shutdown(self) -> None:
        await self.inner.shutdown()


__all__ = ["ContextParallelRolloutSchedule"]
