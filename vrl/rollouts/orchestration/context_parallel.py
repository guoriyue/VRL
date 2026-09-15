"""Rollout collection for context-parallel training: collect once, share within the CP group."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.schedule import RolloutSchedule
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.distributed import ContextParallelGroups


def batch_fingerprint(batches: list[Any]) -> list[tuple[int, list[int], float]]:
    """A cheap identity of a batch list: per batch, sample count, group ids, reward sum.

    Enough to catch a peer holding a different or reordered sample set; not a
    checksum of the trajectory tensors (those ride on the same pickle).
    """

    return [
        (
            int(batch.rewards.shape[0]),
            [int(g) for g in batch.group_ids.reshape(-1).tolist()],
            round(float(batch.rewards.double().sum()), 6),
        )
        for batch in batches
    ]


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
        batches = iteration.batches if leader else list(payload[0])
        # Token sharding assumes every CP peer replays the same samples in the
        # same order; a divergence would not hang (slot counts stay balanced)
        # but would train on garbage silently, so agree on the batch identity
        # before anything downstream can consume it.
        fingerprints: list[Any] = [None] * self.groups.cp_size
        dist.all_gather_object(
            fingerprints, batch_fingerprint(batches), group=self.groups.object_group
        )
        if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
            raise RuntimeError(
                "context-parallel peers disagree on the rollout batches after the leader "
                f"broadcast; per-rank fingerprints: {fingerprints}",
            )
        if leader:
            return iteration
        return RolloutIteration(batches=batches, stats=iteration.stats)

    async def after_train_step(self) -> RolloutStats:
        return await self.inner.after_train_step()

    def reset(self) -> None:
        self.inner.reset()

    async def shutdown(self) -> None:
        await self.inner.shutdown()


__all__ = ["ContextParallelRolloutSchedule", "batch_fingerprint"]
