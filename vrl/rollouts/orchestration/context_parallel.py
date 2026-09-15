"""Rollout collection for context-parallel training: collect once, share within the CP group."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.schedule import RolloutSchedule
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.distributed import ContextParallelPeerGroup


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
    """Wrap a rank-local schedule so a CP group collects one sample set together.

    Every peer generates its own slice of the group's prompts on its own
    rollout engine (no card idles), scores it, and the peers all-gather the
    resulting batches over the CP group's CPU-capable group; each rank then
    trains on the same union in the same order, which is what sequence
    sharding inside the model assumes. Every rank drives the wrapped schedule
    (its rollout phase parks and restores the trainer shards, exports weights
    and syncs its engine, all of which are collectives every rank must join).
    """

    def __init__(self, inner: RolloutSchedule, *, groups: ContextParallelPeerGroup) -> None:
        self.inner = inner
        self.groups = groups

    def local_prompts(self, prompts: list[Any]) -> list[Any]:
        """This peer's contiguous slice of the group's prompts (last peer takes the remainder)."""

        per_peer, extra = divmod(len(prompts), self.groups.cp_size)
        start = self.groups.cp_rank * per_peer
        stop = start + per_peer + (extra if self.groups.cp_rank == self.groups.cp_size - 1 else 0)
        return list(prompts[start:stop])

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        import torch.distributed as dist

        iteration = await self.inner.next_iteration(
            self.local_prompts(list(prompts)),
            group_size=group_size,
            runtime_debug=runtime_debug,
            next_prompts=None if next_prompts is None else self.local_prompts(list(next_prompts)),
        )
        gathered: list[Any] = [None] * self.groups.cp_size
        dist.all_gather_object(gathered, iteration.batches, group=self.groups.object_group)
        batches = [batch for peer_batches in gathered for batch in peer_batches]
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
                "context-parallel peers disagree on the gathered rollout batches; "
                f"per-rank fingerprints: {fingerprints}",
            )
        return RolloutIteration(batches=batches, stats=iteration.stats)

    async def after_train_step(self) -> RolloutStats:
        return await self.inner.after_train_step()

    def reset(self) -> None:
        self.inner.reset()

    async def shutdown(self) -> None:
        await self.inner.shutdown()


__all__ = ["ContextParallelRolloutSchedule", "batch_fingerprint"]
