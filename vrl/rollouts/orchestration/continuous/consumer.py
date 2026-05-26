"""Consumer that turns ready queue items into a trainer ``RolloutIteration``.

The consumer owns same-policy batch selection: it waits until one homogeneous
policy version has a full iteration worth of distinct groups, reassigns
contiguous ``group_ids`` so per-prompt advantage normalization stays correct,
and hands a standard ``RolloutIteration`` back to the schedule.
"""

from __future__ import annotations

import asyncio
import time

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    annotate_batch_context,
    build_rollout_iteration,
)


class ContinuousRolloutConsumer:
    """Drain same-policy ready groups into a trainer iteration."""

    def __init__(
        self,
        *,
        queue: ContinuousRolloutQueue,
        staleness: StalenessPolicy,
    ) -> None:
        self.queue = queue
        self.staleness = staleness

    async def drain_for_iteration(
        self,
        *,
        rollout_id: int,
        min_groups: int,
        current_version: int | None,
        mode: RolloutScheduleMode,
        wait_timeout_s: float,
        poll_interval_s: float,
    ) -> RolloutIteration:
        """Block until a homogeneous-version iteration is ready, then build it."""

        deadline = time.monotonic() + float(wait_timeout_s)
        wait_start = time.perf_counter()
        while True:
            selected = self.queue.select_iteration(
                min_groups=min_groups,
                current_version=current_version,
                staleness=self.staleness,
            )
            if selected is not None:
                version, items = selected
                wait_s = time.perf_counter() - wait_start
                return self._build_iteration(
                    rollout_id=rollout_id,
                    version=version,
                    items=items,
                    mode=mode,
                    current_version=current_version,
                    queue_wait_s=wait_s,
                )
            if time.monotonic() >= deadline:
                stats = self.queue.stats()
                raise TimeoutError(
                    "continuous rollout consumer timed out waiting for "
                    f"{min_groups} same-policy groups after {wait_timeout_s}s "
                    f"(queue={stats})",
                )
            await asyncio.sleep(poll_interval_s)

    def _build_iteration(
        self,
        *,
        rollout_id: int,
        version: int | None,
        items: list[ContinuousRolloutItem],
        mode: RolloutScheduleMode,
        current_version: int | None,
        queue_wait_s: float,
    ) -> RolloutIteration:
        batches: list[RolloutBatch] = []
        for index, item in enumerate(items):
            _assign_group_index(item.batch, index)
            batches.append(item.batch)

        staleness = self.staleness.staleness(version, current_version)
        item_age_s = max((item.age_s for item in items), default=0.0)
        phase_times = {"continuous.queue_wait_s": float(queue_wait_s)}

        iteration = build_rollout_iteration(
            rollout_id=rollout_id,
            policy_version=version,
            mode=mode,
            batches=batches,
            prompt_count=len(items),
            phase_times=phase_times,
        )
        iteration.metadata.update(
            {
                "consume_policy_version": (
                    None if current_version is None else int(current_version)
                ),
                "stale_policy_versions": (
                    None if staleness is None else int(staleness)
                ),
                "continuous_item_age_s": float(item_age_s),
            },
        )
        return annotate_batch_context(iteration)


def _assign_group_index(batch: RolloutBatch, index: int) -> None:
    """Force every sample in a single-group batch to one contiguous group id."""

    batch.group_ids = torch.full_like(batch.group_ids, int(index))
    if batch.trajectory is not None:
        trajectory_group_ids = getattr(batch.trajectory, "group_ids", None)
        if isinstance(trajectory_group_ids, torch.Tensor):
            batch.trajectory.group_ids = torch.full_like(
                trajectory_group_ids,
                int(index),
            )


__all__ = ["ContinuousRolloutConsumer"]
