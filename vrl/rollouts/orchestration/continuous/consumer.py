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
from vrl.rollouts.orchestration.continuous.scheduler import RolloutScheduler
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    ContinuousRolloutProducerState,
)
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    build_rollout_iteration,
)
from vrl.rollouts.stats import RolloutStats


class ContinuousRolloutConsumer:
    """Drain same-policy ready groups into a trainer iteration."""

    def __init__(
        self,
        *,
        queue: ContinuousRolloutQueue,
        scheduler: RolloutScheduler,
        # No default: ContinuousRolloutConfig (vrl.trainers.core.types) is the
        # single source of this default; the owner passes settings.fail_fast_errors.
        fail_fast_errors: int,
    ) -> None:
        self.queue = queue
        # The scheduler owns the version/staleness decisions; the consumer drives
        # its select and reuses its policy only to report the staleness number.
        self.scheduler = scheduler
        # Fresh-error count (with zero fresh completions) that ends the wait
        # early with the producer's root cause. 0 disables fail-fast.
        self.fail_fast_errors = max(0, int(fail_fast_errors))

    async def drain_for_iteration(
        self,
        *,
        rollout_id: int,
        min_groups: int,
        current_version: int | None,
        mode: RolloutScheduleMode,
        wait_timeout_s: float,
        poll_interval_s: float,
        producer_state: ContinuousRolloutProducerState | None = None,
    ) -> RolloutIteration:
        """Block until a homogeneous-version iteration is ready, then build it.

        ``producer_state`` lets the wait surface the background producer's
        health: a persistent generation/reward failure ends the wait early with
        the producer's root cause instead of an opaque timeout, and the timeout
        message (when reached) includes the producer's last error and counters.
        """

        deadline = time.monotonic() + float(wait_timeout_s)
        wait_start = time.perf_counter()
        ready_groups_at_demand = len(
            {item.group_key for item in self.queue.snapshot()},
        )
        start_completed = producer_state.completed_count if producer_state else 0
        start_errors = producer_state.error_count if producer_state else 0
        while True:
            self._fail_fast_if_producer_stalled(
                producer_state,
                start_completed=start_completed,
                start_errors=start_errors,
            )
            selected = self.scheduler.select_iteration(
                self.queue,
                min_groups=min_groups,
                current_version=current_version,
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
                    ready_groups_at_demand=ready_groups_at_demand,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    self._timeout_message(min_groups, wait_timeout_s, producer_state),
                )
            await asyncio.sleep(poll_interval_s)

    def _fail_fast_if_producer_stalled(
        self,
        producer_state: ContinuousRolloutProducerState | None,
        *,
        start_completed: int,
        start_errors: int,
    ) -> None:
        """Raise the producer's root cause if every attempt is failing.

        Triggers only when, since this wait started, the producer has logged
        ``fail_fast_errors`` failures and produced *zero* completions — a
        systemic generation/reward failure rather than a transient blip. Slow
        failures that never reach the threshold are still covered by the
        enriched timeout message.
        """

        if producer_state is None:
            return
        if producer_state.fatal_error is not None:
            raise RuntimeError(
                "continuous rollout producer control loop failed",
            ) from producer_state.fatal_error
        if self.fail_fast_errors == 0:
            return
        fresh_errors = producer_state.error_count - start_errors
        fresh_completions = producer_state.completed_count - start_completed
        if fresh_completions == 0 and fresh_errors >= self.fail_fast_errors:
            raise RuntimeError(
                "continuous rollout producer is failing every generation while "
                f"the consumer waits: {fresh_errors} errors and 0 completions "
                f"since wait start (submitted={producer_state.submitted_count}, "
                f"completed={producer_state.completed_count}, "
                f"errors={producer_state.error_count}); "
                f"last_error={producer_state.last_error}",
            )

    def _timeout_message(
        self,
        min_groups: int,
        wait_timeout_s: float,
        producer_state: ContinuousRolloutProducerState | None,
    ) -> str:
        stats = self.queue.stats()
        message = (
            "continuous rollout consumer timed out waiting for "
            f"{min_groups} same-policy groups after {wait_timeout_s}s "
            f"(queue={stats})"
        )
        if producer_state is not None:
            message += (
                f" (producer: submitted={producer_state.submitted_count}, "
                f"completed={producer_state.completed_count}, "
                f"errors={producer_state.error_count}, "
                f"inflight={producer_state.inflight_count}, "
                f"last_error={producer_state.last_error})"
            )
        return message

    def _build_iteration(
        self,
        *,
        rollout_id: int,
        version: int | None,
        items: list[ContinuousRolloutItem],
        mode: RolloutScheduleMode,
        current_version: int | None,
        queue_wait_s: float,
        ready_groups_at_demand: int,
    ) -> RolloutIteration:
        batches: list[RolloutBatch] = []
        for index, item in enumerate(items):
            _assign_group_index(item.batch, index)
            batches.append(item.batch)

        staleness = self.scheduler.staleness.staleness(version, current_version)
        item_age_s = max((item.age_s for item in items), default=0.0)
        stats = RolloutStats()
        stats.add_phase("continuous.queue_wait_s", float(queue_wait_s))
        # Merge the per-item collect stats (each collect call attached its
        # timings to exactly one item) so the iteration reports cumulative
        # generation/reward/build time, matching the strict schedule's
        # one-call-per-iteration accounting.
        for item in items:
            stats.merge(item.stats)

        iteration = build_rollout_iteration(
            rollout_id=rollout_id,
            policy_version=version,
            mode=mode,
            batches=batches,
            prompt_count=len(items),
            stats=stats,
        )
        iteration.metadata.update(
            {
                "consume_policy_version": (
                    None if current_version is None else int(current_version)
                ),
                "stale_policy_versions": (None if staleness is None else int(staleness)),
                "continuous_item_age_s": float(item_age_s),
                "continuous_ready_groups_at_demand": int(ready_groups_at_demand),
            },
        )
        return iteration.annotate_batch_context()


def _assign_group_index(batch: RolloutBatch, index: int) -> None:
    """Force every sample in a single-group batch to one contiguous group id."""

    batch.group_ids = torch.full_like(batch.group_ids, int(index))


__all__ = ["ContinuousRolloutConsumer"]
