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
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    ContinuousRolloutProducerState,
    ContinuousRolloutSettings,
)
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause


class ContinuousRolloutConsumer:
    """Drain same-policy ready groups into a trainer iteration."""

    def __init__(
        self,
        *,
        queue: ContinuousRolloutQueue,
        staleness: StalenessPolicy,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.queue = queue
        self.staleness = staleness
        # Fresh-error count (with zero fresh completions) that ends the wait
        # early with the producer's root cause. 0 disables fail-fast. Range
        # validated once at the config boundary; trusted here.
        self.fail_fast_errors = settings.fail_fast_errors

    async def drain_for_iteration(
        self,
        *,
        min_groups: int,
        current_version: int | None,
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
        demand_snapshot = self.queue.snapshot()
        ready_groups_at_demand = len({item.group_slot for item in demand_snapshot})
        ready_batches_at_demand = len({item.batch_id for item in demand_snapshot})
        start_completed = producer_state.completed_count if producer_state else 0
        start_errors = producer_state.error_count if producer_state else 0
        while True:
            self._fail_fast_if_producer_stalled(
                producer_state,
                start_completed=start_completed,
                start_errors=start_errors,
            )
            selected = self._select_iteration(
                min_groups=min_groups,
                current_version=current_version,
            )
            if selected is not None:
                version, items = selected
                wait_s = time.perf_counter() - wait_start
                return self._build_iteration(
                    version=version,
                    items=items,
                    current_version=current_version,
                    queue_wait_s=wait_s,
                    ready_groups_at_demand=ready_groups_at_demand,
                    ready_batches_at_demand=ready_batches_at_demand,
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
            if (
                find_error_cause(
                    producer_state.fatal_error,
                    TerminalRuntimeError,
                )
                is not None
            ):
                raise producer_state.fatal_error
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
                f"last_error={producer_state.last_error})"
            )
        return message

    def validate_ready_versions(self, *, current_version: int | None) -> None:
        """Fail when a ready item falls outside the trainable version window."""

        if current_version is None:
            return
        for item in self.queue.snapshot():
            version = item.rollout_policy_version
            if self.staleness.is_future(version, current_version):
                raise RuntimeError(
                    "continuous queue item is newer than the trainer policy "
                    f"(item={version}, trainer={current_version}); weight-sync "
                    "barrier invariant violated",
                )
            if self.staleness.too_stale(version, current_version):
                raise RuntimeError(
                    "continuous ready prompt batch is older than the policy window "
                    f"(item={version}, trainer={current_version})",
                )

    def _select_iteration(
        self,
        *,
        min_groups: int,
        current_version: int | None,
    ) -> tuple[int | None, list[ContinuousRolloutItem]] | None:
        """Pop one complete, distinct-group, homogeneous-version batch."""

        # min_groups == len(prompts); the owner already rejected empty prompt
        # lists at the API boundary, so no re-check here.
        self.validate_ready_versions(current_version=current_version)

        items = self.queue.snapshot()
        versions = {item.rollout_policy_version for item in items}
        if len(versions) > 1:
            raise RuntimeError(
                "continuous ready prompt batch mixes policy versions "
                f"{sorted(versions, key=lambda version: -1 if version is None else version)}",
            )
        batch_ids = {item.batch_id for item in items}
        if len(batch_ids) > 1:
            raise RuntimeError(
                f"continuous ready prompt batch mixes batch identities {sorted(batch_ids)}",
            )
        group_slots = [item.group_slot for item in items]
        if len(group_slots) != len(set(group_slots)):
            raise RuntimeError(
                "continuous ready prompt batch contains duplicate group slots",
            )
        if len(items) < min_groups:
            return None
        if len(items) > min_groups:
            raise RuntimeError(
                "continuous ready prompt batch exceeds its expected group count "
                f"(ready={len(items)}, expected={min_groups})",
            )
        self.queue.remove(items)
        return items[0].rollout_policy_version, items

    def _build_iteration(
        self,
        *,
        version: int | None,
        items: list[ContinuousRolloutItem],
        current_version: int | None,
        queue_wait_s: float,
        ready_groups_at_demand: int,
        ready_batches_at_demand: int,
    ) -> RolloutIteration:
        batches: list[RolloutBatch] = []
        for index, item in enumerate(items):
            # Each queued item is one prompt group. Reassign contiguous ids after
            # completion-order selection so advantage normalization cannot join
            # different prompts or retain sparse producer slot ids.
            item.batch.group_ids = torch.full_like(item.batch.group_ids, int(index))
            batches.append(item.batch)

        staleness = self.staleness.staleness(version, current_version)
        item_age_s = max((item.age_s for item in items), default=0.0)
        max_attempt = max(item.attempt for item in items)
        stats = RolloutStats()
        stats.add_phase("continuous.queue_wait_s", float(queue_wait_s))
        # Merge the per-item collect stats (each collect call attached its
        # timings to exactly one item) so the iteration reports cumulative
        # generation/reward/build time, matching the strict schedule's
        # one-call-per-iteration accounting.
        for item in items:
            stats.merge(item.stats)
        stats.observe_gauges(
            {
                "continuous.consume_policy_version": float(
                    0 if current_version is None else current_version
                ),
                "continuous.rollout_policy_version": float(0 if version is None else version),
                "continuous.stale_policy_versions": float(0 if staleness is None else staleness),
                "continuous.item_age_s": float(item_age_s),
                "continuous.ready_groups_at_demand": float(ready_groups_at_demand),
                # display/provenance-only in this sprint: identity in the
                # metric row (selection consumes batch_id from Sprint 2 on).
                "continuous.batch_id": float(items[0].batch_id),
                "continuous.active_batches": float(ready_batches_at_demand),
                # >1 means this update contains retried work (provenance for
                # correlating reward/gradient anomalies with retries).
                "continuous.max_attempt": float(max_attempt),
            },
        )
        return RolloutIteration(batches=batches, stats=stats)


__all__ = ["ContinuousRolloutConsumer"]
