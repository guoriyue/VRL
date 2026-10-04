"""Consumer that turns a completed prompt batch into a trainer ``RolloutIteration``.

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
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutProducerState,
    ContinuousRolloutSettings,
    PromptBatch,
    ScoredRollout,
)
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause


class ContinuousRolloutConsumer:
    """Drain same-policy ready groups into a trainer iteration."""

    def __init__(
        self,
        *,
        staleness: StalenessPolicy,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.staleness = staleness
        # Fresh-error count (with zero fresh completions) that ends the wait
        # early with the producer's root cause. 0 disables fail-fast. Range
        # validated once at the config boundary; trusted here.
        self.fail_fast_errors = settings.fail_fast_errors

    async def collect_iteration(
        self,
        *,
        prompt_batch: PromptBatch,
        current_policy_version: int | None,
        wait_timeout_s: float,
        poll_interval_s: float,
        producer_state: ContinuousRolloutProducerState | None = None,
    ) -> RolloutIteration:
        """Block until a homogeneous-version iteration is ready, then build it.

        The owner supplies the producer's installed batch, including its
        prompt-ordered result slots.

        ``producer_state`` lets the wait surface the background producer's
        health: a persistent generation/reward failure ends the wait early with
        the producer's root cause instead of an opaque timeout, and the timeout
        message (when reached) includes the producer's last error and counters.
        """

        # Both waits were validated by ContinuousRolloutConfig.
        deadline = time.monotonic() + wait_timeout_s
        wait_start = time.perf_counter()
        ready_groups_at_demand = sum(item is not None for item in prompt_batch.results)
        start_completed = producer_state.completed_count if producer_state else 0
        start_errors = producer_state.error_count if producer_state else 0
        while True:
            self._fail_fast_if_producer_stalled(
                producer_state,
                start_completed=start_completed,
                start_errors=start_errors,
            )
            self.validate_ready_versions(
                prompt_batch=prompt_batch,
                current_policy_version=current_policy_version,
            )
            items = [item for item in prompt_batch.results if item is not None]
            if len(items) == len(prompt_batch.prompts):
                prompt_batch.results[:] = [None] * len(prompt_batch.prompts)
                wait_s = time.perf_counter() - wait_start
                return self._build_iteration(
                    items=items,
                    current_policy_version=current_policy_version,
                    queue_wait_s=wait_s,
                    ready_groups_at_demand=ready_groups_at_demand,
                )
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                ready_groups = sum(item is not None for item in prompt_batch.results)
                oldest_age = max(
                    (item.age_s for item in prompt_batch.results if item is not None),
                    default=0.0,
                )
                message = (
                    "continuous rollout consumer timed out waiting for "
                    f"{len(prompt_batch.prompts)} same-policy groups after {wait_timeout_s}s "
                    f"(prompt_batch_id={prompt_batch.batch_id}, "
                    f"ready_groups={ready_groups}/{len(prompt_batch.prompts)}, "
                    f"current_policy_version={current_policy_version}, "
                    f"oldest_item_age_s={oldest_age})"
                )
                if producer_state is not None:
                    message += (
                        f" (producer: submitted={producer_state.submitted_count}, "
                        f"completed={producer_state.completed_count}, "
                        f"errors={producer_state.error_count}, "
                        f"last_error={producer_state.last_error})"
                    )
                raise TimeoutError(message)
            await asyncio.sleep(min(poll_interval_s, remaining_s))

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

    def validate_ready_versions(
        self,
        *,
        prompt_batch: PromptBatch,
        current_policy_version: int | None,
    ) -> None:
        """Fail when a ready item falls outside the trainable version window."""

        for item in prompt_batch.results:
            if item is None:
                continue
            version = item.rollout_policy_version
            version_lag = self.staleness.staleness(version, current_policy_version)
            if version_lag is None:
                continue
            if version_lag < 0:
                raise RuntimeError(
                    "continuous scored result is newer than the trainer policy "
                    f"(item={version}, trainer={current_policy_version}); weight-sync "
                    "barrier invariant violated",
                )
            if version_lag > self.staleness.max_stale_policy_versions:
                raise RuntimeError(
                    "continuous ready prompt batch is older than the policy window "
                    f"(item={version}, trainer={current_policy_version})",
                )

    def _build_iteration(
        self,
        *,
        items: list[ScoredRollout],
        current_policy_version: int | None,
        queue_wait_s: float,
        ready_groups_at_demand: int,
    ) -> RolloutIteration:
        batches: list[RolloutBatch] = []
        for index, item in enumerate(items):
            # Each result slot is one prompt group. Reassign contiguous ids after
            # prompt-order selection so advantage normalization cannot join
            # different prompts or retain sparse producer slot ids.
            item.batch.group_ids = torch.full_like(item.batch.group_ids, index)
            batches.append(item.batch)

        version = items[0].rollout_policy_version
        staleness = self.staleness.staleness(version, current_policy_version)
        item_age_s = max(item.age_s for item in items)
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
                    0 if current_policy_version is None else current_policy_version
                ),
                "continuous.rollout_policy_version": float(0 if version is None else version),
                "continuous.stale_policy_versions": float(0 if staleness is None else staleness),
                "continuous.item_age_s": float(item_age_s),
                "continuous.ready_groups_at_demand": float(ready_groups_at_demand),
                # Report the same batch identity used to select this iteration.
                "continuous.batch_id": float(items[0].batch_id),
            },
        )
        return RolloutIteration(batches=batches, stats=stats)


__all__ = ["ContinuousRolloutConsumer"]
