"""Owner-loop continuous rollout producer.

A background ``asyncio`` task keeps a bounded number of ``RolloutCollector``
collect jobs in flight for one finite prompt batch, stamps each completed group
with the batch's policy version, and stores it in that batch's result slot. The heavy
generation work is dispatched by the collector (e.g. to remote Ray generation
actors), so this loop only schedules and harvests — that is enough to overlap
rollout with training on a cross-node setup.

The producer never computes advantages, calls the evaluator/algorithm, or
touches the optimizer; it owns rollout *production cadence* only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from typing import Any

import torch

from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutProducerState,
    ContinuousRolloutSettings,
    PromptBatch,
    ScoredRollout,
)
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.stats import RolloutStats
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause
from vrl.utils.deadline import require_timeout

_CPU = torch.device("cpu")
_OBSERVABILITY_LOG_INTERVAL_S = 30.0
_STARVATION_LOG_GAP_S = 10.0
_RETRY_BACKOFF_MAX_S = 0.05

logger = logging.getLogger(__name__)


class ContinuousRolloutProducer:
    """Background producer filling one scored-result slot per prompt."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        staleness: StalenessPolicy,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.lifecycle = lifecycle
        self.staleness = staleness
        # Validated once at the config boundary; trusted here (no re-checks).
        self.max_inflight_groups = settings.max_inflight_groups
        self.poll_interval_s = settings.queue_poll_interval_s
        self.fail_fast_errors = settings.fail_fast_errors

        self.state = ContinuousRolloutProducerState()
        self._next_batch_id = 0
        self._loop_task: asyncio.Task[None] | None = None
        # Each running collect task maps to its slot in the installed batch.
        self._running_tasks: dict[
            asyncio.Task[tuple[list[RolloutBatch], RolloutStats]],
            int,
        ] = {}
        self.prompt_batch: PromptBatch | None = None
        self._last_tick_at: float | None = None
        self._last_observability_log_at = 0.0

    @property
    def _has_pending_work(self) -> bool:
        return bool(
            self._running_tasks
            or (self.prompt_batch is not None and self.prompt_batch.pending_slots)
        )

    @property
    def inflight_count(self) -> int:
        """Display-only live task count derived from its owning container."""

        return len(self._running_tasks)

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Start cadence after the owner has committed initial weights.

        Runtime activation and weight ownership do not belong to this mechanism.
        Production continuous runtimes are resident on disjoint GPUs, while the
        dedicated owner commits the main-thread snapshot before starting this
        loop.
        """

        if self.prompt_batch is None:
            raise RuntimeError(
                "ContinuousRolloutProducer.set_prompt_batch() must be called before start()",
            )
        self.state.running = True
        self._loop_task = asyncio.create_task(self._run())

    def set_prompt_batch(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> None:
        """Install one finite prompt batch and freeze its collection inputs.

        The owner sets the next batch only after consuming the current one.
        Rejecting an incomplete replacement keeps that protocol explicit and
        avoids supporting replacement behavior with no production caller.
        """

        current = self.prompt_batch
        if current is not None and (current.pending_slots or self._running_tasks):
            raise RuntimeError(
                "cannot replace an incomplete continuous prompt batch "
                f"(pending={list(current.pending_slots)}, "
                f"inflight={sorted(self._running_tasks.values())})",
            )
        if current is not None and any(item is not None for item in current.results):
            raise RuntimeError(
                "cannot replace a continuous prompt batch before its ready items "
                f"are consumed (ready={sum(item is not None for item in current.results)})",
            )
        prompt_batch = tuple(prompts)
        installed_at = time.monotonic()
        self.prompt_batch = PromptBatch(
            batch_id=self._next_batch_id,
            policy_version=self.lifecycle.current_policy_version(),
            prompts=prompt_batch,
            group_size=group_size,
            runtime_debug=bool(runtime_debug),
            pending_slots=deque(range(len(prompt_batch))),
            pending_since={slot: installed_at for slot in range(len(prompt_batch))},
            results=[None] * len(prompt_batch),
        )
        self._next_batch_id += 1

    def release_results(self) -> None:
        """Empty the installed batch's result slots once the trainer took them."""

        prompt_batch = self.prompt_batch
        if prompt_batch is None:
            return
        prompt_batch.results[:] = [None] * len(prompt_batch.prompts)
        prompt_batch.consumed = True

    async def stop(self, *, wait_timeout_s: float = 30.0) -> None:
        """Cancel producer tasks and bound cooperative teardown.

        Runtime shutdown follows this call in the owner. A collector coroutine
        that suppresses cancellation must therefore be abandoned after the
        deadline instead of blocking release of its Ray actors forever.
        """

        wait_timeout_s = require_timeout(wait_timeout_s, name="wait_timeout_s")
        self.state.running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        for task in self._running_tasks:
            task.cancel()
        tasks = set(self._running_tasks)
        if self._loop_task is not None:
            tasks.add(self._loop_task)
        pending: set[asyncio.Task[Any]] = set()
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=wait_timeout_s,
            )
            for task in done:
                with contextlib.suppress(BaseException):
                    task.result()
        if pending:
            for task in pending:
                task.cancel()
            logger.error(
                "continuous producer abandoned %d task(s) that did not stop "
                "within %.3fs; collector runtime teardown will proceed",
                len(pending),
                wait_timeout_s,
            )
        self._loop_task = None
        self._running_tasks.clear()

    # -- weight-sync barrier -------------------------------------------

    def pause_admission(self) -> None:
        self.state.paused_for_weight_sync = True

    async def drain_prompt_batch(self, *, wait_timeout_s: float) -> None:
        """Complete pending and in-flight slots of the installed prompt batch.

        Mutating worker weights while a generation request is running could mix
        two policies inside one request. A draining backend therefore finishes
        the finite prompt batch, including slots not yet admitted by the in-flight
        cap, before the weight-sync barrier may proceed.
        """

        wait_timeout_s = require_timeout(wait_timeout_s, name="wait_timeout_s")
        prompt_batch = self.prompt_batch
        if prompt_batch is None:
            return
        deadline = time.monotonic() + wait_timeout_s
        while self._has_pending_work:
            errors_before_harvest = self.state.error_count
            self._harvest_done()
            if self.state.error_count > errors_before_harvest:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    raise TimeoutError(
                        self._drain_timeout_message(prompt_batch, wait_timeout_s),
                    )
                await asyncio.sleep(
                    min(self.poll_interval_s, _RETRY_BACKOFF_MAX_S, remaining_s),
                )
            if not self._has_pending_work:
                return
            blocked_reason = self._admit(allow_paused=True)
            tasks = list(self._running_tasks)
            if not tasks:
                raise RuntimeError(
                    "continuous finite prompt batch cannot admit its pending slots "
                    f"(blocked={blocked_reason!r})",
                )
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    self._drain_timeout_message(prompt_batch, wait_timeout_s),
                )
            await asyncio.wait(
                tasks,
                timeout=min(self.poll_interval_s, remaining_s),
                return_when=asyncio.FIRST_COMPLETED,
            )
        self._harvest_done()

    def _drain_timeout_message(
        self,
        prompt_batch: PromptBatch,
        wait_timeout_s: float,
    ) -> str:
        completed_slots = (
            len(prompt_batch.prompts) - len(prompt_batch.pending_slots) - len(self._running_tasks)
        )
        return (
            "continuous weight-sync barrier timed out draining finite prompt batch "
            f"after {wait_timeout_s}s (pending={list(prompt_batch.pending_slots)}, "
            f"inflight={sorted(self._running_tasks.values())}, "
            f"completed={completed_slots}, "
            f"failures={prompt_batch.failure_counts}, last_error={self.state.last_error})"
        )

    def resume_admission(self) -> None:
        self.state.paused_for_weight_sync = False

    def admit_now(self) -> None:
        """Harvest completions and fill admission slots without a polling delay."""

        if not self.state.running or self.state.paused_for_weight_sync:
            return
        self._harvest_done()
        self._admit()

    # -- main loop ------------------------------------------------------

    async def _run(self) -> None:
        try:
            while self.state.running:
                self._record_tick()
                if self.state.paused_for_weight_sync:
                    await asyncio.sleep(self.poll_interval_s)
                    continue
                self._harvest_done()
                self._admit()
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
            raise
        except BaseException as error:
            # A cadence/control failure is not a retryable collect error: this
            # task is the only admission loop, so record a behavior-consumed root
            # that makes the waiting consumer quarantine the owner immediately.
            self.state.running = False
            self.state.error_count += 1
            self.state.last_error = repr(error)
            self.state.fatal_error = error
            if find_error_cause(error, TerminalRuntimeError) is not None:
                siblings = list(self._running_tasks)
                for task in siblings:
                    task.cancel()
                if siblings:
                    await asyncio.gather(*siblings, return_exceptions=True)
                self._running_tasks.clear()
            logger.error(
                "continuous rollout producer control loop failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _admit(
        self,
        *,
        allow_paused: bool = False,
    ) -> str | None:
        prompt_batch = self.prompt_batch
        if prompt_batch is None:
            return "no_active_prompt_batch"
        if self.state.paused_for_weight_sync and not allow_paused:
            return "paused_for_weight_sync"
        while prompt_batch.pending_slots:
            if len(self._running_tasks) >= self.max_inflight_groups:
                return "inflight_full"
            slot = prompt_batch.pending_slots.popleft()
            if slot in self._running_tasks.values():
                raise RuntimeError(
                    f"continuous prompt batch attempted duplicate in-flight slot {slot}",
                )
            self._submit(prompt_batch, slot)
        return None

    def _submit(self, prompt_batch: PromptBatch, slot: int) -> None:
        pending_since = prompt_batch.pending_since.pop(slot)
        admission_wait_s = max(0.0, time.monotonic() - pending_since)
        task = asyncio.create_task(
            self._collect_group(
                prompt_batch=prompt_batch,
                slot=slot,
                admission_wait_s=admission_wait_s,
            ),
        )
        self._running_tasks[task] = slot
        self.state.submitted_count += 1

    async def _collect_group(
        self,
        *,
        prompt_batch: PromptBatch,
        slot: int,
        admission_wait_s: float,
    ) -> tuple[list[RolloutBatch], RolloutStats]:
        stats = RolloutStats()
        stats.observe_gauge("continuous.generation_queue_wait_s", admission_wait_s)
        batches = await self.lifecycle.collector.prepare_training_batches(
            prompts=[prompt_batch.prompts[slot]],
            group_size=prompt_batch.group_size,
            runtime_debug=prompt_batch.runtime_debug,
            policy_version=prompt_batch.policy_version,
            stats=stats,
        )
        return batches, stats

    def _harvest_done(self) -> None:
        prompt_batch = self.prompt_batch
        if prompt_batch is None or not self._running_tasks:
            return
        done = [task for task in self._running_tasks if task.done()]
        for task in done:
            slot = self._running_tasks.pop(task)
            try:
                batches, stats = task.result()
            except asyncio.CancelledError as exc:
                if self.state.running:
                    self.state.error_count += 1
                    self.state.last_error = repr(exc)
                    raise RuntimeError(
                        f"continuous active prompt-batch collect was cancelled (slot={slot})",
                    ) from exc
                continue
            except StaleSlotDiscard as exc:
                if prompt_batch.runtime_debug:
                    logger.info(
                        "continuous rollout lost its fixed policy-version slot: %s",
                        exc,
                    )
                raise RuntimeError(
                    "continuous prompt batch lost its fixed policy-version "
                    f"slot (slot={slot}, version={prompt_batch.policy_version})",
                ) from exc
            except Exception as exc:
                self.state.last_error = repr(exc)
                if find_error_cause(exc, TerminalRuntimeError) is not None:
                    raise
                self.state.error_count += 1
                # Surface immediately: a persistent generation/reward failure
                # would otherwise stay invisible until a periodic tick, and the
                # consumer would only see an opaque wait timeout downstream.
                logger.warning(
                    "continuous rollout collect failed (error_count=%d, completed=%d): %s",
                    self.state.error_count,
                    self.state.completed_count,
                    str(exc),
                )
                failures = prompt_batch.failure_counts.get(slot, 0) + 1
                prompt_batch.failure_counts[slot] = failures
                if self.fail_fast_errors and failures >= self.fail_fast_errors:
                    raise RuntimeError(
                        "continuous prompt batch slot exceeded the failure "
                        f"budget (slot={slot}, failures={failures})",
                    ) from exc
                prompt_batch.pending_slots.append(slot)
                prompt_batch.pending_since[slot] = time.monotonic()
                continue
            self.state.completed_count += 1
            self._store_result(
                prompt_batch=prompt_batch,
                slot=slot,
                batches=batches,
                stats=stats,
            )

    def _store_result(
        self,
        *,
        prompt_batch: PromptBatch,
        slot: int,
        batches: list[RolloutBatch],
        stats: RolloutStats,
    ) -> None:
        # Receipt-time freshness gate. A group can finish generation only after
        # the trainer has already advanced past the staleness window — its
        # version was stamped at submit time, but current_version moved while it
        # was in flight. A finite batch cannot silently drop one completed slot:
        # no replacement can preserve its fixed policy version, and the consumer
        # would otherwise wait for a batch that can never become complete.
        # Future items pass this gate; the consumer rejects their negative
        # staleness as a version-barrier violation. A zero window is retained only for
        # isolated mechanism tests; production continuous config requires >= 1.
        current_version = self.lifecycle.current_policy_version()
        if self.staleness.too_stale(
            prompt_batch.policy_version,
            current_version,
        ):
            raise RuntimeError(
                "continuous prompt batch became stale before completion "
                f"(item={prompt_batch.policy_version}, trainer={current_version})",
            )
        if len(batches) != 1:
            raise RuntimeError(
                "continuous single-slot collect must return exactly one batch, "
                f"got {len(batches)}",
            )
        if prompt_batch.results[slot] is not None:
            raise RuntimeError(f"continuous prompt batch completed a slot twice (slot={slot})")
        stored = batches[0].to_device(_CPU)
        # Per-item stage views with gauge (max-on-merge) reduction: summing
        # concurrent per-slot walls would overstate wall-clock, so
        # the iteration reports the worst per-item interval. The summed phase
        # twins (collect.generation_wall / collect.reward_wall) remain the
        # busy-total view for duty-ratio analysis.
        stats.observe_gauge(
            "continuous.generation_service_s",
            stats.phase_seconds.get("collect.generation_wall", 0.0),
        )
        stats.observe_gauge(
            "continuous.reward_service_s",
            stats.phase_seconds.get("collect.reward_wall", 0.0),
        )
        if stats.reward_queue_wait_ms is not None:
            stats.observe_gauge(
                "continuous.reward_queue_wait_s",
                float(stats.reward_queue_wait_ms) / 1000.0,
            )
        prompt_batch.results[slot] = ScoredRollout(
            batch_id=prompt_batch.batch_id,
            group_slot=slot,
            rollout_policy_version=prompt_batch.policy_version,
            batch=stored,
            completed_at=time.monotonic(),
            stats=stats,
        )

    def _record_tick(self) -> None:
        now = time.monotonic()
        last_tick_at = self._last_tick_at
        self._last_tick_at = now
        self.state.tick_count += 1
        if last_tick_at is None:
            return

        gap_s = now - last_tick_at
        self.state.last_tick_gap_s = gap_s
        self.state.max_tick_gap_s = max(self.state.max_tick_gap_s, gap_s)
        prompt_batch = self.prompt_batch
        should_log_debug = (
            prompt_batch is not None
            and prompt_batch.runtime_debug
            and now - self._last_observability_log_at >= _OBSERVABILITY_LOG_INTERVAL_S
        )
        should_log_starvation = gap_s >= _STARVATION_LOG_GAP_S
        if not should_log_debug and not should_log_starvation:
            return

        self._last_observability_log_at = now
        log_fn = logger.warning if should_log_starvation else logger.info
        log_fn(
            "continuous rollout producer tick: gap_s=%.3f max_gap_s=%.3f "
            "inflight=%d queue_size=%d submitted=%d completed=%d paused=%s errors=%d",
            gap_s,
            self.state.max_tick_gap_s,
            len(self._running_tasks),
            0 if prompt_batch is None else sum(item is not None for item in prompt_batch.results),
            self.state.submitted_count,
            self.state.completed_count,
            self.state.paused_for_weight_sync,
            self.state.error_count,
        )


__all__ = ["ContinuousRolloutProducer"]
