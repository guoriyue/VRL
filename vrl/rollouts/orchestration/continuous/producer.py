"""Owner-loop continuous rollout producer.

A background ``asyncio`` task keeps a bounded number of ``RolloutCollector``
collect jobs in flight for one finite prompt batch, stamps each completed group
with the batch's policy version, and pushes it onto the ready queue. The heavy
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
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import move_training_batch_to_device
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    ContinuousRolloutProducerState,
    ContinuousRolloutSettings,
    estimate_batch_bytes,
)
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_groups
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import RewardCollectionMode
from vrl.rollouts.stats import RolloutStats
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause

_CPU = torch.device("cpu")
_OBSERVABILITY_LOG_INTERVAL_S = 30.0
_STARVATION_LOG_GAP_S = 10.0
_RETRY_BACKOFF_MAX_S = 0.05

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ActivePromptBatch:
    """Frozen inputs and mutable progress for one finite prompt batch.

    Keeping both on one object prevents a retry from reading policy or collection
    settings from the next batch while its slot progress still belongs to this one.
    """

    batch_id: int
    policy_version: int | None
    prompts: tuple[Any, ...]
    group_size: int
    runtime_debug: bool
    pending_slots: deque[int]
    failure_counts: dict[int, int] = field(default_factory=dict)
    # Monotonic stamp of when each pending slot last became admissible (batch
    # install or retry re-queue). _submit() turns it into the slot's admission
    # wait and hands it to the collect task, so each item carries its own
    # timing instead of a shared mutable accumulator (sprint invariant §3.2).
    pending_since: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The value object owns its shape (vLLM's SamplingParams pattern):
        # per-call inputs validate where they are born, not in the mechanism.
        if not self.prompts:
            raise ValueError("continuous prompt batch requires a non-empty prompt list")
        if int(self.group_size) < 1:
            raise ValueError("continuous prompt batch group_size must be >= 1")


class ContinuousRolloutProducer:
    """Background producer feeding the continuous ready queue."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        queue: ContinuousRolloutQueue,
        staleness: StalenessPolicy,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.lifecycle = lifecycle
        self.queue = queue
        self.staleness = staleness
        # Validated once at the config boundary; trusted here (no re-checks).
        self.max_inflight_groups = settings.max_inflight_groups
        self.poll_interval_s = settings.queue_poll_interval_s
        self.fail_fast_errors = settings.fail_fast_errors

        self.state = ContinuousRolloutProducerState()
        self._next_batch_id = 0
        # Backpressure accrual: the reason observed at the previous tick and
        # the monotonic instant of that tick; each tick charges the elapsed
        # interval to the previous tick's reason.
        self._blocked_reason: str | None = None
        self._blocked_since: float | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._inflight: dict[
            asyncio.Task[tuple[list[RolloutBatch], RolloutStats]],
            int,
        ] = {}
        self._active_batch: _ActivePromptBatch | None = None
        self._last_tick_at: float | None = None
        self._last_observability_log_at = 0.0

    @property
    def inflight_count(self) -> int:
        """Display-only live task count derived from its owning container."""

        return len(self._inflight)

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Start cadence after the owner has committed initial weights.

        Runtime activation and weight ownership do not belong to this mechanism.
        Production continuous runtimes are resident on disjoint GPUs, while the
        dedicated owner commits the main-thread snapshot before starting this
        loop.
        """

        if self._active_batch is None:
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

        current = self._active_batch
        if current is not None and (current.pending_slots or self._inflight):
            raise RuntimeError(
                "cannot replace an incomplete continuous prompt batch "
                f"(pending={list(current.pending_slots)}, "
                f"inflight={sorted(self._inflight.values())})",
            )
        if current is not None and self.queue.size():
            raise RuntimeError(
                "cannot replace a continuous prompt batch before its ready items "
                f"are consumed (ready={self.queue.size()})",
            )
        prompt_batch = tuple(prompts)
        installed_at = time.monotonic()
        self._active_batch = _ActivePromptBatch(
            batch_id=self._next_batch_id,
            policy_version=self.lifecycle.current_policy_version(),
            prompts=prompt_batch,
            group_size=int(group_size),
            runtime_debug=bool(runtime_debug),
            pending_slots=deque(range(len(prompt_batch))),
            pending_since={slot: installed_at for slot in range(len(prompt_batch))},
        )
        self._next_batch_id += 1

    async def stop(self, *, wait_timeout_s: float = 30.0) -> None:
        """Cancel producer tasks and bound cooperative teardown.

        Runtime shutdown follows this call in the owner. A collector coroutine
        that suppresses cancellation must therefore be abandoned after the
        deadline instead of blocking release of its Ray actors forever.
        """

        self.state.running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        for task in self._inflight:
            task.cancel()
        tasks = set(self._inflight)
        if self._loop_task is not None:
            tasks.add(self._loop_task)
        pending: set[asyncio.Task[Any]] = set()
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=max(0.0, float(wait_timeout_s)),
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
        self._inflight.clear()

    # -- weight-sync barrier -------------------------------------------

    def pause_admission(self) -> None:
        self.state.paused_for_weight_sync = True

    async def drain_prompt_batch(self, *, wait_timeout_s: float) -> None:
        """Complete every pending and in-flight slot in the active prompt batch.

        Mutating worker weights while a generation request is running could mix
        two policies inside one request. A draining backend therefore finishes
        the finite prompt batch, including slots not yet admitted by the in-flight
        cap, before the weight-sync barrier may proceed.
        """

        prompt_batch = self._active_batch
        if prompt_batch is None:
            return
        deadline = time.monotonic() + float(wait_timeout_s)
        while prompt_batch.pending_slots or self._inflight:
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
            if not (prompt_batch.pending_slots or self._inflight):
                return
            blocked_reason = self._admit(allow_paused=True)
            tasks = list(self._inflight)
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
        prompt_batch: _ActivePromptBatch,
        wait_timeout_s: float,
    ) -> str:
        completed_slots = (
            len(prompt_batch.prompts) - len(prompt_batch.pending_slots) - len(self._inflight)
        )
        return (
            "continuous weight-sync barrier timed out draining finite prompt batch "
            f"after {wait_timeout_s}s (pending={list(prompt_batch.pending_slots)}, "
            f"inflight={sorted(self._inflight.values())}, "
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

    def _starved(self) -> bool:
        """No admissible or in-flight work left in the active batch.

        The batch is fully generated and the producer idles until the trainer
        consumes it and installs the next one — the rollout-GPU starvation
        state the four-L4 baseline must be able to name (sprint §7).
        """

        prompt_batch = self._active_batch
        return prompt_batch is not None and not prompt_batch.pending_slots and not self._inflight

    def _accrue_backpressure(self, reason: str | None) -> None:
        """Charge the interval since the previous tick to that tick's reason.

        Accruing every tick (not on reason exit) keeps a long steady block
        visible in the owner's metric exports while it is still happening.
        """

        now = time.monotonic()
        previous = self._blocked_reason
        since = self._blocked_since
        if previous is not None and since is not None:
            seconds = self.state.backpressure_seconds
            seconds[previous] = seconds.get(previous, 0.0) + max(0.0, now - since)
        if reason is not None and reason != previous:
            entries = self.state.backpressure_entries
            entries[reason] = entries.get(reason, 0.0) + 1.0
        self._blocked_reason = reason
        self._blocked_since = now

    async def _run(self) -> None:
        try:
            while self.state.running:
                self._record_tick()
                if self.state.paused_for_weight_sync:
                    self._accrue_backpressure("paused_for_weight_sync")
                    await asyncio.sleep(self.poll_interval_s)
                    continue
                self._harvest_done()
                blocked = self._admit()
                if blocked is None and self._starved():
                    blocked = "no_pending_slots"
                self._accrue_backpressure(blocked)
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
                siblings = list(self._inflight)
                for task in siblings:
                    task.cancel()
                if siblings:
                    await asyncio.gather(*siblings, return_exceptions=True)
                self._inflight.clear()
            logger.error(
                "continuous rollout producer control loop failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _admit(
        self,
        *,
        allow_paused: bool = False,
    ) -> str | None:
        prompt_batch = self._active_batch
        if prompt_batch is None:
            return "no_active_prompt_batch"
        if self.state.paused_for_weight_sync and not allow_paused:
            return "paused_for_weight_sync"
        while prompt_batch.pending_slots:
            if len(self._inflight) >= self.max_inflight_groups:
                return "inflight_full"
            slot = prompt_batch.pending_slots.popleft()
            if slot in self._inflight.values():
                raise RuntimeError(
                    f"continuous prompt batch attempted duplicate in-flight slot {slot}",
                )
            self._submit(prompt_batch, slot)
        return None

    def _submit(self, prompt_batch: _ActivePromptBatch, slot: int) -> None:
        pending_since = prompt_batch.pending_since.pop(slot, None)
        admission_wait_s = (
            0.0 if pending_since is None else max(0.0, time.monotonic() - pending_since)
        )
        task = asyncio.create_task(
            self._collect_group(
                prompt_batch=prompt_batch,
                slot=slot,
                admission_wait_s=admission_wait_s,
            ),
        )
        self._inflight[task] = slot
        self.state.submitted_count += 1

    async def _collect_group(
        self,
        *,
        prompt_batch: _ActivePromptBatch,
        slot: int,
        admission_wait_s: float,
    ) -> tuple[list[RolloutBatch], RolloutStats]:
        stats = RolloutStats()
        stats.observe_gauge("continuous.generation_queue_wait_s", admission_wait_s)
        batches = await collect_prompt_groups(
            collector=self.lifecycle.collector,
            prompts=[prompt_batch.prompts[slot]],
            group_size=prompt_batch.group_size,
            runtime_debug=prompt_batch.runtime_debug,
            policy_version=prompt_batch.policy_version,
            stats=stats,
            reward_mode=RewardCollectionMode.BATCHED_SERIAL,
        )
        return batches, stats

    def _harvest_done(self) -> None:
        if not self._inflight:
            return
        done = [task for task in self._inflight if task.done()]
        for task in done:
            slot = self._inflight.pop(task)
            prompt_batch = self._active_batch
            if prompt_batch is None:
                raise RuntimeError("continuous collect completed without an active prompt batch")
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
                    exc,
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
            self._enqueue_result(
                prompt_batch=prompt_batch,
                slot=slot,
                batches=batches,
                stats=stats,
            )

    def _enqueue_result(
        self,
        *,
        prompt_batch: _ActivePromptBatch,
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
        # too_stale() returns False for absent versions (no gating) and for
        # future items (staleness < 0, a bug), so those still flow to the
        # consumer, which fails fast on them. A zero window is retained only for
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
        stored = move_training_batch_to_device(batches[0], _CPU)
        # Per-item stage views with gauge (max-on-merge) reduction: summing
        # concurrent per-slot walls would overstate wall-clock (sprint §6), so
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
        item = ContinuousRolloutItem(
            batch_id=prompt_batch.batch_id,
            group_slot=slot,
            rollout_policy_version=prompt_batch.policy_version,
            attempt=prompt_batch.failure_counts.get(slot, 0) + 1,
            batch=stored,
            completed_at=time.monotonic(),
            nbytes=estimate_batch_bytes(stored),
            stats=stats,
        )
        self.queue.put(item)

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
        prompt_batch = self._active_batch
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
            len(self._inflight),
            self.queue.size(),
            self.state.submitted_count,
            self.state.completed_count,
            self.state.paused_for_weight_sync,
            self.state.error_count,
        )


__all__ = ["ContinuousRolloutProducer"]
