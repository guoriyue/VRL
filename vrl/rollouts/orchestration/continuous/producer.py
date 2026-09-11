"""Owner-loop continuous rollout producer.

A background ``asyncio`` task keeps a bounded number of ``RolloutCollector``
collect jobs in flight for a bounded prompt-batch window, stamps each completed group
with the batch's policy version, and pushes it onto the ready queue. The heavy
generation work is dispatched by the collector (e.g. to remote Ray generation
actors), so this loop only schedules and harvests — that is enough to overlap
rollout with training on a cross-node setup. The opt-in split path releases
its generation slot on receipt, keeps a bounded artifact reservation, and
serializes reward independently. Unsupported collectors retain composite jobs.

The producer never computes advantages, calls the evaluator/algorithm, or
touches the optimizer; it owns rollout *production cadence* only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import move_training_batch_to_device
from vrl.rollouts.collector.core import RewardCollectionMode
from vrl.rollouts.orchestration.continuous.generated_capacity import GeneratedRolloutCapacity
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    ContinuousRolloutProducerState,
    ContinuousRolloutSettings,
)
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.stats import RolloutStats
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause
from vrl.trajectory import trajectory_tensor_bytes
from vrl.utils.config import require_exact_int
from vrl.utils.deadline import require_timeout

_CPU = torch.device("cpu")
_OBSERVABILITY_LOG_INTERVAL_S = 30.0
_STARVATION_LOG_GAP_S = 10.0
_RETRY_BACKOFF_MAX_S = 0.05

logger = logging.getLogger(__name__)


class _GroupProductionError(RuntimeError):
    """A generated group cannot be retried without replacing its sampled data."""


@dataclass(slots=True)
class _ActivePromptBatch:
    """Batch-owned inputs and mutable progress for one finite prompt batch.

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
    # timing instead of a shared mutable accumulator.
    pending_since: dict[int, float] = field(default_factory=dict)
    failure: BaseException | None = None

    def __post_init__(self) -> None:
        # The value object owns its shape (vLLM's SamplingParams pattern):
        # per-call inputs validate where they are born, not in the mechanism.
        if not self.prompts:
            raise ValueError("continuous prompt batch requires a non-empty prompt list")
        require_exact_int(self.group_size, path="continuous prompt batch.group_size", minimum=1)


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

        self._split_reward = settings.split_generation_reward
        if self._split_reward and not lifecycle.collector.supports_reward_generation_overlap:
            raise ValueError(
                "split generation/reward requires nonblocking reward scoring and "
                "verified accelerator isolation",
            )
        self._generated_capacity = GeneratedRolloutCapacity(
            max_groups=settings.max_unscored_groups,
            max_bytes=settings.max_unscored_bytes_mb * 1024 * 1024,
        )
        self._group_byte_ceiling = settings.max_generated_group_bytes_mb * 1024 * 1024
        self._generating: set[tuple[int, int]] = set()
        self._reward_lock = asyncio.Lock()
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
            tuple[int, int],
        ] = {}
        self._batches: dict[int, _ActivePromptBatch] = {}
        self._last_tick_at: float | None = None
        self._last_observability_log_at = 0.0

    @property
    def _active_batch(self) -> _ActivePromptBatch | None:
        return next(iter(self._batches.values()), None)

    @property
    def _has_pending_work(self) -> bool:
        return bool(self._inflight or any(batch.pending_slots for batch in self._batches.values()))

    @property
    def inflight_count(self) -> int:
        """Display-only live task count derived from its owning container."""

        return len(self._inflight)

    @property
    def current_batch_id(self) -> int:
        """Identity the consumer must demand, independent of completion order."""

        if self._active_batch is None:
            raise RuntimeError("continuous producer has no installed prompt batch")
        if self._active_batch.failure is not None:
            raise self._active_batch.failure
        return self._active_batch.batch_id

    def stage_stats(self) -> dict[str, float]:
        """Owner-loop snapshot of generation and reward capacity."""

        if not self._split_reward:
            return {}
        return {
            "active_batches": float(len(self._batches)),
            "generation_inflight": float(len(self._generating)),
            **self._generated_capacity.stats(),
        }

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
        batch = self._new_prompt_batch(prompts, group_size=group_size, runtime_debug=runtime_debug)
        self._batches = {batch.batch_id: batch}
        self._next_batch_id += 1

    def append_prompt_batch(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> None:
        """Append the recipe's one preview without replacing unfinished work."""

        if not self._split_reward:
            raise RuntimeError("early prefetch requires split generation/reward")
        if len(self._batches) != 1:
            raise RuntimeError("continuous prefetch requires exactly one current batch")
        batch = self._new_prompt_batch(prompts, group_size=group_size, runtime_debug=runtime_debug)
        self._batches[batch.batch_id] = batch
        self._next_batch_id += 1

    def consume_prompt_batch(self, batch_id: int) -> None:
        """Advance the head only after its complete iteration was removed."""

        if batch_id != self.current_batch_id:
            raise RuntimeError("continuous consumption attempted to skip the head batch")
        batch = self._batches[batch_id]
        if batch.pending_slots or any(key[0] == batch_id for key in self._inflight.values()):
            raise RuntimeError("continuous consumption attempted to retire unfinished work")
        if any(item.batch_id == batch_id for item in self.queue.snapshot()):
            raise RuntimeError("continuous consumption left ready items in the head batch")
        del self._batches[batch_id]

    def _new_prompt_batch(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> _ActivePromptBatch:
        """Construct and validate a batch before changing installed state."""

        prompt_batch = tuple(prompts)
        installed_at = time.monotonic()
        return _ActivePromptBatch(
            batch_id=self._next_batch_id,
            policy_version=self.lifecycle.current_policy_version(),
            prompts=prompt_batch,
            group_size=group_size,
            runtime_debug=bool(runtime_debug),
            pending_slots=deque(range(len(prompt_batch))),
            pending_since={slot: installed_at for slot in range(len(prompt_batch))},
        )

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
        for task in self._inflight:
            task.cancel()
        tasks = set(self._inflight)
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
        self._inflight.clear()
        self._generating.clear()
        self._generated_capacity.close()

    # -- weight-sync barrier -------------------------------------------

    def pause_admission(self) -> None:
        self.state.paused_for_weight_sync = True

    async def drain_prompt_batch(self, *, wait_timeout_s: float) -> None:
        """Complete pending and in-flight slots across the installed batch window.

        Mutating worker weights while a generation request is running could mix
        two policies inside one request. A draining backend therefore finishes
        the finite prompt batch, including slots not yet admitted by the in-flight
        cap, before the weight-sync barrier may proceed.
        """

        wait_timeout_s = require_timeout(wait_timeout_s, name="wait_timeout_s")
        prompt_batch = self._active_batch
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
        completed_slots = sum(
            len(batch.prompts) - len(batch.pending_slots) for batch in self._batches.values()
        ) - len(self._inflight)
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
        return prompt_batch is not None and not self._has_pending_work

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
        for prompt_batch in self._batches.values():
            if prompt_batch.failure is not None:
                if prompt_batch is self._active_batch:
                    raise prompt_batch.failure
                continue
            while prompt_batch.pending_slots:
                active = len(self._generating) if self._split_reward else len(self._inflight)
                if active >= self.max_inflight_groups:
                    return "inflight_full"
                slot = prompt_batch.pending_slots[0]
                if self._split_reward and not self._generated_capacity.reserve(
                    (prompt_batch.batch_id, slot),
                    max_group_bytes=self._group_byte_ceiling,
                ):
                    return "unscored_full"
                prompt_batch.pending_slots.popleft()
                if (prompt_batch.batch_id, slot) in self._inflight.values():
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
        self._inflight[task] = (prompt_batch.batch_id, slot)
        if self._split_reward:
            self._generating.add((prompt_batch.batch_id, slot))
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
        if self._split_reward:
            return await self._collect_split_group(prompt_batch, slot, stats)
        batches = await self.lifecycle.collector.collect_prompt_groups(
            prompts=[prompt_batch.prompts[slot]],
            group_size=prompt_batch.group_size,
            runtime_debug=prompt_batch.runtime_debug,
            policy_version=prompt_batch.policy_version,
            stats=stats,
            reward_mode=RewardCollectionMode.BATCHED_SERIAL,
        )
        return batches, stats

    async def _collect_split_group(
        self,
        prompt_batch: _ActivePromptBatch,
        slot: int,
        stats: RolloutStats,
    ) -> tuple[list[RolloutBatch], RolloutStats]:
        key = (prompt_batch.batch_id, slot)
        started = time.perf_counter()
        generated = False
        try:
            async for receipt in self.lifecycle.collector.generate_prompt_groups(
                prompts=[prompt_batch.prompts[slot]],
                group_size=prompt_batch.group_size,
                runtime_debug=prompt_batch.runtime_debug,
                policy_version=prompt_batch.policy_version,
            ):
                generated = True
                if self.staleness.too_stale(
                    prompt_batch.policy_version,
                    self.lifecycle.current_policy_version(),
                ):
                    raise RuntimeError("continuous generated group became stale before reward")
                self._generated_capacity.record_generated(
                    key, nbytes=trajectory_tensor_bytes(receipt.unscored)
                )
                # Only GPU generation occupies a generation slot. Capacity for
                # the artifact remains reserved until scoring settles below.
                self._generating.discard(key)
                queued_at = time.perf_counter()
                async with self._reward_lock:
                    self._generated_capacity.start_scoring(key)
                    stats.observe_gauge(
                        "continuous.reward_queue_wait_s", time.perf_counter() - queued_at
                    )
                    reward_started = time.perf_counter()
                    failures = 0
                    while True:
                        try:
                            batches = await self.lifecycle.collector.score_rollouts(
                                [receipt.unscored]
                            )
                            break
                        except Exception as error:
                            if find_error_cause(error, TerminalRuntimeError) is not None:
                                raise
                            failures += 1
                            stats.add_counter("continuous.reward_retries", 1)
                            # Reward cannot regenerate the group on exhaustion.
                            # Even when collect fail-fast is disabled, keep this
                            # stage bounded by one attempt rather than retry forever.
                            if failures >= max(1, self.fail_fast_errors):
                                raise RuntimeError(
                                    "continuous reward exhausted its retry budget"
                                ) from error
                            await asyncio.sleep(min(self.poll_interval_s, _RETRY_BACKOFF_MAX_S))
                    reward_wall = time.perf_counter() - reward_started
                batches = self.lifecycle.collector.finish_scored_prompt_groups(
                    [(receipt.unscored, receipt.prompt_indices)],
                    batches,
                    stats,
                )
                stats.add_phases(
                    {
                        "collect.wall": time.perf_counter() - started,
                        "collect.generation_wall": receipt.completed_at - receipt.started_at,
                        "collect.reward_wall": reward_wall,
                    }
                )
                stats.add_counter("collect.group_count", len(batches))
                stats.add_counter(
                    "collect.sample_count", sum(int(batch.rewards.shape[0]) for batch in batches)
                )
                return batches, stats
            raise RuntimeError("continuous generation returned no prompt group")
        except Exception as error:
            if generated and find_error_cause(error, TerminalRuntimeError) is None:
                raise _GroupProductionError(
                    f"continuous generated group failed before scored publication (group={key})",
                ) from error
            raise
        finally:
            self._generating.discard(key)
            self._generated_capacity.release(key)

    def _fail_batch(self, batch: _ActivePromptBatch, error: BaseException) -> None:
        """Defer a preview-local failure until it becomes the demanded head.

        Runtime terminal errors still quarantine the entire fleet immediately;
        only a failure known to belong to one batch may preserve current work.
        """

        if (
            batch is self._active_batch
            or find_error_cause(error, TerminalRuntimeError) is not None
        ):
            raise error
        batch.failure = error
        # Keep the traceback locations, but release media retained by completed
        # coroutine frames. Run after harvesting returns so its own frame can
        # be cleared as well. Explicit causes preserve the original failure.
        seen: set[int] = set()
        cause: BaseException | None = error
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if cause.__traceback__ is not None:
                asyncio.get_running_loop().call_soon(traceback.clear_frames, cause.__traceback__)
            cause = cause.__cause__ or cause.__context__
        batch.pending_slots.clear()
        for task, key in self._inflight.items():
            if key[0] == batch.batch_id:
                task.cancel()

    def _harvest_done(self) -> None:
        if not self._inflight:
            return
        done = [task for task in self._inflight if task.done()]
        for task in done:
            batch_id, slot = self._inflight.pop(task)
            prompt_batch = self._batches.get(batch_id)
            if prompt_batch is None:
                raise RuntimeError("continuous collect completed without an active prompt batch")
            if prompt_batch.failure is not None:
                with contextlib.suppress(BaseException):
                    task.result()
                key = (batch_id, slot)
                if key in self._generating:
                    # A task cancelled before its first execution never entered
                    # _collect_split_group's finally block.
                    self._generating.discard(key)
                    self._generated_capacity.release(key)
                continue
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
            except _GroupProductionError as exc:
                self._fail_batch(prompt_batch, exc)
                continue
            except StaleSlotDiscard as exc:
                if prompt_batch.runtime_debug:
                    logger.info(
                        "continuous rollout lost its fixed policy-version slot: %s",
                        exc,
                    )
                error = _GroupProductionError(
                    "continuous prompt batch lost its fixed policy-version "
                    f"slot (slot={slot}, version={prompt_batch.policy_version})",
                )
                error.__cause__ = exc
                self._fail_batch(prompt_batch, error)
                continue
            except Exception as exc:
                self.state.last_error = repr(exc)
                if find_error_cause(exc, TerminalRuntimeError) is not None:
                    raise
                if prompt_batch is self._active_batch:
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
                    error = _GroupProductionError(
                        "continuous prompt batch slot exceeded the failure "
                        f"budget (slot={slot}, failures={failures})",
                    )
                    error.__cause__ = exc
                    self._fail_batch(prompt_batch, error)
                    continue
                prompt_batch.pending_slots.append(slot)
                prompt_batch.pending_since[slot] = time.monotonic()
                continue
            self.state.completed_count += 1
            try:
                self._enqueue_result(
                    prompt_batch=prompt_batch,
                    slot=slot,
                    batches=batches,
                    stats=stats,
                )
            except Exception as error:
                self._fail_batch(prompt_batch, error)

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
        # too_stale() returns False for absent versions (no gating). Future
        # items also pass this gate; the consumer rejects their negative
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
        stored = move_training_batch_to_device(batches[0], _CPU)
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
        item = ContinuousRolloutItem(
            batch_id=prompt_batch.batch_id,
            group_slot=slot,
            rollout_policy_version=prompt_batch.policy_version,
            attempt=prompt_batch.failure_counts.get(slot, 0) + 1,
            batch=stored,
            completed_at=time.monotonic(),
            nbytes=stored.estimated_payload_bytes(),
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
