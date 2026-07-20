"""Dedicated event-loop owner for disaggregated continuous rollout.

The trainer's asyncio loop must remain free to run synchronous forward/backward
work without starving rollout admission and completion harvesting.  This module
owns the continuous producer, queue, consumer, and every asynchronous runtime
operation on one dedicated thread/event loop.  The trainer side communicates
only through ``concurrent.futures.Future`` command boundaries.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.scheduler import RolloutScheduler
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.rollouts.orchestration.rollout_runtime import (
    RolloutRuntimeCoordinator,
    record_phase,
)
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
)
from vrl.utils.stats import RolloutStats

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_OWNER_START_TIMEOUT_S = 10.0
_OWNER_STOP_TIMEOUT_S = 30.0

_T = TypeVar("_T")


def _prompt_batches_equal(left: list[Any], right: list[Any]) -> bool:
    """Compare complete prompt inputs without assuming scalar equality."""

    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right, strict=True):
        if left_item is right_item:
            continue
        try:
            if not bool(left_item == right_item):
                return False
        except (TypeError, ValueError, RuntimeError):
            return False
    return True


class _ContinuousOwnerRuntime:
    """Continuous pipeline state that is touched only by the owner loop."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.lifecycle = lifecycle
        self.max_inflight_groups = int(settings.max_inflight_groups)
        self.max_ready_groups = int(settings.max_ready_groups)
        self.max_ready_bytes = int(settings.max_ready_bytes_mb) * _MB
        self.wait_timeout_s = float(settings.wait_timeout_s)
        self.queue_poll_interval_s = float(settings.queue_poll_interval_s)
        self.fail_fast_errors = int(settings.fail_fast_errors)
        self.staleness = StalenessPolicy(
            max_stale_policy_versions=int(settings.max_stale_policy_versions),
        )

        self.state = RolloutScheduleState()
        self.queue: ContinuousRolloutQueue | None = None
        self.consumer: ContinuousRolloutConsumer | None = None
        self.producer: ContinuousRolloutProducer | None = None
        self.scheduler: RolloutScheduler | None = None
        self._active_prompts: list[Any] | None = None
        self._active_group_size: int | None = None
        self._active_batch_consumed = True

        self._command_lock = asyncio.Lock()
        self._active_commands: set[asyncio.Task[Any]] = set()
        self._terminal_error: BaseException | None = None
        self._terminal_cleanup_task: asyncio.Task[None] | None = None
        self._shutting_down = False
        self._runtime_closed = False

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
        initial_weights: Any,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        async def operation() -> RolloutIteration:
            if not prompts:
                raise ValueError("continuous rollout requires at least one prompt")
            phase_times: dict[str, float] = {}

            if self.producer is None:
                await self._start_pipeline(
                    prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    initial_weights=initial_weights,
                    phase_times=phase_times,
                )
            elif self._active_batch_consumed:
                self._set_prompt_batch(
                    prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                )
            elif self._active_prompts is None or not _prompt_batches_equal(
                self._active_prompts,
                prompts,
            ):
                raise RuntimeError(
                    "continuous lookahead prompt batch does not match the next prompts "
                    "presented by the trainer",
                )
            elif self._active_group_size != int(group_size):
                raise RuntimeError(
                    "continuous lookahead group size does not match the batch "
                    f"presented by the trainer: expected={self._active_group_size}, "
                    f"requested={group_size}",
                )
            assert self.consumer is not None
            assert self.producer is not None
            rollout_id = self.state.rollout_id
            self.state.rollout_id += 1
            current_version = self.lifecycle.current_policy_version()

            iteration = await self.consumer.drain_for_iteration(
                rollout_id=rollout_id,
                min_groups=len(prompts),
                current_version=current_version,
                mode=RolloutScheduleMode.CONTINUOUS,
                wait_timeout_s=self.wait_timeout_s,
                poll_interval_s=self.queue_poll_interval_s,
                producer_state=self.producer.state,
            )
            self._active_batch_consumed = True
            lookahead_requested = 0.0
            if next_prompts is not None:
                if not next_prompts:
                    raise ValueError("continuous lookahead prompts must be non-empty")
                # Debug metadata belongs to generation time. This lookahead runs
                # during the current training step, even when the trainer consumes
                # it after state.step (and therefore runtime_debug) changes.
                self._set_prompt_batch(
                    next_prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                )
                lookahead_requested = 1.0
            iteration.stats.add_phases(phase_times)
            iteration.stats.observe_gauge(
                "continuous.lookahead_requested",
                lookahead_requested,
            )
            self._attach_producer_metrics(iteration)
            return iteration

        return await self._run_command(operation)

    def _set_prompt_batch(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> None:
        """Install the next finite prompt batch and dispatch it immediately."""

        assert self.producer is not None
        assert self.queue is not None
        if len(prompts) > self.queue.max_items:
            raise ValueError(
                "continuous prompt batch exceeds ready-queue item capacity: "
                f"prompts={len(prompts)}, capacity={self.queue.max_items}",
            )
        self.producer.set_prompt_batch(
            list(prompts),
            group_size=group_size,
            runtime_debug=runtime_debug,
        )
        self._active_prompts = list(prompts)
        self._active_group_size = int(group_size)
        self._active_batch_consumed = False
        self.producer.admit_now()

    async def commit_weights(self, prepared_weights: Any) -> RolloutStats:
        """Commit one main-thread snapshot and reopen admission only on success."""

        async def operation() -> RolloutStats:
            phase_times: dict[str, float] = {}
            stats = RolloutStats()
            producer = self.producer
            if producer is None:
                await self.lifecycle.push_prepared_weights(
                    prepared_weights,
                    phase_times,
                )
                stats.add_phases(phase_times)
                return stats

            non_draining = self.lifecycle.supports_non_draining_weight_sync()
            with record_phase(phase_times, "continuous.weight_sync_pause_s"):
                producer.pause_admission()
                # There is deliberately no finally-resume here.  A partial worker
                # update leaves the fleet's installed version unknown, so failure
                # keeps admission closed and _run_command quarantines the owner.
                if not non_draining:
                    await producer.drain_prompt_batch(wait_timeout_s=self.wait_timeout_s)
                await self.lifecycle.push_prepared_weights(
                    prepared_weights,
                    phase_times,
                )
                assert self.queue is not None
                assert self.scheduler is not None
                self.scheduler.validate_ready_versions(
                    self.queue,
                    current_version=self.lifecycle.current_policy_version(),
                )
                producer.resume_admission()
            stats.add_phases(phase_times)
            stats.observe_gauge(
                "continuous.weight_sync_barrier_mode",
                float(non_draining),
            )
            return stats

        return await self._run_command(operation)

    async def reset(self) -> None:
        """Clear continuous queue/producer state without touching the runtime."""

        async def operation() -> None:
            await self._stop_pipeline()
            self.state = RolloutScheduleState()

        await self._run_command(operation)

    async def shutdown(self) -> None:
        """Cancel owner commands, stop production, and close its runtime once."""

        if self._runtime_closed:
            return
        if not self._shutting_down:
            self._shutting_down = True

            current = asyncio.current_task()
            active = [task for task in self._active_commands if task is not current]
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

        async with self._command_lock:
            await self._run_terminal_cleanup()

    async def _run_command(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio always creates a task here
            raise RuntimeError("continuous owner command has no asyncio task")
        self._active_commands.add(task)
        try:
            async with self._command_lock:
                if self._shutting_down:
                    raise RuntimeError("continuous rollout owner is shutting down")
                if self._terminal_error is not None:
                    raise RuntimeError("continuous rollout owner has failed") from (
                        self._terminal_error
                    )
                return await operation()
        except asyncio.CancelledError as error:
            if not self._shutting_down:
                await self._fail(error)
            raise
        except BaseException as error:
            await self._fail(error)
            raise
        finally:
            self._active_commands.discard(task)

    async def _fail(self, error: BaseException) -> None:
        if self._terminal_error is None:
            self._terminal_error = error
        # Stop admission before runtime cleanup.  RayGenerationRuntime also
        # quarantines itself on a partial worker update; closing the collector
        # here covers generic runtimes and makes subsequent commands fail closed.
        try:
            await self._run_terminal_cleanup()
        except BaseException as cleanup_error:  # preserve the first root cause
            logger.error(
                "continuous owner cleanup failed after root error %r",
                error,
                exc_info=(
                    type(cleanup_error),
                    cleanup_error,
                    cleanup_error.__traceback__,
                ),
            )

    async def _run_terminal_cleanup(self) -> None:
        """Join one owner-local cleanup attempt, retrying only after failure.

        A failed command releases ``_command_lock`` before entering ``_fail``.
        Without this single-flight boundary, a queued command can observe the
        terminal error and call collector shutdown concurrently with the first
        command. Shielding also lets explicit shutdown join the same cleanup if
        it cancels the original command waiter.
        """

        task = self._terminal_cleanup_task
        if task is not None and task.done():
            self._terminal_cleanup_finished(task)
            task = None
        if task is None:
            task = asyncio.create_task(self._terminal_cleanup_once())
            self._terminal_cleanup_task = task
            task.add_done_callback(self._terminal_cleanup_finished)
        await asyncio.shield(task)

    def _terminal_cleanup_finished(self, task: asyncio.Task[None]) -> None:
        if self._terminal_cleanup_task is task:
            self._terminal_cleanup_task = None
        if not task.cancelled():
            task.exception()  # retrieve failures even if every waiter was cancelled

    async def _terminal_cleanup_once(self) -> None:
        await self._stop_pipeline()
        if not self._runtime_closed:
            await self.lifecycle.shutdown_collector_runtime()
            self._runtime_closed = True

    async def _start_pipeline(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
        initial_weights: Any,
        phase_times: dict[str, float],
    ) -> None:
        # The trainer exported this immutable CPU snapshot before crossing the
        # owner boundary. Worker ACK validation happens inside the weight-sync
        # stack before push_prepared_weights publishes the committed version.
        # None means the persistent runtime already owns initialized weights.
        if initial_weights is not None:
            await self.lifecycle.push_prepared_weights(initial_weights, phase_times)

        capacity = max(self.max_ready_groups, len(prompts))
        self.queue = ContinuousRolloutQueue(
            max_items=capacity,
            max_bytes=self.max_ready_bytes,
        )
        self.scheduler = RolloutScheduler(
            staleness=self.staleness,
            max_inflight_groups=self.max_inflight_groups,
            max_bytes=self.max_ready_bytes,
        )
        self.consumer = ContinuousRolloutConsumer(
            queue=self.queue,
            scheduler=self.scheduler,
            fail_fast_errors=self.fail_fast_errors,
        )
        self.producer = ContinuousRolloutProducer(
            lifecycle=self.lifecycle,
            queue=self.queue,
            scheduler=self.scheduler,
            poll_interval_s=self.queue_poll_interval_s,
            fail_fast_errors=self.fail_fast_errors,
        )
        self.producer.set_prompt_batch(
            list(prompts),
            group_size=group_size,
            runtime_debug=runtime_debug,
        )
        self._active_prompts = list(prompts)
        self._active_group_size = int(group_size)
        self._active_batch_consumed = False
        await self.producer.start()
        self.producer.admit_now()

    async def _stop_pipeline(self) -> None:
        producer = self.producer
        queue = self.queue
        self.producer = None
        self.queue = None
        self.consumer = None
        self.scheduler = None
        self._active_prompts = None
        self._active_group_size = None
        self._active_batch_consumed = True
        if producer is not None:
            await producer.stop(wait_timeout_s=_OWNER_STOP_TIMEOUT_S)
        if queue is not None:
            queue.close()

    def _attach_producer_metrics(self, iteration: RolloutIteration) -> None:
        if self.producer is None or self.queue is None:
            return
        state = self.producer.state
        queue_stats = self.queue.stats()
        version = iteration.policy_version
        metadata = iteration.metadata
        iteration.stats.observe_gauges(
            {
                "continuous.producer_inflight": float(state.inflight_count),
                "continuous.producer_tick_count": float(state.tick_count),
                "continuous.producer_last_tick_gap_s": float(state.last_tick_gap_s),
                "continuous.producer_max_tick_gap_s": float(state.max_tick_gap_s),
                "continuous.producer_submitted": float(state.submitted_count),
                "continuous.producer_completed": float(state.completed_count),
                "continuous.queue_ready_items": queue_stats["ready_items"],
                "continuous.queue_ready_groups": queue_stats["ready_groups"],
                "continuous.ready_groups_at_demand": float(
                    metadata.get("continuous_ready_groups_at_demand", 0),
                ),
                "continuous.queue_ready_bytes": queue_stats["ready_bytes"],
                "continuous.item_age_s": float(
                    metadata.get("continuous_item_age_s", 0.0),
                ),
                "continuous.producer_errors": float(state.error_count),
                "continuous.consume_policy_version": float(
                    metadata.get("consume_policy_version") or 0,
                ),
                "continuous.rollout_policy_version": float(
                    0 if version is None else version,
                ),
                "continuous.stale_policy_versions": float(
                    metadata.get("stale_policy_versions") or 0,
                ),
            },
        )


class ContinuousRolloutOwner:
    """Thread/event-loop facade used by ``ContinuousRolloutSchedule``."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self._lifecycle = lifecycle
        self._settings = settings
        self._state_lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime: _ContinuousOwnerRuntime | None = None
        self._bootstrap_error: BaseException | None = None
        self._shutdown_future: concurrent.futures.Future[None] | None = None
        self._closed = False

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
        initial_weights: Any,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        runtime, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(
            runtime.next_iteration(
                list(prompts),
                group_size=group_size,
                runtime_debug=runtime_debug,
                initial_weights=initial_weights,
                next_prompts=(None if next_prompts is None else list(next_prompts)),
            ),
            loop,
        )
        return await _await_owner_future(future)

    async def commit_weights(self, prepared_weights: Any) -> RolloutStats:
        runtime, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(
            runtime.commit_weights(prepared_weights),
            loop,
        )
        return await _await_owner_future(future)

    def reset(self) -> None:
        with self._state_lock:
            if self._thread is None:
                return
        runtime, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(runtime.reset(), loop)
        future.result(timeout=_OWNER_STOP_TIMEOUT_S)

    async def shutdown(self) -> None:
        with self._state_lock:
            already_closed = self._closed
        if already_closed:
            await self._wait_until_stopped()
            return
        runtime, loop = self._ensure_thread()
        with self._state_lock:
            future = self._shutdown_future
            if future is None:
                future = asyncio.run_coroutine_threadsafe(runtime.shutdown(), loop)
                self._shutdown_future = future
                future.add_done_callback(
                    lambda done: self._finish_shutdown(done, loop),
                )
        try:
            await _await_owner_future(future)
        except BaseException:
            with self._state_lock:
                if self._shutdown_future is future and future.done():
                    self._shutdown_future = None
            raise
        await self._wait_until_stopped()

    def _finish_shutdown(
        self,
        future: concurrent.futures.Future[None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            future.result()
        except BaseException:
            with self._state_lock:
                if self._shutdown_future is future:
                    self._shutdown_future = None
            return
        with self._state_lock:
            self._closed = True
        loop.call_soon_threadsafe(loop.stop)

    async def _wait_until_stopped(self) -> None:
        stopped = await asyncio.to_thread(
            self._stopped.wait,
            _OWNER_STOP_TIMEOUT_S,
        )
        if not stopped:
            raise TimeoutError("continuous rollout owner thread did not stop")
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0)

    def _ensure_thread(
        self,
    ) -> tuple[_ContinuousOwnerRuntime, asyncio.AbstractEventLoop]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("continuous rollout owner is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name="vrl-continuous-rollout-owner",
                    daemon=True,
                )
                self._thread.start()
        if not self._ready.wait(_OWNER_START_TIMEOUT_S):
            raise TimeoutError("continuous rollout owner thread did not start")
        if self._bootstrap_error is not None:
            raise RuntimeError(
                "continuous rollout owner failed to start"
            ) from self._bootstrap_error
        runtime = self._runtime
        loop = self._loop
        if runtime is None or loop is None:  # pragma: no cover - guarded by ready
            raise RuntimeError("continuous rollout owner started without a runtime")
        return runtime, loop

    def _thread_main(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            runtime = _ContinuousOwnerRuntime(
                lifecycle=self._lifecycle,
                settings=self._settings,
            )
            with self._state_lock:
                self._loop = loop
                self._runtime = runtime
            self._ready.set()
            loop.run_forever()
        except BaseException as error:  # bootstrap failures must reach the caller
            self._bootstrap_error = error
            self._ready.set()
        finally:
            if loop is not None:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True),
                    )
                loop.close()
            self._stopped.set()


async def _await_owner_future(future: concurrent.futures.Future[_T]) -> _T:
    # asyncio.wrap_future normally propagates waiter cancellation into the
    # concurrent future.  Shielding preserves owner state transitions and
    # terminal cleanup when the trainer-side waiter is cancelled.
    return await asyncio.shield(asyncio.wrap_future(future))


__all__ = ["ContinuousRolloutOwner"]
