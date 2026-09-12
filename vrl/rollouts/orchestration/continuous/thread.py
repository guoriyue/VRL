"""Dedicated event-loop owner for disaggregated continuous rollout.

The trainer's asyncio loop must remain free to run synchronous forward/backward
work without starving rollout admission and completion harvesting.  This module
owns the continuous producer, queue, consumer, and every asynchronous controller
operation on one dedicated thread/event loop.  The trainer side communicates
only through ``concurrent.futures.Future`` command boundaries.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.scored_queue import ScoredRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats
from vrl.utils.validation import require_int

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_OWNER_START_TIMEOUT_S = 10.0
_OWNER_STOP_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class _InstalledPromptBatch:
    """Owner-side identity of one producer batch awaiting trainer consumption."""

    prompts: tuple[Any, ...]
    group_size: int

    def matches_presented_prompts(self, presented_prompts: list[Any]) -> bool:
        """Fail closed when a prompt type has non-scalar or invalid equality."""

        if len(self.prompts) != len(presented_prompts):
            return False
        for installed_prompt, presented_prompt in zip(
            self.prompts,
            presented_prompts,
            strict=True,
        ):
            if installed_prompt is presented_prompt:
                continue
            try:
                if not bool(installed_prompt == presented_prompt):
                    return False
            except (TypeError, ValueError, RuntimeError):
                return False
        return True


class _ContinuousRolloutController:
    """Continuous pipeline state that is touched only by the owner loop."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.lifecycle = lifecycle
        # The validated carrier travels whole; only derived values are unpacked.
        self.settings = settings
        self.max_ready_bytes = settings.max_ready_bytes_mb * _MB
        self.staleness = StalenessPolicy(
            max_stale_policy_versions=settings.max_stale_policy_versions,
        )

        self.queue: ScoredRolloutQueue | None = None
        self.consumer: ContinuousRolloutConsumer | None = None
        self.producer: ContinuousRolloutProducer | None = None
        self._installed_prompt_batch: _InstalledPromptBatch | None = None

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
            require_int(group_size, path="continuous prompt batch.group_size", minimum=1)
            if not prompts:
                raise ValueError("continuous rollout requires at least one prompt")
            if next_prompts is not None and not next_prompts:
                raise ValueError("continuous prefetch prompts must be non-empty")
            # Load-bearing local: pipeline startup pushes the initial weights
            # before the consumer drains an iteration, so the object that will
            # own these timings does not exist yet. Merged in below.
            startup_stats = RolloutStats()

            if self.producer is None:
                await self._start_pipeline(
                    prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                    initial_weights=initial_weights,
                    stats=startup_stats,
                )
            elif self._installed_prompt_batch is None:
                self._set_prompt_batch(
                    prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                )
            elif not self._installed_prompt_batch.matches_presented_prompts(prompts):
                raise RuntimeError(
                    "continuous prefetch prompt batch does not match the next prompts "
                    "presented by the trainer",
                )
            elif self._installed_prompt_batch.group_size != group_size:
                raise RuntimeError(
                    "continuous prefetch group size does not match the batch "
                    "presented by the trainer: "
                    f"expected={self._installed_prompt_batch.group_size}, "
                    f"requested={group_size}",
                )
            assert self.consumer is not None
            assert self.producer is not None
            current_policy_version = self.lifecycle.current_policy_version()
            batch_id = self.producer.current_batch_id
            prefetch_next_batch_early = self.settings.split_generation_reward
            prefetched_prompt_batch: _InstalledPromptBatch | None = None
            if prefetch_next_batch_early and next_prompts is not None:
                self.producer.append_prompt_batch(
                    next_prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                )
                assert self.queue is not None
                self.queue.set_item_limit(len(prompts) + len(next_prompts))
                prefetched_prompt_batch = _InstalledPromptBatch(tuple(next_prompts), group_size)
                self.producer.admit_now()

            iteration = await self.consumer.collect_iteration(
                expected_group_count=len(prompts),
                prompt_batch_id=batch_id,
                current_policy_version=current_policy_version,
                wait_timeout_s=self.settings.wait_timeout_s,
                poll_interval_s=self.settings.queue_poll_interval_s,
                producer_state=self.producer.state,
            )
            self._installed_prompt_batch = None
            prefetch_next_batch_requested = float(next_prompts is not None)
            if prefetch_next_batch_early:
                self.producer.consume_prompt_batch(batch_id)
                self._installed_prompt_batch = prefetched_prompt_batch
                assert self.queue is not None
                self.queue.set_item_limit(1 if next_prompts is None else len(next_prompts))
            elif next_prompts is not None:
                # Debug metadata belongs to generation time. This prefetch runs
                # during the current training step, even when the trainer consumes
                # it after state.step (and therefore runtime_debug) changes.
                self._set_prompt_batch(
                    next_prompts,
                    group_size=group_size,
                    runtime_debug=runtime_debug,
                )
            iteration.stats.merge(startup_stats)
            # Preserve the persisted metrics schema used by existing training logs.
            iteration.stats.observe_gauge(
                "continuous.lookahead_requested",
                prefetch_next_batch_requested,
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
        self.producer.set_prompt_batch(
            list(prompts),
            group_size=group_size,
            runtime_debug=runtime_debug,
        )
        self.queue.set_item_limit(len(prompts))
        self._installed_prompt_batch = _InstalledPromptBatch(
            prompts=tuple(prompts),
            group_size=group_size,
        )
        self.producer.admit_now()

    async def commit_weights(self, prepared_weights: Any) -> RolloutStats:
        """Commit one main-thread snapshot and reopen admission only on success."""

        async def operation() -> RolloutStats:
            stats = RolloutStats()
            producer = self.producer
            if producer is None:
                await self.lifecycle.push_prepared_weights(prepared_weights, stats)
                return stats

            non_draining = self.lifecycle.supports_non_draining_weight_sync()
            with stats.phase("continuous.weight_sync_pause_s"):
                producer.pause_admission()
                # There is deliberately no finally-resume here.  A partial worker
                # update leaves the fleet's installed version unknown, so failure
                # keeps admission closed and _run_command quarantines the owner.
                if not non_draining:
                    await producer.drain_prompt_batch(
                        wait_timeout_s=self.settings.wait_timeout_s,
                    )
                await self.lifecycle.push_prepared_weights(prepared_weights, stats)
                assert self.consumer is not None
                self.consumer.validate_ready_versions(
                    current_policy_version=self.lifecycle.current_policy_version(),
                )
                producer.resume_admission()
            stats.observe_gauge(
                "continuous.weight_sync_barrier_mode",
                float(non_draining),
            )
            return stats

        return await self._run_command(operation)

    async def reset(self) -> None:
        """Clear continuous queue/producer state without touching the controller."""

        await self._run_command(self._stop_pipeline)

    async def shutdown(self) -> None:
        """Cancel owner commands, stop production, and close its controller once."""

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

    async def _run_command[T](self, operation: Callable[[], Awaitable[T]]) -> T:
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
        # Stop admission before controller cleanup.  RayGenerationRuntime also
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
        stats: RolloutStats,
    ) -> None:
        # The trainer exported this immutable CPU snapshot before crossing the
        # owner boundary. Worker ACK validation happens inside the weight-sync
        # stack before push_prepared_weights publishes the committed version.
        # None means the persistent controller already owns initialized weights.
        if initial_weights is not None:
            await self.lifecycle.push_prepared_weights(initial_weights, stats)

        self.queue = ScoredRolloutQueue(
            max_items=len(prompts),
            max_bytes=self.max_ready_bytes,
        )
        self.consumer = ContinuousRolloutConsumer(
            queue=self.queue,
            staleness=self.staleness,
            settings=self.settings,
        )
        self.producer = ContinuousRolloutProducer(
            lifecycle=self.lifecycle,
            queue=self.queue,
            staleness=self.staleness,
            settings=self.settings,
        )
        self.producer.set_prompt_batch(
            list(prompts),
            group_size=group_size,
            runtime_debug=runtime_debug,
        )
        self._installed_prompt_batch = _InstalledPromptBatch(
            prompts=tuple(prompts),
            group_size=group_size,
        )
        await self.producer.start()
        self.producer.admit_now()

    async def _stop_pipeline(self) -> None:
        producer = self.producer
        queue = self.queue
        if producer is not None:
            await producer.stop(wait_timeout_s=_OWNER_STOP_TIMEOUT_S)
        if queue is not None:
            queue.clear()
        # Retain owners until cleanup succeeds so shutdown can retry a failure.
        self.producer = None
        self.queue = None
        self.consumer = None
        self._installed_prompt_batch = None

    def _attach_producer_metrics(self, iteration: RolloutIteration) -> None:
        if self.producer is None:
            return
        state = self.producer.state
        iteration.stats.observe_gauges(
            {f"continuous.{name}": value for name, value in self.producer.stage_stats().items()}
        )
        iteration.stats.observe_gauges(
            {
                "continuous.producer_inflight": float(self.producer.inflight_count),
                "continuous.producer_tick_count": float(state.tick_count),
                "continuous.producer_last_tick_gap_s": float(state.last_tick_gap_s),
                "continuous.producer_max_tick_gap_s": float(state.max_tick_gap_s),
                "continuous.producer_submitted": float(state.submitted_count),
                "continuous.producer_completed": float(state.completed_count),
                "continuous.producer_errors": float(state.error_count),
            },
        )
        # Cumulative-since-start snapshots, like producer_submitted above:
        # per-update deltas are computed offline from consecutive rows.
        for reason, seconds in state.backpressure_seconds.items():
            iteration.stats.observe_gauge(
                f"continuous.backpressure_{reason}_s",
                float(seconds),
            )
        for reason, entries in state.backpressure_entries.items():
            iteration.stats.observe_gauge(
                f"continuous.backpressure_{reason}_count",
                float(entries),
            )


class ContinuousRolloutThread:
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
        self._controller: _ContinuousRolloutController | None = None
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
        controller, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(
            controller.next_iteration(
                list(prompts),
                group_size=group_size,
                runtime_debug=runtime_debug,
                initial_weights=initial_weights,
                next_prompts=(None if next_prompts is None else list(next_prompts)),
            ),
            loop,
        )
        return await self._await_command(future)

    async def commit_weights(self, prepared_weights: Any) -> RolloutStats:
        controller, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(
            controller.commit_weights(prepared_weights),
            loop,
        )
        return await self._await_command(future)

    def reset(self) -> None:
        with self._state_lock:
            if self._thread is None:
                return
        controller, loop = self._ensure_thread()
        future = asyncio.run_coroutine_threadsafe(controller.reset(), loop)
        future.result(timeout=_OWNER_STOP_TIMEOUT_S)

    async def shutdown(self) -> None:
        with self._state_lock:
            already_closed = self._closed
        if already_closed:
            await self._wait_until_stopped()
            return
        controller, loop = self._ensure_thread()
        with self._state_lock:
            future = self._shutdown_future
            if future is None:
                future = concurrent.futures.Future()
                loop.call_soon_threadsafe(
                    loop.create_task, self._shutdown_controller(controller, loop, future)
                )
                self._shutdown_future = future
        await self._await_command(future)
        await self._wait_until_stopped()

    @staticmethod
    async def _await_command[T](future: concurrent.futures.Future[T]) -> T:
        """Wait without cancelling an owner command when its trainer waiter exits."""

        # Preserve owner state transitions and terminal cleanup independently
        # of trainer-side cancellation; wrap_future alone propagates cancellation.
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _shutdown_controller(
        self,
        controller: _ContinuousRolloutController,
        loop: asyncio.AbstractEventLoop,
        future: concurrent.futures.Future[None],
    ) -> None:
        """Own cleanup and publish its result independently of shutdown waiters."""

        try:
            await controller.shutdown()
        except BaseException as error:
            with self._state_lock:
                self._shutdown_future = None
            future.set_exception(error)
        else:
            with self._state_lock:
                self._closed = True
            # Publish before stopping the loop; no completion callback needs
            # another loop turn. Future callbacks must run outside the lock.
            future.set_result(None)
            loop.stop()

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
    ) -> tuple[_ContinuousRolloutController, asyncio.AbstractEventLoop]:
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
        controller = self._controller
        loop = self._loop
        if controller is None or loop is None:  # pragma: no cover - guarded by ready
            raise RuntimeError("continuous rollout owner started without a controller")
        return controller, loop

    def _thread_main(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            controller = _ContinuousRolloutController(
                lifecycle=self._lifecycle,
                settings=self._settings,
            )
            with self._state_lock:
                self._loop = loop
                self._controller = controller
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


__all__ = ["ContinuousRolloutThread"]
