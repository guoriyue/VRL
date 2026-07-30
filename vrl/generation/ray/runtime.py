"""Resident actor owner for Ray-distributed generation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import replace
from typing import Any

from vrl.generation.execution.types import (
    DistributedWorkerHandle,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.health_monitor import RolloutWorkerHealthMonitor
from vrl.generation.ray.lifecycle_fsm import RuntimeLifecycle, RuntimePhase
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.dependencies import require_ray
from vrl.ray.resource_cleanup import kill_and_retain
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause

logger = logging.getLogger(__name__)


class RayGenerationRuntime:
    """Own resident Ray workers, dispatch, health monitoring, and teardown."""

    def __init__(
        self,
        executor: RayGenerationExecutor | Any,
        *,
        weight_sync: GenerationWeightSync | None = None,
        owned_workers: list[DistributedWorkerHandle] | None = None,
        colocated: bool = False,
        health_check_interval_s: float = 0.0,
        health_check_timeout_s: float = 30.0,
        health_check_first_wait_s: float = 0.0,
    ) -> None:
        if executor is None:
            raise ValueError("resident Ray generation requires an executor")
        self.executor = executor
        self.weight_sync = weight_sync
        executor_dispatcher = getattr(executor, "actor_dispatcher", None)
        weight_sync_dispatcher = getattr(weight_sync, "actor_dispatcher", None)
        if (
            executor_dispatcher is not None
            and weight_sync_dispatcher is not None
            and executor_dispatcher is not weight_sync_dispatcher
        ):
            raise ValueError(
                "Ray generation and weight sync must share one actor dispatcher",
            )
        self._owned_workers = list(owned_workers or [])
        self._colocated = bool(colocated)
        # Operation deadlines bound active business calls; this complementary
        # monitor covers process death between calls and independently verifies
        # that the actor's health concurrency group remains reachable.
        self._health_monitor = RolloutWorkerHealthMonitor(
            self,
            interval_s=health_check_interval_s,
            timeout_s=health_check_timeout_s,
            first_wait_s=health_check_first_wait_s,
        )
        # Rollout schedules own pause/drain. The runtime lifecycle only closes
        # terminal admission and records the first cleanup-worthy failure.
        self.lifecycle = RuntimeLifecycle()
        self._shutdown_task: asyncio.Task[None] | None = None
        self._release_wait_task: asyncio.Task[Any] | None = None
        self._force_shutdown = False
        self.current_policy_version: int | None = None
        # Set True by the launcher when every resident worker retains versioned
        # trainable-state slots, which lets the continuous schedule skip the drain
        # bubble. Default False keeps the safe draining barrier. Read as a plain
        # attribute (not a method) via RolloutLifecycle.supports_non_draining_weight_sync.
        self.supports_non_draining_weight_sync = False
        # Resolved by the startup chunk-size probe on the first request that
        # carries sampling.samples_per_chunk == "auto". The actor-owning runtime
        # survives sleep/wake, so that lifecycle reuses this run-level verdict.
        self._probed_samples_per_chunk: int | None = None
        self._samples_per_chunk_probe_lock = asyncio.Lock()

    @property
    def requires_driver_model_offload(self) -> bool:
        """Resident workers never require a phase-boundary trainer offload."""

        return False

    @property
    def supports_weight_sync(self) -> bool:
        """Whether trainer weight pushes have a resident worker consumer."""

        return self.weight_sync is not None

    def start_health_monitoring(self) -> None:
        """Begin probing this runtime's own workers. Idempotent and opt-in."""

        if self._health_monitor.start():
            self._health_monitor.resume()

    async def activate(self) -> None:
        """Validate admission; resident workers are already active."""

        self.lifecycle.require_running("activate")

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.lifecycle.require_running("generate")
        try:
            if request.sampling.get("samples_per_chunk") == "auto":
                # Resolve at the actor-owning runtime so a failed probe enters the
                # same terminal boundary as every other submitted generation RPC.
                resolved = await self._resolve_probed_samples_per_chunk(request)
                request = replace(
                    request,
                    sampling={**dict(request.sampling), "samples_per_chunk": resolved},
                )
            if request.policy_version is None and self.current_policy_version is not None:
                request = replace(request, policy_version=self.current_policy_version)
            output = await self.executor.execute(request)
            self.lifecycle.require_running("complete generation")
            return output
        except asyncio.CancelledError as error:
            if (
                find_error_cause(error, TerminalRuntimeError) is not None
                or self.lifecycle.failure is not None
            ):
                failure = await self._terminalize_after_failure(error)
                error.__cause__ = failure
            raise
        except BaseException as error:
            if (
                find_error_cause(error, TerminalRuntimeError) is None
                and self.lifecycle.failure is None
            ):
                raise
            # A terminal operation error or an earlier health failure makes all
            # results from the fleet untrustworthy. Close admission and destroy
            # the actors before a later local processing error can replace it.
            failure = await self._terminalize_after_failure(error)
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def _resolve_probed_samples_per_chunk(
        self,
        request: GenerationRequest,
    ) -> int:
        """Run the startup chunk-size probe once and cache the verdict.

        vLLM shape (EngineCore init -> determine_available_memory): sizing runs
        before the first real request without a separate user command. Every
        worker is probed concurrently; the fleet answer is the minimum.
        """

        if self._probed_samples_per_chunk is not None:
            return self._probed_samples_per_chunk
        # Multiple rollout groups may reach the first generation concurrently.
        # Serialize before submission so only the owner probes its synchronous
        # actors and local waiters do not spend a remote-operation budget.
        async with self._samples_per_chunk_probe_lock:
            if self._probed_samples_per_chunk is not None:
                return self._probed_samples_per_chunk
            self.lifecycle.require_running("probe generation chunk size")
            max_samples = max(1, int(request.samples_per_prompt))
            local_results = await self.executor.probe_chunk_sizes(
                request,
                max_samples=max_samples,
            )
            if not local_results:
                raise RuntimeError(
                    "samples_per_chunk: auto found no generation workers to probe",
                )
            resolved = min(int(result["samples_per_chunk"]) for result in local_results)
            for result in local_results:
                logger.info(
                    "chunk-size probe: n=%d (budget=%.0fMB trials=%s)",
                    int(result["samples_per_chunk"]),
                    result["budget_bytes"] / 2**20,
                    [
                        (
                            trial["label"],
                            trial["n"],
                            "OOM"
                            if trial["oom"]
                            else f"{trial['peak_bytes'] / 2**20:.0f}MB/{trial['wall_s']:.1f}s",
                        )
                        for trial in result["trials"]
                    ],
                )
            self._probed_samples_per_chunk = resolved
            return resolved

    def is_colocated(self) -> bool:
        return self._colocated

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        # Contract failures do not make installed worker state unknown and
        # therefore must not enter terminal quarantine.
        self.lifecycle.require_running("update_weights")
        weight_sync = self.weight_sync
        if weight_sync is None:
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")

        try:
            resolved_policy_version = int(policy_version)
            await weight_sync.push_to_rollout_workers(
                state_ref,
                resolved_policy_version,
            )
            with self.lifecycle.publication_guard("publish policy version"):
                self.current_policy_version = resolved_policy_version
        except asyncio.CancelledError as error:
            if (
                find_error_cause(error, TerminalRuntimeError) is not None
                or self.lifecycle.failure is not None
            ):
                failure = await self._terminalize_after_failure(error)
                error.__cause__ = failure
            raise
        except BaseException as error:
            failure = await self._terminalize_after_failure(error)
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def _terminalize_after_failure(
        self,
        error: BaseException,
    ) -> BaseException:
        """Close admission and return the stable first failure after cleanup."""

        failure = self._publish_failure(error)
        try:
            await self.shutdown()
        except BaseException as cleanup_error:
            # The failed install/ACK is the transaction's root cause. Cleanup is
            # retryable by the schedule owner, so it must never replace that
            # first failure at the trainer boundary.
            logger.error(
                "generation terminal cleanup failed after operation error %r",
                error,
                exc_info=(
                    type(cleanup_error),
                    cleanup_error,
                    cleanup_error.__traceback__,
                ),
            )
            failure.add_note(f"generation terminal cleanup also failed: {cleanup_error!r}")
        return failure

    def _publish_failure(self, error: BaseException) -> BaseException:
        """Publish the first failure and synchronously upgrade terminal cleanup."""

        # Publish the operation root before upgrading an existing graceful
        # shutdown. Cancellation of its release barrier must never win the
        # first-failure slot over the terminal error that required the upgrade.
        terminal_error = find_error_cause(error, TerminalRuntimeError)
        proposed_failure = terminal_error if terminal_error is not None else error
        failure = self.lifecycle.fail(proposed_failure)
        terminal_failure = find_error_cause(failure, TerminalRuntimeError)
        if terminal_error is not None or terminal_failure is not None:
            self._force_shutdown = True
            # A terminal distributed operation proves that graceful RPC progress
            # cannot be trusted. Cancel a runtime-owned graceful release barrier
            # so an existing shutdown upgrades immediately to forceful teardown.
            current = asyncio.current_task()
            release_wait_task = self._release_wait_task
            if (
                release_wait_task is not None
                and release_wait_task is not current
                and not release_wait_task.done()
            ):
                release_wait_task.cancel()
        return failure

    async def offload(self) -> None:
        """Keep resident workers active until terminal shutdown."""

    async def shutdown(self) -> None:
        """Close admission and tear down owned resources exactly once.

        Shielding the shared task keeps cancellation of one caller from
        cancelling cleanup for every caller. A failed cleanup remains
        SHUTTING_DOWN and clears the task so a later shutdown call can retry
        owned resources. The schedule stops and joins generation before calling
        shutdown.
        """

        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return None
        task = self._shutdown_task
        if task is not None and task.done():
            self._shutdown_finished(task)
            task = None
        if task is None:
            self.lifecycle.begin_shutdown()
            task = asyncio.create_task(self._shutdown_once())
            self._shutdown_task = task
            task.add_done_callback(self._shutdown_finished)
        await asyncio.shield(task)
        return None

    def _shutdown_finished(self, task: asyncio.Task[None]) -> None:
        if self._shutdown_task is task:
            self._shutdown_task = None
        if not task.cancelled():
            task.exception()  # retrieve failures even if every waiter was cancelled

    async def _shutdown_once(self) -> None:
        try:
            await self._teardown_owned_resources()
        except BaseException as error:
            root_failure = self.lifecycle.failure
            self.lifecycle.fail(error)
            if root_failure is not None and root_failure is not error:
                raise error from root_failure
            raise
        self.lifecycle.finish_shutdown()

    async def _teardown_owned_resources(self) -> None:
        # stop() joins the monitor thread for a bounded probe interval. Keep
        # that synchronous lifecycle operation off the runtime event loop so
        # concurrent shutdown/terminal waiters can continue making progress.
        await asyncio.to_thread(self._health_monitor.stop)
        if not self._owned_workers:
            return None
        ray = require_ray()
        doomed = [worker.actor for worker in self._owned_workers if worker.actor is not None]
        logger.info(
            "runtime shutdown: killing %d owned worker actor(s)",
            len(doomed),
        )
        # A timed-out business call may still occupy the actor's default
        # concurrency group. Waiting for release_policy on that same group would
        # add another 60 seconds before the only reliable action: ray.kill.
        if self.lifecycle.failure is None and not self._force_shutdown:
            release_refs: list[Any] = []
            for worker in self._owned_workers:
                actor = worker.actor
                if actor is None:
                    continue
                with contextlib.suppress(Exception):
                    release_refs.append(actor.release_policy.remote())
            if release_refs:
                release_wait_task = asyncio.create_task(
                    asyncio.to_thread(ray.get, release_refs, timeout=60),
                )
                self._release_wait_task = release_wait_task
                try:
                    await release_wait_task
                except asyncio.CancelledError:
                    if not self._force_shutdown:
                        raise
                except Exception:
                    pass
                finally:
                    if self._release_wait_task is release_wait_task:
                        self._release_wait_task = None
        surviving, worker_failures = kill_and_retain(
            ray,
            self._owned_workers,
            lambda worker: worker.actor,
        )
        self._owned_workers[:] = surviving

        failures = [error for _, error in worker_failures]
        if failures:
            raise RuntimeError(
                "Ray runtime cleanup incomplete: "
                f"{len(worker_failures)} worker actor kill(s) failed",
            ) from failures[0]
        return None

    async def sleep_workers(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Offload every owned worker's model to host RAM, freeing the GPU.

        The workers stay alive (process + placement bundle retained); only their
        weights leave the GPU. Failures are not suppressed: a worker that fails to
        offload would otherwise hold the GPU a colocated trainer is about to use.
        """
        # Parking transitions and parked intervals are outside the active
        # serving SLA, so lifecycle policy pauses monitoring across them.
        self._health_monitor.pause()
        if not self._owned_workers:
            return ()
        missing_actor_ids = tuple(
            worker.worker_id for worker in self._owned_workers if worker.actor is None
        )
        if missing_actor_ids:
            raise RuntimeError(
                f"generation workers have no actor: {missing_actor_ids}",
            )
        active_workers = tuple(
            worker for worker in self._owned_workers if worker.actor is not None
        )
        worker_ids = tuple(worker.worker_id for worker in active_workers)
        if len(set(worker_ids)) != len(worker_ids):
            raise RuntimeError(f"duplicate generation worker ids: {worker_ids}")
        refs = [worker.actor.sleep.remote() for worker in active_workers]
        values = await asyncio.wait_for(asyncio.gather(*refs), timeout=120) if refs else []
        snapshots: list[WorkerMemoryParkingSnapshot] = []
        for worker, value in zip(active_workers, values, strict=True):
            if not isinstance(value, WorkerMemoryParkingSnapshot):
                raise TypeError(
                    f"worker {worker.worker_id!r} returned invalid memory-parking "
                    f"report {type(value).__name__}",
                )
            if value.worker_id != worker.worker_id:
                raise RuntimeError(
                    "mismatched worker memory-parking report: "
                    f"expected={worker.worker_id!r} actual={value.worker_id!r}",
                )
            value.validate()
            snapshots.append(value)
        return tuple(snapshots)

    async def wake_workers(self) -> None:
        """Restore every owned worker's model from host RAM back onto its GPU."""
        if not self._owned_workers:
            self._health_monitor.resume()
            return None
        refs = [
            worker.actor.wake.remote()
            for worker in self._owned_workers
            if worker.actor is not None
        ]
        if refs:
            await asyncio.wait_for(asyncio.gather(*refs), timeout=120)
        self._health_monitor.resume()
        return None


__all__ = ["RayGenerationRuntime"]
