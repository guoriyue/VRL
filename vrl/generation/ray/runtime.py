"""Collector-facing lifecycle for Ray-distributed generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.ray.health_monitor import RolloutWorkerHealthMonitor
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_group import RayActorHandle
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause
from vrl.utils.deadline import OperationDeadline
from vrl.utils.lifecycle import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)

logger = logging.getLogger(__name__)

_RaySessionFactory = Callable[[], Awaitable[RayGenerationSession]]


@dataclass(frozen=True, slots=True)
class _PendingPolicyInstall:
    """Trainer state retained only until one worker fleet acknowledges it."""

    state_ref: Any
    policy_version: int


class RayGenerationRuntime:
    """Own one Ray generation lifecycle and its optional live actor session.

    Dedicated-GPU runtimes start with a session. Shared-GPU runtimes receive a
    factory and create that session only after the rollout schedule has parked
    trainer state. Both shapes use this one admission, failure, and shutdown
    owner; ``RayGenerationSession`` never implements the public runtime protocol.
    """

    def __init__(
        self,
        *,
        session: RayGenerationSession | None,
        session_factory: _RaySessionFactory | None = None,
        initial_policy_version: int | None = None,
        supports_weight_sync: bool | None = None,
        colocated: bool = False,
        health_check_interval_s: float = 0.0,
        health_check_timeout_s: float = 30.0,
        health_check_first_wait_s: float = 0.0,
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError(
                "Ray generation requires a live session or a deferred session factory",
            )
        if session is not None and session_factory is not None:
            raise ValueError(
                "Ray generation cannot start with both a session and a deferred factory",
            )
        if colocated and session_factory is None:
            raise ValueError(
                "colocated Ray generation requires a deferred session factory",
            )

        self._session = session
        self._session_factory = session_factory
        self._colocated = bool(colocated)
        if session is None:
            if supports_weight_sync is None:
                raise ValueError(
                    "deferred Ray generation requires an explicit weight-sync capability",
                )
        else:
            session_supports_weight_sync = session.weight_sync is not None
            if supports_weight_sync is None:
                supports_weight_sync = session_supports_weight_sync
            elif bool(supports_weight_sync) != session_supports_weight_sync:
                raise ValueError(
                    "Ray generation runtime weight-sync capability does not match "
                    "its launched session",
                )
        self._supports_weight_sync = bool(supports_weight_sync)

        self.lifecycle = RuntimeLifecycle(owner="rollout runtime")
        # Accepted targets stamp new requests immediately; installed tracks the
        # live fleet ACK, while pending retains the payload until that ACK exists.
        self.current_policy_version = initial_policy_version
        self._installed_policy_version = initial_policy_version if session is not None else None
        self._pending_install: _PendingPolicyInstall | None = None
        self._session_parked = False

        self._activation_task: asyncio.Task[RayGenerationSession] | None = None
        self._offload_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._force_shutdown = False

        self._probed_samples_per_chunk: int | None = None
        self._samples_per_chunk_probe_lock = asyncio.Lock()
        self._health_check_timeout_s = float(health_check_timeout_s)
        self._health_monitor = RolloutWorkerHealthMonitor(
            self,
            interval_s=health_check_interval_s,
            timeout_s=health_check_timeout_s,
            first_wait_s=health_check_first_wait_s,
        )

    @property
    def _owned_workers(self) -> list[RayActorHandle]:
        """Fleet view consumed by the health-monitor framework adapter."""

        session = self._session
        return [] if session is None else session.workers

    @property
    def requires_driver_model_offload(self) -> bool:
        return self._colocated

    @property
    def supports_weight_sync(self) -> bool:
        return self._supports_weight_sync

    @property
    def supports_non_draining_weight_sync(self) -> bool:
        """Whether the currently published session can sync without draining."""

        session = self._session
        return bool(session and session.supports_non_draining_weight_sync)

    def start_health_monitoring(self) -> None:
        """Begin probing active workers. Idempotent and opt-in."""

        self._health_monitor.start()
        if self._session is not None and not self._session_parked:
            self._health_monitor.resume()

    async def preflight(self) -> None:
        """Probe every live worker once before the schedule starts.

        The generation twin of ``RewardRuntime.preflight``. Deferred or parked
        sessions skip — their launch/activate validates the fleet. A probe
        failure closes admission exactly like a monitor detection.
        """

        await self._admit_operation("preflight")
        session = self._session
        if session is None or self._session_parked:
            return
        refs = []
        for worker in session.workers:
            # Tolerate probe-less test doubles the way the health monitor does.
            remote = getattr(getattr(worker.actor, "health", None), "remote", None)
            if callable(remote):
                refs.append(remote())
        if not refs:
            return
        deadline = OperationDeadline(
            "generation.preflight",
            self._health_check_timeout_s,
            context=f"workers={len(refs)}",
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*refs),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            if deadline.remaining_s() > 0:
                self.lifecycle.fail(cause)
                raise
            failure = deadline.timeout_error()
            self.lifecycle.fail(failure)
            raise failure from cause
        except BaseException as error:
            self.lifecycle.fail(error)
            raise

    async def _admit_operation(self, operation: str) -> None:
        """Reject closed admission and finish cleanup after monitor failures."""

        try:
            self.lifecycle.require_running(operation)
        except RuntimeLifecycleError:
            failure = self.lifecycle.failure
            if failure is None or self.lifecycle.phase is RuntimePhase.TERMINATED:
                raise
        else:
            return
        stable_failure = await self._terminalize_after_failure(
            failure,
            force_shutdown=True,
        )
        raise stable_failure

    async def activate(self) -> None:
        """Make a deferred or parked worker session ready for generation."""

        await self._admit_operation("activate")
        if self._session_factory is None:
            return
        offload = self._offload_task
        if offload is not None and offload.done():
            self._offload_finished(offload)
            offload = None
        if offload is not None:
            try:
                await asyncio.shield(offload)
            except BaseException as error:
                await self._finish_control_wait_failure(error)
                raise
            await self._admit_operation("activate")
        task = self._activation_task
        if task is not None and task.done():
            self._activation_finished(task)
            task = None
        if task is None:
            task = asyncio.create_task(self._activate_once())
            self._activation_task = task
            task.add_done_callback(self._activation_finished)
        try:
            await asyncio.shield(task)
        except BaseException as error:
            await self._finish_control_wait_failure(error)
            raise

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        await self._admit_operation("generate")
        activation = self._activation_task
        if activation is not None and not activation.done():
            raise RuntimeError(
                "generate requires rollout activation to complete; "
                "the rollout schedule must await activate() before collection",
            )
        offload = self._offload_task
        if offload is not None and not offload.done():
            raise RuntimeError(
                "generate requires rollout offload to be idle; "
                "the rollout schedule must drain before the GPU handoff",
            )
        try:
            session = self._session
            if session is None or self._session_parked:
                raise RuntimeError(
                    "generate requires an active rollout runtime; "
                    "the rollout schedule must await activate() first",
                )
            if request.sampling.get("samples_per_chunk") == "auto":
                resolved = await self._resolve_probed_samples_per_chunk(
                    session,
                    request,
                )
                request = replace(
                    request,
                    sampling={**dict(request.sampling), "samples_per_chunk": resolved},
                )
            if request.policy_version is None and self.current_policy_version is not None:
                request = replace(
                    request,
                    policy_version=self.current_policy_version,
                )
            output = await session.executor.execute(request)
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
            failure = await self._terminalize_after_failure(error)
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def _resolve_probed_samples_per_chunk(
        self,
        session: RayGenerationSession,
        request: GenerationRequest,
    ) -> int:
        """Run the startup chunk-size probe once and cache the fleet verdict."""

        if self._probed_samples_per_chunk is not None:
            return self._probed_samples_per_chunk
        async with self._samples_per_chunk_probe_lock:
            if self._probed_samples_per_chunk is not None:
                return self._probed_samples_per_chunk
            self.lifecycle.require_running("probe generation chunk size")
            max_samples = max(1, int(request.samples_per_prompt))
            local_results = await session.executor.probe_chunk_sizes(
                request,
                max_samples=max_samples,
            )
            if not local_results:
                raise RuntimeError(
                    "samples_per_chunk: auto found no generation workers to probe",
                )
            resolved = min(result.samples_per_chunk for result in local_results)
            for result in local_results:
                logger.info(
                    "chunk-size probe: n=%d (budget=%.0fMB trials=%s)",
                    result.samples_per_chunk,
                    result.budget_bytes / 2**20,
                    [
                        (
                            trial.label,
                            trial.n,
                            "OOM"
                            if trial.oom
                            else f"{trial.peak_bytes / 2**20:.0f}MB/{trial.wall_s:.1f}s",
                        )
                        for trial in result.trials
                    ],
                )
            self._probed_samples_per_chunk = resolved
            return resolved

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Install on active workers or stage the accepted target while inactive."""

        await self._admit_operation("update_weights")
        if not self._supports_weight_sync:
            raise RuntimeError("Ray generation has no weight sync")
        activation = self._activation_task
        if activation is not None and not activation.done():
            raise RuntimeError(
                "update_weights requires rollout activation to be idle; "
                "the rollout schedule must pause/drain before syncing",
            )
        offload = self._offload_task
        if offload is not None and not offload.done():
            raise RuntimeError(
                "update_weights requires rollout offload to be idle; "
                "the rollout schedule must await the GPU handoff before syncing",
            )

        policy = _PendingPolicyInstall(
            state_ref=state_ref,
            policy_version=int(policy_version),
        )
        try:
            session = self._session
            if session is not None and not self._session_parked:
                await session.update_weights(
                    policy.state_ref,
                    policy.policy_version,
                )
                with self.lifecycle.publication_guard("publish policy version"):
                    self._installed_policy_version = policy.policy_version
                    self._pending_install = None
                    self.current_policy_version = policy.policy_version
                return
            with self.lifecycle.publication_guard("publish policy version"):
                self._pending_install = policy
                self.current_policy_version = policy.policy_version
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

    async def offload(self) -> None:
        """Park a deferred runtime's idle worker session at a GPU handoff."""

        if self._session_factory is None:
            return
        if self._session is None and self._activation_task is None:
            return
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            if self.lifecycle.failure is not None:
                await self._admit_operation("offload")
            await self.shutdown()
            return
        activation = self._activation_task
        if activation is not None and not activation.done():
            try:
                await asyncio.shield(activation)
            except BaseException as error:
                await self._finish_control_wait_failure(error)
                raise
            if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
                if self.lifecycle.failure is not None:
                    await self._admit_operation("offload")
                await self.shutdown()
                return
        task = self._offload_task
        if task is not None and task.done():
            self._offload_finished(task)
            task = None
        if task is None:
            task = asyncio.create_task(self._offload_once())
            self._offload_task = task
            task.add_done_callback(self._offload_finished)
        try:
            await asyncio.shield(task)
        except BaseException as error:
            await self._finish_control_wait_failure(error)
            raise
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            if self.lifecycle.failure is not None:
                await self._admit_operation("offload")
            await self.shutdown()

    async def _offload_once(self) -> None:
        try:
            if self.lifecycle.phase is not RuntimePhase.RUNNING:
                return
            session = self._session
            if session is None or self._session_parked:
                return
            self._health_monitor.pause()
            await session.sleep_workers()
            if self.lifecycle.failure is not None:
                self.lifecycle.require_running("complete worker sleep")
            self._session_parked = True
        except BaseException as error:
            shutdown_task = self._shutdown_task
            if shutdown_task is not None and not shutdown_task.done():
                failure = self._publish_failure(error, force_shutdown=True)
            else:
                failure = await self._terminalize_after_failure(
                    error,
                    join_control_tasks=False,
                    force_shutdown=True,
                )
            if isinstance(error, asyncio.CancelledError):
                if failure is not error:
                    error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def shutdown(self) -> None:
        """Close admission and release the one owned actor session."""

        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        task = self._shutdown_task
        if task is not None and task.done():
            self._shutdown_finished(task)
            task = None
        if task is None:
            self.lifecycle.begin_shutdown()
            self._pending_install = None
            task = asyncio.create_task(self._shutdown_once())
            self._shutdown_task = task
            task.add_done_callback(self._shutdown_finished)
        await asyncio.shield(task)

    async def _terminalize_after_failure(
        self,
        error: BaseException,
        *,
        join_control_tasks: bool = True,
        force_shutdown: bool = False,
    ) -> BaseException:
        """Close admission and return the stable first failure after cleanup."""

        failure = self._publish_failure(
            error,
            force_shutdown=force_shutdown,
        )
        self._pending_install = None
        try:
            if join_control_tasks:
                await self.shutdown()
            else:
                await self._teardown_session()
                self.lifecycle.finish_shutdown()
        except BaseException as cleanup_error:
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

    def _publish_failure(
        self,
        error: BaseException,
        *,
        force_shutdown: bool = False,
    ) -> BaseException:
        """Publish the first failure and upgrade the owned session to force-close."""

        terminal_error = find_error_cause(error, TerminalRuntimeError)
        proposed_failure = terminal_error if terminal_error is not None else error
        failure = self.lifecycle.fail(proposed_failure)
        terminal_failure = find_error_cause(failure, TerminalRuntimeError)
        if force_shutdown or terminal_error is not None or terminal_failure is not None:
            self._force_shutdown = True
            session = self._session
            if session is not None:
                session.force_close()
            current = asyncio.current_task()
            activation = self._activation_task
            if (
                activation is not None
                and activation is not current
                and not activation.done()
                and self._session is not None
            ):
                activation.cancel()
            offload = self._offload_task
            if offload is not None and offload is not current and not offload.done():
                offload.cancel()
        return failure

    async def _finish_control_wait_failure(self, error: BaseException) -> None:
        """Join shared cleanup and restore roots hidden by ``asyncio.shield``."""

        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            try:
                await self.shutdown()
            except asyncio.CancelledError as cleanup_error:
                root_failure = self.lifecycle.failure
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    if root_failure is not None:
                        cleanup_error.__cause__ = root_failure
                    raise
                if root_failure is None:
                    raise
                logger.error(
                    "generation cleanup retry was cancelled after control error %r",
                    error,
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
                root_failure.add_note(
                    "generation terminal cleanup retry was also cancelled",
                )
            except BaseException as cleanup_error:
                root_failure = self.lifecycle.failure
                current = asyncio.current_task()
                if (
                    isinstance(error, asyncio.CancelledError)
                    and current is not None
                    and current.cancelling()
                ):
                    error.__cause__ = root_failure or cleanup_error
                    if root_failure is None or root_failure is cleanup_error:
                        return
                if root_failure is None or root_failure is cleanup_error:
                    raise
                logger.error(
                    "generation cleanup retry failed after control error %r",
                    error,
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
                root_failure.add_note(
                    f"generation terminal cleanup retry also failed: {cleanup_error!r}",
                )
        if isinstance(error, asyncio.CancelledError):
            failure = self.lifecycle.failure
            if failure is not None:
                error.__cause__ = failure

    def _activation_finished(
        self,
        task: asyncio.Task[RayGenerationSession],
    ) -> None:
        if self._activation_task is task:
            self._activation_task = None
        if not task.cancelled():
            task.exception()

    def _offload_finished(self, task: asyncio.Task[None]) -> None:
        if self._offload_task is task:
            self._offload_task = None
        if not task.cancelled():
            task.exception()

    def _shutdown_finished(self, task: asyncio.Task[None]) -> None:
        if self._shutdown_task is task:
            self._shutdown_task = None
        if not task.cancelled():
            task.exception()

    async def _activate_once(self) -> RayGenerationSession:
        candidate: RayGenerationSession | None = None
        force_shutdown = False
        try:
            session = self._session
            if session is not None:
                if self._session_parked:
                    force_shutdown = True
                    await session.wake_workers()
                    self._session_parked = False
                    force_shutdown = False
                    if self.lifecycle.failure is not None:
                        self.lifecycle.require_running("complete worker wake")
                    pending = self._pending_install
                    if pending is not None and (
                        pending.policy_version != self._installed_policy_version
                    ):
                        await session.update_weights(
                            pending.state_ref,
                            pending.policy_version,
                        )
                    if pending is not None:
                        with self.lifecycle.publication_guard(
                            "publish restored policy version",
                        ):
                            self._installed_policy_version = pending.policy_version
                            self._pending_install = None
                self._health_monitor.resume()
                return session

            factory = self._session_factory
            if factory is None:
                raise RuntimeError("deferred Ray generation has no session factory")
            candidate = await factory()
            if (candidate.weight_sync is not None) != self._supports_weight_sync:
                raise RuntimeError(
                    "deferred Ray generation session reported a different "
                    "weight-sync capability than its runtime",
                )
            pending = self._pending_install
            active_policy_version = self.current_policy_version
            if pending is not None:
                await candidate.update_weights(
                    pending.state_ref,
                    pending.policy_version,
                )
                active_policy_version = pending.policy_version
            with self.lifecycle.publication_guard("publish activated session"):
                self._installed_policy_version = active_policy_version
                self._pending_install = None
                self._session = candidate
            self._health_monitor.resume()
            return candidate
        except BaseException as error:
            if candidate is not None and self._session is not candidate:
                try:
                    await candidate.close(force=True)
                except BaseException as cleanup_error:
                    self._session = candidate
                    logger.error(
                        "candidate session cleanup failed after activation error %r",
                        error,
                        exc_info=cleanup_error,
                    )
            if (
                isinstance(error, RuntimeLifecycleError)
                and error.phase is not RuntimePhase.RUNNING
                and error.__cause__ is None
            ):
                raise
            shutdown_task = self._shutdown_task
            if shutdown_task is not None and not shutdown_task.done():
                failure = self._publish_failure(error)
            else:
                failure = await self._terminalize_after_failure(
                    error,
                    join_control_tasks=False,
                    force_shutdown=force_shutdown,
                )
            if isinstance(error, asyncio.CancelledError):
                error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def _shutdown_once(self) -> None:
        current = asyncio.current_task()
        pending = [
            task
            for task in (self._activation_task, self._offload_task)
            if task is not None and task is not current
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        try:
            await self._teardown_session()
        except BaseException as error:
            root_failure = self.lifecycle.failure
            self.lifecycle.fail(error)
            if root_failure is not None and root_failure is not error:
                raise error from root_failure
            raise
        self.lifecycle.finish_shutdown()

    async def _teardown_session(self) -> None:
        await asyncio.to_thread(self._health_monitor.stop)
        session = self._session
        if session is not None:
            await session.close(
                force=self._force_shutdown or self.lifecycle.failure is not None,
            )
        self._session = None
        self._session_parked = False
        self._installed_policy_version = None


__all__ = ["RayGenerationRuntime"]
