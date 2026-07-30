"""On-demand lifecycle owner for shared-GPU Ray generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.lifecycle_fsm import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.placement import RolePlacement
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause

if TYPE_CHECKING:
    from vrl.generation.ray.launcher import RayGenerationLauncher

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PendingPolicyInstall:
    """Trainer state retained only until one worker fleet acknowledges it."""

    state_ref: Any
    policy_version: int


class _OnDemandRayGenerationRuntime:
    """Launch, park, and wake a resident Ray runtime at GPU phase boundaries."""

    def __init__(
        self,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        launcher: RayGenerationLauncher,
        placement: RolePlacement,
    ) -> None:
        resources = config.resources
        if resources.lifecycle.rollout.mode != "on_demand":
            raise ValueError(
                "on-demand Ray generation requires a resolved on-demand rollout lifecycle plan",
            )
        contract = launch_inputs.launch_contract
        sleep_contract = contract
        if resources.rollout_gpus_per_worker > 0:
            sleep_contract = replace(
                contract,
                sleep_offload=True,
            )

        self._config = config
        # GPU workers pool their model in CuMemAllocator so offload/activate
        # release and restore physical pages. CPU-only worker tests retain the
        # ordinary allocator because there is no CUDA memory to pool.
        self._launch_inputs = replace(
            launch_inputs,
            launch_contract=sleep_contract,
        )
        self._launcher = launcher
        self._placement = placement
        self._inner_runtime: RayGenerationRuntime | None = None
        self._activation_task: asyncio.Task[RayGenerationRuntime] | None = None
        self._offload_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._pending_policy: _PendingPolicyInstall | None = None
        self._active_policy_version: int | None = None
        self._workers_offloaded = False
        self.lifecycle = RuntimeLifecycle()
        self.current_policy_version = contract.policy_version
        # The resident runtime may later prove retained version slots, but the
        # on-demand schedule still uses the conservative draining handoff.
        self.supports_non_draining_weight_sync = False

    @property
    def requires_driver_model_offload(self) -> bool:
        """Whether rollout activation must vacate the trainer's physical GPU."""

        return bool(
            self._config.resources.colocated and self._config.resources.rollout_gpus_per_worker > 0
        )

    @property
    def supports_weight_sync(self) -> bool:
        """Whether the normalized worker config accepts trainable-state pushes."""

        return bool(self._config.worker.sync_trainable_state)

    async def activate(self) -> None:
        """Launch or wake workers and complete any pending policy install."""

        self.lifecycle.require_running("activate")
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
        self.lifecycle.require_running("generate")
        try:
            inner_runtime = self._inner_runtime
            if inner_runtime is None or self._workers_offloaded:
                raise RuntimeError(
                    "generate requires an active rollout runtime; "
                    "the rollout schedule must await activate() first",
                )
            output = await inner_runtime.generate(request)
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

    def is_colocated(self) -> bool:
        return bool(self._config.resources.colocated)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Install on active workers or stage the accepted target while inactive."""

        # Validate before entering terminal quarantine: a rejected overlapping
        # call has not made installed worker state unknown.
        self.lifecycle.require_running("update_weights")
        if not self._config.worker.sync_trainable_state:
            raise RuntimeError("on-demand Ray generation has no weight sync")
        activation = self._activation_task
        if activation is not None and not activation.done():
            raise RuntimeError(
                "update_weights requires rollout activation to be idle; "
                "the rollout schedule must pause/drain before syncing",
            )

        policy = _PendingPolicyInstall(
            state_ref=state_ref,
            policy_version=int(policy_version),
        )
        try:
            inner_runtime = self._inner_runtime
            if inner_runtime is not None and not self._workers_offloaded:
                await inner_runtime.update_weights(
                    policy.state_ref,
                    policy.policy_version,
                )
                with (
                    inner_runtime.lifecycle.publication_guard(
                        "publish outer policy version",
                    ),
                    self.lifecycle.publication_guard("publish policy version"),
                ):
                    self._active_policy_version = policy.policy_version
                    self._pending_policy = None
                    self.current_policy_version = policy.policy_version
                return
            with self.lifecycle.publication_guard("publish policy version"):
                self._pending_policy = policy
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

    async def _terminalize_after_failure(
        self,
        error: BaseException,
        *,
        join_control_tasks: bool = True,
    ) -> BaseException:
        """Close admission and return the stable first failure after cleanup."""

        failure = self._publish_failure(error)
        # Terminal admission has no retry path, so retaining a staged full-model
        # CPU snapshot cannot contribute to recovery.
        self._pending_policy = None
        try:
            if join_control_tasks:
                await self.shutdown()
            else:
                # A control task cannot await shutdown because shutdown joins that
                # same task. The control-task owner tears down directly.
                await self._teardown_inner_runtime()
                self.lifecycle.finish_shutdown()
        except BaseException as cleanup_error:
            logger.error(
                "on-demand generation cleanup failed after operation error %r",
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
        """Publish the first failure and cancel untrustworthy control work."""

        terminal_error = find_error_cause(error, TerminalRuntimeError)
        proposed_failure = terminal_error if terminal_error is not None else error
        failure = self.lifecycle.fail(proposed_failure)
        terminal_failure = find_error_cause(failure, TerminalRuntimeError)
        if terminal_error is not None or terminal_failure is not None:
            current = asyncio.current_task()
            for task in (self._activation_task, self._offload_task):
                if task is not None and task is not current and not task.done():
                    task.cancel()
        return failure

    async def offload(self) -> None:
        """Park idle workers while retaining their actors and placement."""

        if self._inner_runtime is None and self._activation_task is None:
            return
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
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
            await self.shutdown()

    def _offload_finished(self, task: asyncio.Task[None]) -> None:
        if self._offload_task is task:
            self._offload_task = None
        if not task.cancelled():
            task.exception()

    async def _offload_once(self) -> None:
        try:
            if self.lifecycle.phase is not RuntimePhase.RUNNING:
                return
            inner_runtime = self._inner_runtime
            if inner_runtime is None:
                return
            if not self._workers_offloaded:
                await inner_runtime.sleep_workers()
                self._workers_offloaded = True
        except BaseException as error:
            shutdown_task = self._shutdown_task
            if shutdown_task is not None and not shutdown_task.done():
                failure = self._publish_failure(error)
            else:
                # A shielded control task can outlive every public waiter. It must
                # own terminal cleanup instead of relying on a waiter to re-enter
                # offload() and notice SHUTTING_DOWN.
                failure = await self._terminalize_after_failure(
                    error,
                    join_control_tasks=False,
                )
            if isinstance(error, asyncio.CancelledError):
                if failure is not error:
                    error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def shutdown(self) -> None:
        """Close admission and destroy the retained inner runtime exactly once."""

        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        task = self._shutdown_task
        if task is not None and task.done():
            self._shutdown_finished(task)
            task = None
        if task is None:
            self.lifecycle.begin_shutdown()
            # No operation can consume a staged install after admission closes.
            # Release what may be a full CPU model snapshot before cleanup waits.
            self._pending_policy = None
            task = asyncio.create_task(self._shutdown_once())
            self._shutdown_task = task
            task.add_done_callback(self._shutdown_finished)
        await asyncio.shield(task)

    async def _finish_control_wait_failure(self, error: BaseException) -> None:
        """Join facade cleanup and restore roots hidden by ``asyncio.shield``."""

        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            root_failure = self.lifecycle.failure
            try:
                await self.shutdown()
            except asyncio.CancelledError as cleanup_error:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    # shutdown() shields its shared cleanup task. Preserve caller
                    # cancellation while attaching the already-published root.
                    if root_failure is not None:
                        cleanup_error.__cause__ = root_failure
                    raise
                if root_failure is None:
                    raise
                logger.error(
                    "on-demand generation cleanup retry was cancelled after control error %r",
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
                # A control operation that already published a stable root keeps
                # that identity across repeated cleanup failures. With no prior
                # root, cleanup itself is the first material failure and escapes.
                if root_failure is None:
                    raise
                logger.error(
                    "on-demand generation cleanup retry failed after control error %r",
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

    def _shutdown_finished(self, task: asyncio.Task[None]) -> None:
        if self._shutdown_task is task:
            self._shutdown_task = None
        if not task.cancelled():
            task.exception()

    async def _shutdown_once(self) -> None:
        await self._join_control_tasks()
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        try:
            await self._teardown_inner_runtime()
        except BaseException as error:
            root_failure = self.lifecycle.failure
            self.lifecycle.fail(error)
            if root_failure is not None and root_failure is not error:
                raise error from root_failure
            raise
        self.lifecycle.finish_shutdown()

    async def _join_control_tasks(self) -> None:
        """Join launch/wake and offload tasks owned below the schedule."""

        current = asyncio.current_task()
        pending = [
            task
            for task in (self._activation_task, self._offload_task)
            if task is not None and task is not current
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _teardown_inner_runtime(self) -> None:
        inner_runtime = self._inner_runtime
        if inner_runtime is not None:
            await inner_runtime.shutdown()
        self._inner_runtime = None
        self._workers_offloaded = False
        self._active_policy_version = None

    def _activation_finished(
        self,
        task: asyncio.Task[RayGenerationRuntime],
    ) -> None:
        if self._activation_task is task:
            self._activation_task = None
        if not task.cancelled():
            task.exception()

    async def _activate_once(self) -> RayGenerationRuntime:
        try:
            inner_runtime = self._inner_runtime
            if inner_runtime is not None:
                await inner_runtime.activate()
                if self._workers_offloaded:
                    await inner_runtime.wake_workers()
                    # Record physical truth before restoring weights so cleanup
                    # treats a failed restore as an awake runtime.
                    self._workers_offloaded = False
                    pending = self._pending_policy
                    if pending is not None and (
                        pending.policy_version != self._active_policy_version
                    ):
                        await inner_runtime.update_weights(
                            pending.state_ref,
                            pending.policy_version,
                        )
                    if pending is not None:
                        with (
                            inner_runtime.lifecycle.publication_guard(
                                "publish restored policy version",
                            ),
                            self.lifecycle.publication_guard(
                                "publish restored policy version",
                            ),
                        ):
                            self._active_policy_version = pending.policy_version
                            self._pending_policy = None
                return inner_runtime

            candidate = await self._launcher.launch_async(
                self._config,
                self._launch_inputs,
                placement=self._placement,
            )
            try:
                pending = self._pending_policy
                active_policy_version = candidate.current_policy_version
                if pending is not None:
                    await candidate.update_weights(
                        pending.state_ref,
                        pending.policy_version,
                    )
                    active_policy_version = pending.policy_version
                with (
                    candidate.lifecycle.publication_guard("publish activated runtime"),
                    self.lifecycle.publication_guard("publish activated runtime"),
                ):
                    self._active_policy_version = active_policy_version
                    self._pending_policy = None
                    self._inner_runtime = candidate
            except BaseException as restore_error:
                try:
                    await candidate.shutdown()
                except BaseException as cleanup_error:
                    # Preserve ownership when candidate cleanup fails so a later
                    # outer shutdown can retry the retained handles.
                    self._inner_runtime = candidate
                    self._workers_offloaded = False
                    logger.error(
                        "candidate cleanup failed after restore error %r",
                        restore_error,
                        exc_info=cleanup_error,
                    )
                raise
            return candidate
        except BaseException as error:
            if (
                isinstance(error, RuntimeLifecycleError)
                and error.lifecycle is self.lifecycle
                and error.phase is not RuntimePhase.RUNNING
                and error.__cause__ is None
            ):
                # Graceful shutdown may close admission while an unpublished
                # candidate finishes launching. Candidate cleanup is already
                # owned by the inner boundary.
                raise
            shutdown_task = self._shutdown_task
            if shutdown_task is not None and not shutdown_task.done():
                failure = self._publish_failure(error)
            else:
                failure = await self._terminalize_after_failure(
                    error,
                    join_control_tasks=False,
                )
            if isinstance(error, asyncio.CancelledError):
                error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__
