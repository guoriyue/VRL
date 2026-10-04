"""Collector-facing lifecycle for Ray-distributed generation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.runtime_errors import TerminalRuntimeError, find_error_cause
from vrl.utils.lifecycle import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)
from vrl.utils.validation import require_int

logger = logging.getLogger(__name__)

_RaySessionFactory = Callable[[], Awaitable[RayGenerationSession]]


@dataclass(frozen=True, slots=True)
class _PendingPolicyInstall:
    """Trainer payload retained until fleet acknowledgement, before Ray serialization."""

    trainable_state: Any
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
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError(
                "Ray generation requires a live session or a deferred session factory",
            )
        if session is not None and session_factory is not None:
            raise ValueError(
                "Ray generation cannot start with both a session and a deferred factory",
            )
        self._session = session
        self._session_factory = session_factory

        self.lifecycle = RuntimeLifecycle(owner="rollout runtime")
        # Accepted targets stamp new requests immediately; installed tracks the
        # live fleet ACK, while pending retains the payload until that ACK exists.
        self.current_policy_version = initial_policy_version
        self._installed_policy_version = initial_policy_version if session is not None else None
        self._pending_install: _PendingPolicyInstall | None = None
        self._session_parked = False
        # Concurrent terminal failures each call shutdown; the lock lets the
        # first release the fleet while the rest observe the terminated phase.
        self._shutdown_lock = asyncio.Lock()

    @property
    def supports_non_draining_weight_sync(self) -> bool:
        """Whether the currently published session can sync without draining."""

        session = self._session
        return bool(session and session.supports_non_draining_weight_sync)

    async def _admit_operation(self, operation: str) -> None:
        """Reject closed admission and retry cleanup a failed shutdown left pending."""

        try:
            self.lifecycle.require_running(operation)
        except RuntimeLifecycleError:
            failure = self.lifecycle.failure
            if failure is None or self.lifecycle.phase is RuntimePhase.TERMINATED:
                raise
        else:
            return
        stable_failure = await self._terminalize_after_failure(failure)
        raise stable_failure

    async def activate(self) -> None:
        """Make a deferred or parked worker session ready for generation."""

        await self._admit_operation("activate")
        if self._session_factory is None:
            return
        await self._activate_once()

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        await self._admit_operation("generate")
        try:
            session = self._session
            if session is None or self._session_parked:
                raise RuntimeError(
                    "generate requires an active rollout runtime; "
                    "the rollout schedule must await activate() first",
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

    async def update_weights(self, trainable_state: Any, policy_version: int) -> None:
        """Install on active workers or stage the accepted target while inactive."""

        await self._admit_operation("update_weights")
        require_int(policy_version, path="policy_version", minimum=0)
        policy = _PendingPolicyInstall(
            trainable_state=trainable_state,
            policy_version=policy_version,
        )
        try:
            session = self._session
            if session is not None and not self._session_parked:
                await session.update_weights(
                    policy.trainable_state,
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
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            if self.lifecycle.failure is not None:
                await self._admit_operation("offload")
            await self.shutdown()
            return
        session = self._session
        if session is None or self._session_parked:
            return
        try:
            await session.sleep_engines()
            self._session_parked = True
        except BaseException as error:
            failure = await self._terminalize_after_failure(error)
            if isinstance(error, asyncio.CancelledError):
                if failure is not error:
                    error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def shutdown(self) -> None:
        """Close admission and release the one owned actor session.

        Concurrent terminal failures (several in-flight requests losing the same
        fleet) each call this; the lock lets the first release the fleet and the
        rest observe the terminated phase.
        """

        async with self._shutdown_lock:
            if self.lifecycle.phase is RuntimePhase.TERMINATED:
                return
            self.lifecycle.begin_shutdown()
            self._pending_install = None
            try:
                await self._teardown_session()
            except BaseException as error:
                root_failure = self.lifecycle.failure
                self.lifecycle.fail(error)
                if root_failure is not None and root_failure is not error:
                    raise error from root_failure
                raise
            self.lifecycle.finish_shutdown()

    async def _terminalize_after_failure(self, error: BaseException) -> BaseException:
        """Close admission and return the stable first failure after cleanup."""

        failure = self._publish_failure(error)
        self._pending_install = None
        try:
            await self.shutdown()
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

    def _publish_failure(self, error: BaseException) -> BaseException:
        """Publish the first failure and upgrade the owned session to force-close.

        A failed runtime never releases ranks gracefully, so an in-progress
        graceful shutdown is interrupted here rather than waited out.
        """

        terminal_error = find_error_cause(error, TerminalRuntimeError)
        failure = self.lifecycle.fail(terminal_error if terminal_error is not None else error)
        session = self._session
        if session is not None:
            session.force_close()
        return failure

    async def _activate_once(self) -> None:
        candidate: RayGenerationSession | None = None
        try:
            session = self._session
            if session is not None:
                if self._session_parked:
                    await session.wake_engines()
                    self._session_parked = False
                    pending = self._pending_install
                    if pending is not None and (
                        pending.policy_version != self._installed_policy_version
                    ):
                        await session.update_weights(
                            pending.trainable_state,
                            pending.policy_version,
                        )
                    if pending is not None:
                        with self.lifecycle.publication_guard(
                            "publish restored policy version",
                        ):
                            self._installed_policy_version = pending.policy_version
                            self._pending_install = None
                return

            factory = self._session_factory
            if factory is None:
                raise RuntimeError("deferred Ray generation has no session factory")
            # The launch runs actor startup in a worker thread that cancellation
            # cannot stop. A cancelled waiter therefore collects the fleet the
            # thread still produces, so the cleanup below can close it.
            launch = asyncio.ensure_future(factory())
            try:
                candidate = await asyncio.shield(launch)
            except asyncio.CancelledError:
                with contextlib.suppress(BaseException):
                    candidate = await launch
                raise
            pending = self._pending_install
            active_policy_version = self.current_policy_version
            if pending is not None:
                await candidate.update_weights(
                    pending.trainable_state,
                    pending.policy_version,
                )
                active_policy_version = pending.policy_version
            with self.lifecycle.publication_guard("publish activated session"):
                self._installed_policy_version = active_policy_version
                self._pending_install = None
                self._session = candidate
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
            failure = await self._terminalize_after_failure(error)
            if isinstance(error, asyncio.CancelledError):
                error.__cause__ = failure
                raise
            if failure is error:
                raise
            raise failure from failure.__cause__

    async def _teardown_session(self) -> None:
        session = self._session
        if session is not None:
            await session.close(force=self.lifecycle.failure is not None)
        self._session = None
        self._session_parked = False
        self._installed_policy_version = None


__all__ = ["RayGenerationRuntime"]
