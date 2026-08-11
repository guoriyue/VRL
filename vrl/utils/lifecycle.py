"""Terminal lifecycle state shared by the two inference engines.

Admission and draining belong to the owning schedule. This state only guards
the terminal boundary: work is accepted while RUNNING, failures close
admission through SHUTTING_DOWN, and successful resource teardown publishes
TERMINATED. Both the Ray generation runtime and the reward runtime drive this
one FSM so terminal semantics cannot drift between the engines.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum


class RuntimePhase(Enum):
    """Terminal state exposed for diagnostics and fail-fast errors."""

    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class RuntimeLifecycleError(RuntimeError):
    """A public runtime operation was invoked after admission closed."""

    def __init__(
        self,
        operation: str,
        phase: RuntimePhase,
        *,
        owner: str = "runtime",
    ) -> None:
        phase_name = phase.value.replace("_", " ")
        super().__init__(f"{operation} rejected: {owner} is {phase_name}")
        self.operation = operation
        self.phase = phase


class RuntimeLifecycle:
    """Fail-fast terminal lifecycle; schedules own pause/drain barriers."""

    def __init__(self, *, owner: str = "runtime") -> None:
        self._owner = owner
        self._phase = RuntimePhase.RUNNING
        self._failure: BaseException | None = None
        # Health probing publishes failures from an OS thread while runtime
        # operations transition on the trainer's event-loop thread. The lock is
        # also the linearization boundary for policy-version publication.
        self._lock = threading.Lock()

    @property
    def phase(self) -> RuntimePhase:
        with self._lock:
            return self._phase

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def require_running(self, operation: str) -> None:
        with self._lock:
            self._require_running_locked(operation)

    @contextmanager
    def publication_guard(self, operation: str) -> Iterator[None]:
        """Atomically publish local state only while terminal admission is open.

        The guarded body must contain only non-blocking assignments. Holding the
        lifecycle lock across an await would prevent the health-monitor thread
        from closing admission.
        """

        with self._lock:
            self._require_running_locked(operation)
            yield

    def fail(self, error: BaseException) -> BaseException:
        """Close admission and return the first failure retained as root cause."""

        with self._lock:
            if self._phase is RuntimePhase.TERMINATED:
                return self._failure or error
            if self._failure is None:
                self._failure = error
            self._phase = RuntimePhase.SHUTTING_DOWN
            return self._failure

    def begin_shutdown(self) -> None:
        """Idempotently close admission before the shared shutdown task starts."""

        with self._lock:
            if self._phase is RuntimePhase.RUNNING:
                self._phase = RuntimePhase.SHUTTING_DOWN

    def finish_shutdown(self) -> None:
        """Publish successful terminal cleanup."""

        with self._lock:
            if self._phase is not RuntimePhase.SHUTTING_DOWN:
                raise RuntimeLifecycleError(
                    "finish shutdown",
                    self._phase,
                    owner=self._owner,
                )
            self._phase = RuntimePhase.TERMINATED

    def _require_running_locked(self, operation: str) -> None:
        if self._phase is RuntimePhase.RUNNING:
            return
        error = RuntimeLifecycleError(operation, self._phase, owner=self._owner)
        if self._failure is not None:
            raise error from self._failure
        raise error


__all__ = [
    "RuntimeLifecycle",
    "RuntimeLifecycleError",
    "RuntimePhase",
]
