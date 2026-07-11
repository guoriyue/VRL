"""Terminal lifecycle state for one Ray generation runtime.

Rollout admission and draining belong to the rollout schedule.  This state only
guards the terminal boundary: work is accepted while RUNNING, failures close
admission through SHUTTING_DOWN, and successful resource teardown publishes
TERMINATED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimePhase(Enum):
    """Terminal state exposed for diagnostics and fail-fast errors."""

    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class RuntimeLifecycleError(RuntimeError):
    """A public runtime operation was invoked after admission closed."""

    def __init__(self, operation: str, phase: RuntimePhase) -> None:
        phase_name = phase.value.replace("_", " ")
        super().__init__(f"{operation} rejected: rollout runtime is {phase_name}")
        self.operation = operation
        self.phase = phase


@dataclass
class RuntimeLifecycle:
    """Fail-fast terminal lifecycle; schedules own pause/drain barriers."""

    phase: RuntimePhase = RuntimePhase.RUNNING
    failure: BaseException | None = None

    def require_running(self, operation: str) -> None:
        if self.phase is RuntimePhase.RUNNING:
            return
        error = RuntimeLifecycleError(operation, self.phase)
        if self.failure is not None:
            raise error from self.failure
        raise error

    def fail(self, error: BaseException) -> None:
        """Close admission and retain the first failure as the root cause."""

        if self.phase is RuntimePhase.TERMINATED:
            return
        self.failure = self.failure or error
        self.phase = RuntimePhase.SHUTTING_DOWN

    def begin_shutdown(self) -> None:
        """Idempotently close admission before the shared shutdown task starts."""

        if self.phase is RuntimePhase.RUNNING:
            self.phase = RuntimePhase.SHUTTING_DOWN

    def finish_shutdown(self) -> None:
        """Publish successful terminal cleanup."""

        if self.phase is not RuntimePhase.SHUTTING_DOWN:
            raise RuntimeLifecycleError("finish shutdown", self.phase)
        self.phase = RuntimePhase.TERMINATED


__all__ = [
    "RuntimeLifecycle",
    "RuntimeLifecycleError",
    "RuntimePhase",
]
