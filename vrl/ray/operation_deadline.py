"""Monotonic deadlines for driver-owned Ray calls."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.ray.dependencies import require_ray
from vrl.runtime_errors import TerminalRuntimeError


class RayOperationTimeout(TimeoutError, TerminalRuntimeError):
    """A driver-side Ray call exceeded its configured wall-clock budget."""

    def __init__(
        self,
        operation: str,
        timeout_s: float,
        *,
        context: str | None = None,
    ) -> None:
        self.operation = operation
        self.timeout_s = float(timeout_s)
        self.context = context
        message = f"Ray operation {operation!r} exceeded {self.timeout_s:g}s"
        if context:
            message = f"{message} ({context})"
        super().__init__(message)


class RayOperationCancelled(TerminalRuntimeError):
    """A caller abandoned submitted Ray work whose actor state is now unknown."""

    def __init__(
        self,
        operation: str,
        *,
        context: str | None = None,
    ) -> None:
        self.operation = operation
        self.context = context
        message = f"submitted Ray operation {operation!r} was cancelled"
        if context:
            message = f"{message} ({context})"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RayCallDeadline:
    """One absolute deadline starting at the driver submission boundary."""

    # Protocol identity consumed by the raised terminal error.
    operation: str
    # Behavior-consumed by monotonic expiry and every blocking wait.
    timeout_s: float
    # Provenance-only: identifies the worker/request boundary in diagnostics.
    context: str | None = None
    _expires_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timeout_s = validate_ray_timeout(self.timeout_s)
        if not self.operation:
            raise ValueError("Ray deadline operation must not be empty")
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "_expires_at", time.monotonic() + timeout_s)

    @property
    def expires_at(self) -> float:
        """Return the monotonic expiry used to compare concurrent calls."""

        return self._expires_at

    def remaining_s(self) -> float:
        """Return the non-negative budget left for a blocking wait."""

        return max(0.0, self._expires_at - time.monotonic())

    def timeout_error(self) -> RayOperationTimeout:
        """Build the stable protocol error for this call."""

        return RayOperationTimeout(
            self.operation,
            self.timeout_s,
            context=self.context,
        )


def validate_ray_timeout(
    timeout_s: float,
    *,
    name: str = "timeout_s",
) -> float:
    """Validate one public or direct-constructor Ray timeout value."""

    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return timeout_s


def cancel_ray_refs(
    ray: Any | None,
    refs: Iterable[Any],
    *,
    root_error: BaseException | None,
) -> tuple[Exception, ...]:
    """Best-effort cancel submitted refs and preserve any operation root."""

    try:
        ray = ray if ray is not None else require_ray()
    except Exception as error:
        if root_error is not None:
            root_error.add_note(f"Ray ref cancellation could not start: {error!r}")
        return (error,)

    failures: list[Exception] = []
    for ref in refs:
        try:
            # Ray forbids force=True for actor tasks. Correctness comes from the
            # owner's subsequent actor kill, not from assuming this interrupts
            # synchronous model code.
            ray.cancel(ref, force=False)
        except Exception as error:
            failures.append(error)
    if failures and root_error is not None:
        root_error.add_note(
            "Ray ref cancellation incomplete: "
            f"{len(failures)} cancellation attempt(s) failed; first={failures[0]!r}",
        )
    return tuple(failures)


def get_ray_refs(
    ray: Any,
    refs: Sequence[Any],
    *,
    operation: str,
    timeout_s: float,
    context: str | None = None,
) -> Any:
    """Bound one synchronous ``ray.get`` barrier and cancel refs on timeout."""

    if not refs:
        return []
    deadline = RayCallDeadline(operation, timeout_s, context=context)
    try:
        return ray.get(list(refs), timeout=deadline.remaining_s())
    except ray.exceptions.GetTimeoutError as cause:
        error = deadline.timeout_error()
        cancel_ray_refs(ray, refs, root_error=error)
        raise error from cause


__all__ = [
    "RayCallDeadline",
    "RayOperationCancelled",
    "RayOperationTimeout",
    "cancel_ray_refs",
    "get_ray_refs",
    "validate_ray_timeout",
]
