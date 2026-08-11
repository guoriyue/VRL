"""Monotonic deadlines for driver-owned Ray calls.

The transport-neutral deadline core lives in ``vrl.utils.deadline``; this
module adds the Ray flavor (error naming, ref cancellation, ``ray.get``
barriers) on top of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.ray.dependencies import require_ray
from vrl.runtime_errors import TerminalRuntimeError
from vrl.utils.deadline import OperationDeadline, OperationTimeout


class RayOperationTimeout(OperationTimeout):
    """A driver-side Ray call exceeded its configured wall-clock budget."""

    transport_label: ClassVar[str] = "Ray operation"


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
class RayCallDeadline(OperationDeadline):
    """The shared deadline core raising the Ray-flavored terminal timeout."""

    timeout_error_type: ClassVar[type[OperationTimeout]] = RayOperationTimeout


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
]
