"""Transport-neutral monotonic operation deadlines.

One wall-clock budget primitive shared by both inference engines: Ray-owned
generation barriers subclass it in ``vrl.ray.operation_deadline`` and the
in-process reward runtime consumes it directly, so "how long may this block"
has one implementation and one terminal error family.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import ClassVar

from vrl.runtime_errors import TerminalRuntimeError


class OperationTimeout(TerminalRuntimeError):
    """A driver-side call exceeded its configured wall-clock budget."""

    # Subclasses name their transport so messages keep their historical shape.
    transport_label: ClassVar[str] = "operation"

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
        message = f"{self.transport_label} {operation!r} exceeded {self.timeout_s:g}s"
        if context:
            message = f"{message} ({context})"
        super().__init__(message)


def require_timeout(
    timeout_s: float,
    *,
    name: str = "timeout_s",
) -> float:
    """Validate one public or direct-constructor timeout value."""

    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return timeout_s


@dataclass(frozen=True, slots=True)
class OperationDeadline:
    """One absolute deadline starting when this object is constructed.

    Callers that own submission construct it immediately before submitting when
    submission time belongs inside the operation budget. Helpers receiving
    existing work can only bound the waits they own.
    """

    timeout_error_type: ClassVar[type[OperationTimeout]] = OperationTimeout

    # Protocol identity consumed by the raised terminal error.
    operation: str
    # Behavior-consumed by monotonic expiry and every blocking wait.
    timeout_s: float
    # Provenance-only: identifies the worker/request boundary in diagnostics.
    context: str | None = None
    _expires_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timeout_s = require_timeout(self.timeout_s)
        if not self.operation:
            raise ValueError("deadline operation must not be empty")
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "_expires_at", time.monotonic() + timeout_s)

    @property
    def expires_at(self) -> float:
        """Return the monotonic expiry used to compare concurrent calls."""

        return self._expires_at

    def remaining_s(self) -> float:
        """Return the non-negative budget left for a blocking wait."""

        return max(0.0, self._expires_at - time.monotonic())

    def timeout_error(self) -> OperationTimeout:
        """Build the stable protocol error for this call."""

        return type(self).timeout_error_type(
            self.operation,
            self.timeout_s,
            context=self.context,
        )


__all__ = [
    "OperationDeadline",
    "OperationTimeout",
    "require_timeout",
]
