"""Rollout runtime lifecycle: behavior-consumed phases and expected-kill records.

The phase machine answers the two questions fault handling cannot answer from
call order alone:

- *May this operation run?* ``RUNNING`` admits requests; ``DRAINING`` rejects
  new work while teardown proceeds; ``STOPPED`` fail-fasts every public
  operation; ``FAILED`` re-raises the preserved root cause instead of a vague
  secondary error.
- *Was this actor death ours?* A driver-initiated kill is recorded BEFORE the
  kill is sent (operation id, fleet generation, actors, reason, timestamp), so
  a death observed during ``DRAINING``/teardown is classified expected by
  matching the record — not by guessing from the phase alone.

``RECOVERING`` is part of the protocol now so the transition table is stable,
but nothing enters it yet: automatic fleet recovery lands behind this seam
(sprint step 3+), and its single-flight rule is enforced by the transition
table (``RECOVERING -> RECOVERING`` is invalid).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimePhase(Enum):
    STARTING = "starting"
    RUNNING = "running"
    RECOVERING = "recovering"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


_VALID_TRANSITIONS: dict[RuntimePhase, frozenset[RuntimePhase]] = {
    RuntimePhase.STARTING: frozenset({RuntimePhase.RUNNING, RuntimePhase.FAILED}),
    RuntimePhase.RUNNING: frozenset(
        {RuntimePhase.RECOVERING, RuntimePhase.DRAINING, RuntimePhase.FAILED},
    ),
    RuntimePhase.RECOVERING: frozenset(
        {RuntimePhase.RUNNING, RuntimePhase.DRAINING, RuntimePhase.FAILED},
    ),
    RuntimePhase.DRAINING: frozenset({RuntimePhase.STOPPED, RuntimePhase.FAILED}),
    RuntimePhase.STOPPED: frozenset(),
    RuntimePhase.FAILED: frozenset(),
}


class RuntimeLifecycleError(RuntimeError):
    """A public runtime operation was invoked in a phase that forbids it."""

    def __init__(self, operation: str, phase: RuntimePhase) -> None:
        super().__init__(
            f"{operation} rejected: rollout runtime is {phase.value}",
        )
        self.operation = operation
        self.phase = phase


@dataclass(frozen=True, slots=True)
class ExpectedKillRecord:
    """Provenance of one driver-initiated actor kill.

    Written before the kill is sent so a concurrently observed death can be
    matched against it. ``actor_keys`` are stable identity keys (Ray actor id
    hex when available, else the handle's repr) — a handle object itself is
    not a durable key across restarts.
    """

    operation_id: str
    fleet_generation: int
    actor_keys: tuple[str, ...]
    reason: str
    wall_time: float


def actor_identity_key(actor: Any) -> str:
    """Stable identity key for a Ray actor handle (fake-tolerant)."""

    actor_id = getattr(actor, "_actor_id", None)
    hex_of = getattr(actor_id, "hex", None)
    if callable(hex_of):
        return str(hex_of())
    return repr(actor)


@dataclass
class RuntimeLifecycle:
    """Phase holder + expected-kill registry, owned by the rollout runtime."""

    phase: RuntimePhase = RuntimePhase.STARTING
    failure: BaseException | None = None
    _kills: list[ExpectedKillRecord] = field(default_factory=list)
    _operation_ids: Any = field(
        default_factory=lambda: (f"kill-{n}" for n in itertools.count(1)),
    )

    def transition(self, to: RuntimePhase) -> None:
        if to not in _VALID_TRANSITIONS[self.phase]:
            raise RuntimeLifecycleError(
                f"transition to {to.value}", self.phase,
            )
        self.phase = to

    def fail(self, error: BaseException) -> None:
        """Enter FAILED, preserving the first root cause."""

        if self.phase in (RuntimePhase.STOPPED, RuntimePhase.FAILED):
            return
        self.failure = self.failure or error
        self.phase = RuntimePhase.FAILED

    def require_running(self, operation: str) -> None:
        """Admission gate for public runtime operations."""

        if self.phase is RuntimePhase.FAILED and self.failure is not None:
            raise RuntimeLifecycleError(operation, self.phase) from self.failure
        if self.phase not in (RuntimePhase.STARTING, RuntimePhase.RUNNING):
            raise RuntimeLifecycleError(operation, self.phase)

    def record_expected_kill(
        self,
        actors: list[Any],
        *,
        fleet_generation: int,
        reason: str,
    ) -> ExpectedKillRecord:
        record = ExpectedKillRecord(
            operation_id=next(self._operation_ids),
            fleet_generation=fleet_generation,
            actor_keys=tuple(actor_identity_key(actor) for actor in actors),
            reason=reason,
            wall_time=time.time(),
        )
        self._kills.append(record)
        return record

    def expected_kill_for(self, actor: Any) -> ExpectedKillRecord | None:
        """The kill record covering this actor, or None (death was not ours)."""

        key = actor_identity_key(actor)
        for record in reversed(self._kills):
            if key in record.actor_keys:
                return record
        return None


__all__ = [
    "ExpectedKillRecord",
    "RuntimeLifecycle",
    "RuntimeLifecycleError",
    "RuntimePhase",
    "actor_identity_key",
]
