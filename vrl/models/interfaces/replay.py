"""Trainer replay and runtime model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class ReplayModel(Protocol):
    """Minimal model contract consumed by replay evaluators."""

    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        """Replay recorded trajectory actions under the current model.

        Returns the replay values by name: ``noise_pred`` (and its CFG branches)
        for a single-step replay, ``log_probs`` for a grouped one.
        """
        ...

    def reference_policy(self) -> AbstractContextManager[None]:
        """Run the forward with the reference (pre-training) weights."""
        ...


@runtime_checkable
class RuntimeModel(ReplayModel, Protocol):
    """Minimum model contract shared by trainer runtime and Ray sync."""

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load the flattened trainable-state payload pushed to rollout workers."""
        ...


def _require_protocol(value: Any, proto: Any, *, owner: str) -> Any:
    """isinstance-or-raise shared by the boundary casts below; the method list
    derives from the Protocol (same pattern as the contract tests), so a method
    add/rename auto-widens the runtime checks and error messages."""

    methods = tuple(sorted(proto.__protocol_attrs__))
    missing = [name for name in methods if not callable(getattr(value, name, None))]
    if not missing and isinstance(value, proto):
        return value
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(f"{owner} must satisfy {proto.__name__}({', '.join(methods)}){detail}")


def require_replay_model(value: Any, *, owner: str = "model") -> ReplayModel:
    """Return ``value`` as a ReplayModel or fail at the replay boundary."""

    return cast("ReplayModel", _require_protocol(value, ReplayModel, owner=owner))


def require_runtime_model(value: Any, *, owner: str = "model") -> RuntimeModel:
    """Return ``value`` as a RuntimeModel or fail at the runtime boundary."""

    return cast("RuntimeModel", _require_protocol(value, RuntimeModel, owner=owner))


__all__ = [
    "ReplayModel",
    "RuntimeModel",
    "require_replay_model",
    "require_runtime_model",
]
