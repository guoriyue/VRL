"""Trainer replay and runtime model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Options for replaying a recorded trajectory."""

    segment_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.segment_names is None:
            return
        for name in self.segment_names:
            if not isinstance(name, str) or not name:
                raise ValueError("ReplayRequest.segment_names must contain non-empty strings")


@dataclass(slots=True)
class ReplaySegmentResult:
    """Current replay result for one trajectory segment."""

    segment: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError("ReplaySegmentResult.segment must be a non-empty string")
        if not isinstance(self.values, dict):
            raise TypeError("ReplaySegmentResult.values must be a dict")


@dataclass(slots=True)
class ReplayResult:
    """Current replay result for a rollout batch."""

    segments: dict[str, ReplaySegmentResult]
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("ReplayResult.segments must be non-empty")
        for key, segment in self.segments.items():
            if not isinstance(segment, ReplaySegmentResult):
                raise TypeError(f"ReplayResult segment {key!r} must be a ReplaySegmentResult")
            if key != segment.segment:
                raise ValueError(
                    f"ReplayResult segment key {key!r} must match "
                    f"ReplaySegmentResult.segment={segment.segment!r}",
                )


@runtime_checkable
class ReplayModel(Protocol):
    """Minimal model contract consumed by replay evaluators."""

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Replay recorded trajectory actions under the current model."""
        ...

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Temporarily disable adapters, or return a no-op context."""
        ...


@runtime_checkable
class RuntimeModel(ReplayModel, Protocol):
    """Minimum model contract shared by trainer runtime and Ray sync."""

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load the flattened trainable-state payload pushed to rollout workers."""
        ...


def require_replay_model(value: Any, *, owner: str = "model") -> ReplayModel:
    """Return ``value`` as a ReplayModel or fail at the replay boundary."""

    if isinstance(value, ReplayModel):
        return cast(ReplayModel, value)
    missing = _missing_callables(value, ("replay_forward", "disable_adapter"))
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(f"{owner} must satisfy ReplayModel(replay_forward, disable_adapter){detail}")


def require_runtime_model(value: Any, *, owner: str = "model") -> RuntimeModel:
    """Return ``value`` as a RuntimeModel or fail at the runtime boundary."""

    if isinstance(value, RuntimeModel):
        return cast(RuntimeModel, value)
    missing = _missing_callables(
        value,
        ("replay_forward", "disable_adapter", "load_trainable_state"),
    )
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(
        f"{owner} must satisfy RuntimeModel(replay_forward, disable_adapter, "
        f"load_trainable_state){detail}",
    )


def _missing_callables(value: Any, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not callable(getattr(value, name, None))]


__all__ = [
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeModel",
    "require_replay_model",
    "require_runtime_model",
]
