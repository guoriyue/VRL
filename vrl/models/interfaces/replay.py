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
        if isinstance(self.segment_names, (str, bytes)):
            raise ValueError("ReplayRequest.segment_names must be a sequence of names")
        segment_names = tuple(self.segment_names)
        for name in segment_names:
            if not isinstance(name, str) or not name:
                raise ValueError("ReplayRequest.segment_names must contain non-empty strings")
        object.__setattr__(self, "segment_names", segment_names)


@dataclass(slots=True)
class ReplaySegmentResult:
    """Current replay result for one trajectory segment."""

    segment: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError("ReplaySegmentResult.segment must be a non-empty string")

    def require_value(self, key: str) -> Any:
        """Return a named replay payload or fail with the available payload keys."""

        if key not in self.values:
            raise KeyError(
                f"ReplaySegmentResult for segment {self.segment!r} "
                f"missing required key {key!r}; got {sorted(self.values)}",
            )
        return self.values[key]


@dataclass(slots=True)
class ReplayResult:
    """Current replay result for a rollout batch."""

    segments: dict[str, ReplaySegmentResult]

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

    def require_segment(self, segment_name: str) -> ReplaySegmentResult:
        """Return a replay segment result or fail with a helpful segment list."""

        try:
            return self.segments[segment_name]
        except KeyError as err:
            raise KeyError(
                f"ReplayResult missing segment {segment_name!r}; got {sorted(self.segments)}",
            ) from err


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
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeModel",
    "require_replay_model",
    "require_runtime_model",
]
