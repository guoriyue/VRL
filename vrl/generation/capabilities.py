"""Behavioral capability contract carried from the family registry to workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

TrajectoryKind = Literal[
    "diffusion",
    "ar_discrete",
    "ar_continuous",
    "multisegment",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class FamilyCapability:
    """Runtime decisions supported by one family/task pair.

    Family/task identify the launch contract; trajectory_kind selects the execution path.
    """

    family: str
    task: str
    trajectory_kind: TrajectoryKind

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("FamilyCapability.family must be non-empty")
        if not self.task:
            raise ValueError("FamilyCapability.task must be non-empty")
        _trajectory_kind(self.trajectory_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task": self.task,
            "trajectory_kind": self.trajectory_kind,
        }

    @classmethod
    def from_value(
        cls,
        value: FamilyCapability | Mapping[str, Any],
    ) -> FamilyCapability:
        if isinstance(value, cls):
            return value
        return cls(
            family=str(value["family"]),
            task=str(value["task"]),
            trajectory_kind=_trajectory_kind(value.get("trajectory_kind", "unknown")),
        )


def family_capability_from_value(value: Any) -> FamilyCapability | None:
    """Normalize a serialized or typed capability value."""

    if value is None:
        return None
    if isinstance(value, FamilyCapability):
        return value
    if isinstance(value, Mapping):
        return FamilyCapability.from_value(value)
    raise TypeError(
        "family capability must be a FamilyCapability, mapping, or None; "
        f"got {type(value).__name__}",
    )


def _trajectory_kind(value: Any) -> TrajectoryKind:
    text = str(value)
    if text in {"diffusion", "ar_discrete", "ar_continuous", "multisegment", "unknown"}:
        return text  # type: ignore[return-value]
    raise ValueError(f"unsupported trajectory_kind: {value!r}")


__all__ = [
    "FamilyCapability",
    "TrajectoryKind",
    "family_capability_from_value",
]
