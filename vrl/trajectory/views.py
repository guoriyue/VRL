"""Derived views over trajectory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.trajectory.types import TrajectorySegment, TrajectoryTensor, require_string_tuple

RewardValueRange = Literal["unit", "tanh"]


@dataclass(frozen=True, slots=True)
class RewardView:
    """Names the trajectory facts a reward function should read."""

    name: str
    tensor_refs: tuple[str, ...] = ()
    # Declared pixel value range of the reward media. The collector normalizes
    # to "unit" [0, 1] for scoring; "tanh" [-1, 1] sources (VQ decode) get
    # rescaled. This is a model fact owned by the producer, not an operator knob.
    value_range: RewardValueRange = "unit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RewardView.name must be non-empty")
        if self.value_range not in {"unit", "tanh"}:
            raise ValueError(
                f"RewardView.value_range must be 'unit' or 'tanh', got {self.value_range!r}",
            )
        require_string_tuple("RewardView.tensor_refs", self.tensor_refs)


def role_tensor(segment: TrajectorySegment, role: str) -> TrajectoryTensor:
    matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"segment {segment.name!r} requires exactly one role {role!r}, found {len(matches)}",
        )
    return matches[0]


def named_tensor(segment: TrajectorySegment, name: str) -> TrajectoryTensor:
    """Read one named tensor from a segment or fail with the missing name."""

    try:
        return segment.tensors[name]
    except KeyError as exc:
        raise RuntimeError(
            f"segment {segment.name!r} is missing tensor {name!r}",
        ) from exc


__all__ = [
    "RewardValueRange",
    "RewardView",
    "named_tensor",
    "role_tensor",
]
