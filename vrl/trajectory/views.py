"""Derived views over trajectory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.trajectory.types import (
    AdvantageScope,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.utils.validation import require_string_tuple

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


@dataclass(frozen=True, slots=True)
class LossUnit:
    """One logical replay/loss unit inside a training view."""

    segment: str
    axis: str
    action_ref: str
    old_log_prob_ref: str
    mask_ref: str
    advantage_scope: AdvantageScope = "sample"
    replay_input_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("LossUnit.segment", self.segment),
            ("LossUnit.axis", self.axis),
            ("LossUnit.action_ref", self.action_ref),
            ("LossUnit.old_log_prob_ref", self.old_log_prob_ref),
            ("LossUnit.mask_ref", self.mask_ref),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        require_string_tuple("LossUnit.replay_input_refs", self.replay_input_refs)


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Names the loss units a trainer or algorithm should iterate."""

    loss_units: tuple[LossUnit, ...]
    primary_segment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_segment is not None and not self.primary_segment:
            raise ValueError("TrainingView.primary_segment must be non-empty when set")


def build_training_view(
    trajectory: TrajectoryBatch,
    *,
    primary_segment: str | None = None,
) -> TrainingView:
    """Build a default policy-gradient TrainingView for trainable segments."""

    from vrl.trajectory.validation import (
        TrajectoryValidator,
        replay_input_ref,
        tensor_ref,
    )

    loss_units: list[LossUnit] = []
    for segment in trajectory.segments.values():
        if not segment.trainable:
            continue
        action = role_tensor(segment, "action")
        old_log_prob = role_tensor(segment, "old_log_prob")
        mask = role_tensor(segment, "mask")
        loss_axis = _loss_axis(action.axes)
        loss_units.append(
            LossUnit(
                segment=segment.name,
                axis=loss_axis,
                action_ref=tensor_ref(segment.name, action.name),
                old_log_prob_ref=tensor_ref(segment.name, old_log_prob.name),
                mask_ref=tensor_ref(segment.name, mask.name),
                advantage_scope=segment.advantage_scope,
                replay_input_refs=tuple(
                    replay_input_ref(segment.name, name)
                    for name in segment.replay_inputs
                ),
            )
        )

    view = TrainingView(
        loss_units=tuple(loss_units),
        primary_segment=primary_segment or (loss_units[0].segment if loss_units else None),
    )
    return TrajectoryValidator(trajectory).validate_training_view(view)


def role_tensor(segment: TrajectorySegment, role: str) -> TrajectoryTensor:
    matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"segment {segment.name!r} requires exactly one role {role!r}, "
            f"found {len(matches)}",
        )
    return matches[0]


def _loss_axis(axes: tuple[str, ...]) -> str:
    for axis in axes:
        if axis != "sample":
            return axis
    return axes[-1]


__all__ = [
    "LossUnit",
    "RewardValueRange",
    "RewardView",
    "TrainingView",
    "build_training_view",
    "role_tensor",
]
