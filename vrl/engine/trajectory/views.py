"""Derived views over trajectory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.engine.trajectory.types import (
    AdvantageScope,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)

RewardModality = Literal["image", "video", "text", "mixed"]
AlgorithmFamily = Literal["policy_gradient", "supervised", "preference", "custom"]


@dataclass(frozen=True, slots=True)
class RewardView:
    """Names the trajectory facts a reward function should read."""

    name: str
    modality: RewardModality
    tensor_refs: tuple[str, ...] = ()
    prompt_refs: tuple[str, ...] = ()
    target_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RewardView.name must be non-empty")
        _require_string_tuple("RewardView.tensor_refs", self.tensor_refs)
        _require_string_tuple("RewardView.prompt_refs", self.prompt_refs)
        _require_string_tuple("RewardView.target_refs", self.target_refs)


@dataclass(frozen=True, slots=True)
class LossUnit:
    """One logical replay/loss unit inside a training view."""

    segment: str
    axis: str
    axis_index: int | None
    action_ref: str
    old_log_prob_ref: str
    mask_ref: str
    advantage_scope: AdvantageScope = "sample"
    signal_requirements: tuple[str, ...] = ()
    replay_input_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

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
        if self.axis_index is not None and self.axis_index < 0:
            raise ValueError("LossUnit.axis_index must be >= 0 when set")
        _require_string_tuple("LossUnit.signal_requirements", self.signal_requirements)
        _require_string_tuple("LossUnit.replay_input_refs", self.replay_input_refs)


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Names the loss units a trainer or algorithm should iterate."""

    loss_units: tuple[LossUnit, ...]
    primary_segment: str | None = None
    algorithm_family: AlgorithmFamily = "policy_gradient"
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

    from vrl.engine.trajectory.validation import (
        TrajectoryValidator,
        replay_input_ref,
        tensor_ref,
    )

    loss_units: list[LossUnit] = []
    for segment in trajectory.segments.values():
        if not segment.trainable:
            continue
        action = _role_tensor(segment, "action")
        old_log_prob = _role_tensor(segment, "old_log_prob")
        mask = _role_tensor(segment, "mask")
        loss_axis = _loss_axis(action.axes)
        loss_units.append(
            LossUnit(
                segment=segment.name,
                axis=loss_axis,
                axis_index=None,
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


def _require_string_tuple(name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain non-empty strings")


def _role_tensor(segment: TrajectorySegment, role: str) -> TrajectoryTensor:
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
    "AlgorithmFamily",
    "LossUnit",
    "RewardModality",
    "RewardView",
    "TrainingView",
    "build_training_view",
]
