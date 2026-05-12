"""Builders for derived trajectory views."""

from __future__ import annotations

from vrl.engine.trajectory.types import TrajectoryBatch, TrajectorySegment, TrajectoryTensor
from vrl.engine.trajectory.validation import (
    replay_input_ref,
    tensor_ref,
    validate_training_view,
)
from vrl.engine.trajectory.views import LossUnit, TrainingView


def build_training_view(
    trajectory: TrajectoryBatch,
    *,
    primary_segment: str | None = None,
) -> TrainingView:
    """Build a default policy-gradient TrainingView for trainable segments."""

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
    return validate_training_view(trajectory, view)


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


__all__ = ["build_training_view"]
