"""Validation helpers for trajectory contracts."""

from __future__ import annotations

import io
import types
from collections.abc import Mapping
from typing import Any

from vrl.engine.trajectory.types import (
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.engine.trajectory.views import LossUnit, RewardView, TrainingView

FORBIDDEN_TRAJECTORY_METRICS = frozenset(
    {
        "queue_wait_s",
        "execution_s",
        "peak_memory_mb",
        "micro_batches",
    }
)
SINGLETON_TENSOR_ROLES = frozenset({"action", "old_log_prob", "mask"})


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory record violates the contract."""


def validate_trajectory_batch(batch: TrajectoryBatch) -> TrajectoryBatch:
    """Validate structural trajectory invariants and return ``batch``."""

    if not batch.sample_specs:
        _fail("TrajectoryBatch.sample_specs must be non-empty")
    if not batch.axes:
        _fail("TrajectoryBatch.axes must be non-empty")
    if "sample" not in batch.axes:
        _fail("TrajectoryBatch.axes must include a 'sample' axis")
    if not batch.segments:
        _fail("TrajectoryBatch.segments must be non-empty")

    for axis_name, axis in batch.axes.items():
        if axis_name != axis.name:
            _fail(f"Axis key {axis_name!r} does not match AxisSpec.name={axis.name!r}")
        if axis.length is not None and axis.length < 0:
            _fail(f"Axis {axis_name!r} length must be >= 0")

    sample_axis = batch.axes["sample"]
    if sample_axis.length is not None and sample_axis.length != len(batch.sample_specs):
        _fail(
            "sample axis length does not match sample_specs: "
            f"{sample_axis.length} != {len(batch.sample_specs)}",
        )
    group_length = _leading_length(batch.group_ids)
    if group_length is None:
        _fail("TrajectoryBatch.group_ids must be a sample-aligned sequence")
    if group_length != len(batch.sample_specs):
        _fail(
            "group_ids length does not match sample_specs: "
            f"{group_length} != {len(batch.sample_specs)}",
        )
    if batch.metrics.num_samples is not None and batch.metrics.num_samples != len(
        batch.sample_specs,
    ):
        _fail(
            "TrajectoryMetrics.num_samples does not match sample_specs: "
            f"{batch.metrics.num_samples} != {len(batch.sample_specs)}",
        )
    for axis_name, length in batch.metrics.axis_lengths.items():
        axis = batch.axes.get(axis_name)
        if axis is None:
            _fail(f"TrajectoryMetrics.axis_lengths references unknown axis {axis_name!r}")
        if axis.length is not None and int(length) != axis.length:
            _fail(
                f"TrajectoryMetrics.axis_lengths[{axis_name!r}]={length} "
                f"does not match AxisSpec.length={axis.length}",
            )
    forbidden_metrics = FORBIDDEN_TRAJECTORY_METRICS.intersection(batch.metrics.values)
    if forbidden_metrics:
        _fail(
            "TrajectoryMetrics.values contains engine runtime metrics: "
            + ", ".join(sorted(forbidden_metrics)),
        )

    tensor_refs: dict[str, TrajectoryTensor] = {}
    for segment_key, segment in batch.segments.items():
        _validate_segment(batch, segment_key, segment, tensor_refs)

    for view_key, view in batch.reward_views.items():
        if not isinstance(view, RewardView):
            _fail(f"reward_views[{view_key!r}] must be a RewardView")
        if view_key != view.name:
            _fail(f"RewardView key {view_key!r} does not match name={view.name!r}")
        validate_reward_view(batch, view, tensor_refs=tensor_refs)

    _reject_runtime_state(batch.context, "TrajectoryBatch.context")
    _reject_runtime_state(batch.metrics.values, "TrajectoryBatch.metrics.values")
    return batch


def validate_reward_view(
    batch: TrajectoryBatch,
    view: RewardView,
    *,
    tensor_refs: dict[str, TrajectoryTensor] | None = None,
) -> RewardView:
    """Validate that a reward view only references known trajectory tensors."""

    refs = tensor_refs or _collect_tensor_refs(batch)
    for ref in view.tensor_refs:
        if ref not in refs:
            _fail(f"RewardView {view.name!r} references unknown tensor {ref!r}")
    _reject_runtime_state(view.metadata, f"RewardView {view.name!r}.metadata")
    return view


def validate_training_view(batch: TrajectoryBatch, view: TrainingView) -> TrainingView:
    """Validate that a training view references existing loss tensors."""

    refs = _collect_tensor_refs(batch)
    if not view.loss_units:
        _fail("TrainingView.loss_units must be non-empty")
    if view.primary_segment is not None and view.primary_segment not in batch.segments:
        _fail(f"TrainingView.primary_segment={view.primary_segment!r} is unknown")

    for unit in view.loss_units:
        validate_loss_unit(batch, unit, tensor_refs=refs)

    _reject_runtime_state(view.metadata, "TrainingView.metadata")
    return view


def validate_loss_unit(
    batch: TrajectoryBatch,
    unit: LossUnit,
    *,
    tensor_refs: dict[str, TrajectoryTensor] | None = None,
) -> LossUnit:
    """Validate one loss unit against trajectory tensor refs and axes."""

    refs = tensor_refs or _collect_tensor_refs(batch)
    if unit.segment not in batch.segments:
        _fail(f"LossUnit.segment={unit.segment!r} is unknown")
    if unit.axis not in batch.axes:
        _fail(f"LossUnit.axis={unit.axis!r} is unknown")
    axis = batch.axes[unit.axis]
    if unit.axis_index is not None and axis.length is not None and unit.axis_index >= axis.length:
        _fail(
            f"LossUnit.axis_index={unit.axis_index} is out of range for "
            f"axis {unit.axis!r} length={axis.length}",
        )
    for field_name, ref in (
        ("action_ref", unit.action_ref),
        ("old_log_prob_ref", unit.old_log_prob_ref),
        ("mask_ref", unit.mask_ref),
    ):
        if ref not in refs:
            _fail(f"LossUnit.{field_name} references unknown tensor {ref!r}")
        ref_segment = ref.split(".", 1)[0]
        if ref_segment != unit.segment:
            _fail(
                f"LossUnit.{field_name} must reference segment {unit.segment!r}, "
                f"got {ref_segment!r}",
            )
        if unit.axis not in refs[ref].axes:
            _fail(
                f"LossUnit.{field_name} must reference tensor with axis "
                f"{unit.axis!r}",
            )
    _require_ref_role(refs, unit.action_ref, "action", "LossUnit.action_ref")
    _require_ref_role(
        refs,
        unit.old_log_prob_ref,
        "old_log_prob",
        "LossUnit.old_log_prob_ref",
    )
    _require_ref_role(refs, unit.mask_ref, "mask", "LossUnit.mask_ref")
    for replay_ref in unit.replay_input_refs:
        if "." not in replay_ref:
            _fail(f"LossUnit replay input ref {replay_ref!r} must be 'segment.name'")
        segment_name, replay_name = replay_ref.split(".", 1)
        segment = batch.segments.get(segment_name)
        if segment is None or replay_name not in segment.replay_inputs:
            _fail(f"LossUnit references unknown replay input {replay_ref!r}")
    return unit


def tensor_ref(segment: str, tensor: str) -> str:
    """Return the canonical string ref for a segment tensor."""

    if not segment or not tensor:
        raise ValueError("segment and tensor must be non-empty")
    return f"{segment}.{tensor}"


def replay_input_ref(segment: str, replay_input: str) -> str:
    """Return the canonical string ref for a segment replay input."""

    if not segment or not replay_input:
        raise ValueError("segment and replay_input must be non-empty")
    return f"{segment}.{replay_input}"


def _validate_segment(
    batch: TrajectoryBatch,
    segment_key: str,
    segment: TrajectorySegment,
    tensor_refs: dict[str, TrajectoryTensor],
) -> None:
    if segment_key != segment.name:
        _fail(
            f"Segment key {segment_key!r} does not match "
            f"TrajectorySegment.name={segment.name!r}",
        )
    if segment.reward_view is not None and segment.reward_view not in batch.reward_views:
        _fail(
            f"TrajectorySegment {segment.name!r} references unknown reward_view "
            f"{segment.reward_view!r}",
        )
    if not segment.tensors:
        _fail(f"TrajectorySegment {segment.name!r} must contain tensors")

    roles: dict[str, TrajectoryTensor] = {}
    for tensor_key, tensor in segment.tensors.items():
        if tensor_key != tensor.name:
            _fail(
                f"Tensor key {tensor_key!r} in segment {segment.name!r} does not "
                f"match TrajectoryTensor.name={tensor.name!r}",
            )
        _validate_tensor_axes(batch, segment.name, tensor)
        ref = tensor_ref(segment.name, tensor.name)
        if ref in tensor_refs:
            _fail(f"duplicate trajectory tensor ref {ref!r}")
        tensor_refs[ref] = tensor
        if tensor.role in SINGLETON_TENSOR_ROLES and tensor.role in roles:
            _fail(
                f"trainable segment {segment.name!r} has multiple "
                f"{tensor.role!r} tensors",
            )
        roles.setdefault(tensor.role, tensor)

    if segment.trainable:
        missing = [role for role in ("action", "old_log_prob", "mask") if role not in roles]
        if missing:
            _fail(
                f"trainable segment {segment.name!r} is missing required roles: "
                + ", ".join(missing),
            )
        old_log_prob = roles["old_log_prob"]
        mask = roles["mask"]
        if old_log_prob.axes != mask.axes:
            _fail(
                f"trainable segment {segment.name!r} old_log_prob axes "
                f"{old_log_prob.axes!r} must match mask axes {mask.axes!r}",
            )
        action = roles["action"]
        if action.axes != old_log_prob.axes:
            _fail(
                f"trainable segment {segment.name!r} action axes "
                f"{action.axes!r} must match old_log_prob axes {old_log_prob.axes!r}",
            )

    for replay_key, replay in segment.replay_inputs.items():
        if replay_key != replay.name:
            _fail(
                f"ReplayInput key {replay_key!r} in segment {segment.name!r} "
                f"does not match name={replay.name!r}",
            )
        for ref in replay.tensor_refs:
            if "." not in ref:
                ref = tensor_ref(segment.name, ref)
            if ref not in tensor_refs:
                _fail(f"ReplayInput {replay.name!r} references unknown tensor {ref!r}")
        _reject_runtime_state(replay.metadata, f"ReplayInput {segment.name}.{replay.name}.metadata")

    _reject_runtime_state(segment.metadata, f"TrajectorySegment {segment.name!r}.metadata")


def _validate_tensor_axes(
    batch: TrajectoryBatch,
    segment_name: str,
    tensor: TrajectoryTensor,
) -> None:
    _reject_runtime_state(
        tensor.value,
        f"TrajectoryTensor {segment_name}.{tensor.name}.value",
        allow_tensor_like=True,
    )
    _reject_runtime_state(tensor.metadata, f"TrajectoryTensor {segment_name}.{tensor.name}.metadata")

    seen: set[str] = set()
    for axis_name in tensor.axes:
        if axis_name not in batch.axes:
            _fail(
                f"tensor {segment_name}.{tensor.name} references unknown "
                f"axis {axis_name!r}",
            )
        if axis_name in seen:
            _fail(f"tensor {segment_name}.{tensor.name} repeats axis {axis_name!r}")
        seen.add(axis_name)

    shape = getattr(tensor.value, "shape", None)
    if shape is None:
        return
    if len(shape) < len(tensor.axes):
        _fail(
            f"tensor {segment_name}.{tensor.name} rank {len(shape)} is smaller than "
            f"declared axes {tensor.axes!r}",
        )
    for dim, axis_name in enumerate(tensor.axes):
        expected = batch.axes[axis_name].length
        if expected is None:
            continue
        actual = int(shape[dim])
        if actual != expected:
            _fail(
                f"tensor {segment_name}.{tensor.name} axis {axis_name!r} has "
                f"shape {actual}, expected {expected}",
            )


def _collect_tensor_refs(batch: TrajectoryBatch) -> dict[str, TrajectoryTensor]:
    refs: dict[str, TrajectoryTensor] = {}
    for segment in batch.segments.values():
        for tensor in segment.tensors.values():
            refs[tensor_ref(segment.name, tensor.name)] = tensor
    return refs


def _require_ref_role(
    refs: dict[str, TrajectoryTensor],
    ref: str,
    role: str,
    field_name: str,
) -> None:
    actual = refs[ref].role
    if actual != role:
        _fail(f"{field_name} must reference role {role!r}, got {actual!r}")


def _leading_length(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return len(value)
    except TypeError:
        return None


def _reject_runtime_state(value: Any, path: str, *, allow_tensor_like: bool = False) -> None:
    if _looks_runtime_only(value, allow_tensor_like=allow_tensor_like):
        _fail(f"{path} contains runtime-only state: {type(value).__name__}")
    if isinstance(value, Mapping):
        for key, inner in value.items():
            _reject_runtime_state(inner, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            _reject_runtime_state(inner, f"{path}[{index}]")


def _looks_runtime_only(value: Any, *, allow_tensor_like: bool = False) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, int, float, bool)):
        return False
    if isinstance(value, (dict, list, tuple)):
        return False
    if getattr(value, "shape", None) is not None:
        return not allow_tensor_like
    if isinstance(value, (types.ModuleType, io.IOBase)):
        return True
    if callable(value):
        return True
    has_model_api = callable(getattr(value, "state_dict", None)) and callable(
        getattr(value, "parameters", None),
    )
    if has_model_api:
        return True
    type_module = type(value).__module__
    type_name = type(value).__name__.lower()
    if type_module.startswith("ray.") or "actor" in type_name:
        return True
    return True


def _fail(message: str) -> None:
    raise TrajectoryValidationError(message)


__all__ = [
    "TrajectoryValidationError",
    "replay_input_ref",
    "tensor_ref",
    "validate_loss_unit",
    "validate_reward_view",
    "validate_training_view",
    "validate_trajectory_batch",
]
