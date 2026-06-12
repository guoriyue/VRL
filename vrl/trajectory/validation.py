"""Validation helpers for trajectory contracts."""

from __future__ import annotations

import io
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.trajectory.types import (
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.views import LossUnit, RewardView, TrainingView

FORBIDDEN_TRAJECTORY_METRICS = frozenset(
    {
        "queue_wait_s",
        "execution_s",
        "peak_memory_mb",
        "chunks",
    }
)
# The core role triple a trainable segment must have exactly one of each.
# Single ordered source of truth: the uniqueness check (below) and the
# completeness check (TrajectoryValidator) both derive from it.
REQUIRED_TRAINABLE_ROLES = ("action", "old_log_prob", "mask")
SINGLETON_TENSOR_ROLES = frozenset(REQUIRED_TRAINABLE_ROLES)


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory record violates the contract."""


@dataclass(slots=True)
class TrajectoryValidator:
    """Validate one trajectory batch and its derived training/reward views."""

    batch: TrajectoryBatch
    tensor_refs: dict[str, TrajectoryTensor] = field(default_factory=dict)

    def validate_batch(self) -> TrajectoryBatch:
        """Validate structural trajectory invariants and return the batch."""

        batch = self.batch
        if not batch.sample_rows:
            self._fail("TrajectoryBatch.sample_rows must be non-empty")
        if not batch.axes:
            self._fail("TrajectoryBatch.axes must be non-empty")
        if "sample" not in batch.axes:
            self._fail("TrajectoryBatch.axes must include a 'sample' axis")
        if not batch.segments:
            self._fail("TrajectoryBatch.segments must be non-empty")

        for axis_name, axis in batch.axes.items():
            if axis_name != axis.name:
                self._fail(
                    f"Axis key {axis_name!r} does not match "
                    f"TrajectoryAxis.name={axis.name!r}",
                )
            if axis.length is not None and axis.length < 0:
                self._fail(f"Axis {axis_name!r} length must be >= 0")

        sample_axis = batch.axes["sample"]
        if sample_axis.length is not None and sample_axis.length != len(batch.sample_rows):
            self._fail(
                "sample axis length does not match sample_rows: "
                f"{sample_axis.length} != {len(batch.sample_rows)}",
            )
        group_length = self._leading_length(batch.group_ids)
        if group_length is None:
            self._fail("TrajectoryBatch.group_ids must be a sample-aligned sequence")
        if group_length != len(batch.sample_rows):
            self._fail(
                "group_ids length does not match sample_rows: "
                f"{group_length} != {len(batch.sample_rows)}",
            )
        if batch.metrics.num_samples is not None and batch.metrics.num_samples != len(
            batch.sample_rows,
        ):
            self._fail(
                "TrajectoryMetrics.num_samples does not match sample_rows: "
                f"{batch.metrics.num_samples} != {len(batch.sample_rows)}",
            )
        for axis_name, length in batch.metrics.axis_lengths.items():
            axis = batch.axes.get(axis_name)
            if axis is None:
                self._fail(
                    f"TrajectoryMetrics.axis_lengths references unknown axis {axis_name!r}"
                )
            if axis.length is not None and int(length) != axis.length:
                self._fail(
                    f"TrajectoryMetrics.axis_lengths[{axis_name!r}]={length} "
                    f"does not match TrajectoryAxis.length={axis.length}",
                )
        forbidden_metrics = FORBIDDEN_TRAJECTORY_METRICS.intersection(batch.metrics.values)
        if forbidden_metrics:
            self._fail(
                "TrajectoryMetrics.values contains engine runtime metrics: "
                + ", ".join(sorted(forbidden_metrics)),
            )

        self.tensor_refs.clear()
        for segment_key, segment in batch.segments.items():
            self._validate_segment(segment_key, segment)

        for view_key, view in batch.reward_views.items():
            if not isinstance(view, RewardView):
                self._fail(f"reward_views[{view_key!r}] must be a RewardView")
            if view_key != view.name:
                self._fail(f"RewardView key {view_key!r} does not match name={view.name!r}")
            self.validate_reward_view(view)

        self._reject_runtime_state(batch.context, "TrajectoryBatch.context")
        self._reject_runtime_state(batch.metrics.values, "TrajectoryBatch.metrics.values")
        return batch

    def validate_reward_view(self, view: RewardView) -> RewardView:
        """Validate that a reward view only references known trajectory tensors."""

        refs = self._ensure_tensor_refs()
        for ref in view.tensor_refs:
            if ref not in refs:
                self._fail(f"RewardView {view.name!r} references unknown tensor {ref!r}")
        self._reject_runtime_state(view.metadata, f"RewardView {view.name!r}.metadata")
        return view

    def validate_training_view(self, view: TrainingView) -> TrainingView:
        """Validate that a training view references existing loss tensors."""

        if not view.loss_units:
            self._fail("TrainingView.loss_units must be non-empty")
        if view.primary_segment is not None and view.primary_segment not in self.batch.segments:
            self._fail(f"TrainingView.primary_segment={view.primary_segment!r} is unknown")

        for unit in view.loss_units:
            self.validate_loss_unit(unit)

        self._reject_runtime_state(view.metadata, "TrainingView.metadata")
        return view

    def validate_loss_unit(self, unit: LossUnit) -> LossUnit:
        """Validate one loss unit against trajectory tensor refs and axes."""

        refs = self._ensure_tensor_refs()
        if unit.segment not in self.batch.segments:
            self._fail(f"LossUnit.segment={unit.segment!r} is unknown")
        if unit.axis not in self.batch.axes:
            self._fail(f"LossUnit.axis={unit.axis!r} is unknown")
        for field_name, ref in (
            ("action_ref", unit.action_ref),
            ("old_log_prob_ref", unit.old_log_prob_ref),
            ("mask_ref", unit.mask_ref),
        ):
            if ref not in refs:
                self._fail(f"LossUnit.{field_name} references unknown tensor {ref!r}")
            ref_segment = ref.split(".", 1)[0]
            if ref_segment != unit.segment:
                self._fail(
                    f"LossUnit.{field_name} must reference segment {unit.segment!r}, "
                    f"got {ref_segment!r}",
                )
            if unit.axis not in refs[ref].axes:
                self._fail(
                    f"LossUnit.{field_name} must reference tensor with axis "
                    f"{unit.axis!r}",
                )
        self._require_ref_role(unit.action_ref, "action", "LossUnit.action_ref")
        self._require_ref_role(
            unit.old_log_prob_ref,
            "old_log_prob",
            "LossUnit.old_log_prob_ref",
        )
        self._require_ref_role(unit.mask_ref, "mask", "LossUnit.mask_ref")
        for replay_ref in unit.replay_input_refs:
            if "." not in replay_ref:
                self._fail(
                    f"LossUnit replay input ref {replay_ref!r} must be 'segment.name'"
                )
            segment_name, replay_name = replay_ref.split(".", 1)
            segment = self.batch.segments.get(segment_name)
            if segment is None or replay_name not in segment.replay_inputs:
                self._fail(f"LossUnit references unknown replay input {replay_ref!r}")
        return unit

    def _validate_segment(
        self,
        segment_key: str,
        segment: TrajectorySegment,
    ) -> None:
        batch = self.batch
        if segment_key != segment.name:
            self._fail(
                f"Segment key {segment_key!r} does not match "
                f"TrajectorySegment.name={segment.name!r}",
            )
        if segment.reward_view is not None and segment.reward_view not in batch.reward_views:
            self._fail(
                f"TrajectorySegment {segment.name!r} references unknown reward_view "
                f"{segment.reward_view!r}",
            )
        if not segment.tensors:
            self._fail(f"TrajectorySegment {segment.name!r} must contain tensors")

        roles: dict[str, TrajectoryTensor] = {}
        for tensor_key, tensor in segment.tensors.items():
            if tensor_key != tensor.name:
                self._fail(
                    f"Tensor key {tensor_key!r} in segment {segment.name!r} does not "
                    f"match TrajectoryTensor.name={tensor.name!r}",
                )
            self._validate_tensor_axes(segment.name, tensor)
            ref = tensor_ref(segment.name, tensor.name)
            if ref in self.tensor_refs:
                self._fail(f"duplicate trajectory tensor ref {ref!r}")
            self.tensor_refs[ref] = tensor
            if tensor.role in SINGLETON_TENSOR_ROLES and tensor.role in roles:
                self._fail(
                    f"trainable segment {segment.name!r} has multiple "
                    f"{tensor.role!r} tensors",
                )
            roles.setdefault(tensor.role, tensor)

        if segment.trainable:
            missing = [role for role in REQUIRED_TRAINABLE_ROLES if role not in roles]
            if missing:
                self._fail(
                    f"trainable segment {segment.name!r} is missing required roles: "
                    + ", ".join(missing),
                )
            old_log_prob = roles["old_log_prob"]
            mask = roles["mask"]
            if old_log_prob.axes != mask.axes:
                self._fail(
                    f"trainable segment {segment.name!r} old_log_prob axes "
                    f"{old_log_prob.axes!r} must match mask axes {mask.axes!r}",
                )
            action = roles["action"]
            if action.axes != old_log_prob.axes:
                self._fail(
                    f"trainable segment {segment.name!r} action axes "
                    f"{action.axes!r} must match old_log_prob axes {old_log_prob.axes!r}",
                )

        for replay_key, replay in segment.replay_inputs.items():
            if replay_key != replay.name:
                self._fail(
                    f"ReplayInput key {replay_key!r} in segment {segment.name!r} "
                    f"does not match name={replay.name!r}",
                )
            for ref in replay.tensor_refs:
                if "." not in ref:
                    ref = tensor_ref(segment.name, ref)
                if ref not in self.tensor_refs:
                    self._fail(
                        f"ReplayInput {replay.name!r} references unknown tensor {ref!r}"
                    )
            self._reject_runtime_state(
                replay.metadata,
                f"ReplayInput {segment.name}.{replay.name}.metadata",
            )

        self._reject_runtime_state(
            segment.metadata,
            f"TrajectorySegment {segment.name!r}.metadata",
        )

    def _validate_tensor_axes(
        self,
        segment_name: str,
        tensor: TrajectoryTensor,
    ) -> None:
        self._reject_runtime_state(
            tensor.value,
            f"TrajectoryTensor {segment_name}.{tensor.name}.value",
            allow_tensor_like=True,
        )
        self._reject_runtime_state(
            tensor.metadata,
            f"TrajectoryTensor {segment_name}.{tensor.name}.metadata",
        )

        seen: set[str] = set()
        for axis_name in tensor.axes:
            if axis_name not in self.batch.axes:
                self._fail(
                    f"tensor {segment_name}.{tensor.name} references unknown "
                    f"axis {axis_name!r}",
                )
            if axis_name in seen:
                self._fail(f"tensor {segment_name}.{tensor.name} repeats axis {axis_name!r}")
            seen.add(axis_name)

        shape = getattr(tensor.value, "shape", None)
        if shape is None:
            return
        if len(shape) < len(tensor.axes):
            self._fail(
                f"tensor {segment_name}.{tensor.name} rank {len(shape)} is smaller than "
                f"declared axes {tensor.axes!r}",
            )
        for dim, axis_name in enumerate(tensor.axes):
            expected = self.batch.axes[axis_name].length
            if expected is None:
                continue
            actual = int(shape[dim])
            if actual != expected:
                self._fail(
                    f"tensor {segment_name}.{tensor.name} axis {axis_name!r} has "
                    f"shape {actual}, expected {expected}",
                )

    def _ensure_tensor_refs(self) -> dict[str, TrajectoryTensor]:
        if not self.tensor_refs:
            self.tensor_refs.update(self._collect_tensor_refs())
        return self.tensor_refs

    def _collect_tensor_refs(self) -> dict[str, TrajectoryTensor]:
        refs: dict[str, TrajectoryTensor] = {}
        for segment in self.batch.segments.values():
            for tensor in segment.tensors.values():
                refs[tensor_ref(segment.name, tensor.name)] = tensor
        return refs

    def _require_ref_role(self, ref: str, role: str, field_name: str) -> None:
        refs = self._ensure_tensor_refs()
        actual = refs[ref].role
        if actual != role:
            self._fail(f"{field_name} must reference role {role!r}, got {actual!r}")

    @staticmethod
    def _leading_length(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        try:
            return len(value)
        except TypeError:
            return None

    def _reject_runtime_state(
        self,
        value: Any,
        path: str,
        *,
        allow_tensor_like: bool = False,
    ) -> None:
        if self._looks_runtime_only(value, allow_tensor_like=allow_tensor_like):
            self._fail(f"{path} contains runtime-only state: {type(value).__name__}")
        if isinstance(value, Mapping):
            for key, inner in value.items():
                self._reject_runtime_state(inner, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, inner in enumerate(value):
                self._reject_runtime_state(inner, f"{path}[{index}]")

    @staticmethod
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
        # Anything that reached here is neither a serializable scalar/container
        # nor tensor-like, so it's treated as runtime-only.
        return True

    @staticmethod
    def _fail(message: str) -> None:
        raise TrajectoryValidationError(message)


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


__all__ = [
    "TrajectoryValidationError",
    "TrajectoryValidator",
    "replay_input_ref",
    "tensor_ref",
]
