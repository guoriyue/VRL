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
from vrl.trajectory.views import RewardView

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
    tensor_refs: dict[str, TrajectoryTensor] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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
                    f"Axis key {axis_name!r} does not match TrajectoryAxis.name={axis.name!r}",
                )
            if axis.length is not None and axis.length < 0:
                self._fail(f"Axis {axis_name!r} length must be >= 0")

        sample_axis = batch.axes["sample"]
        if sample_axis.length is not None and sample_axis.length != len(batch.sample_rows):
            self._fail(
                "sample axis length does not match sample_rows: "
                f"{sample_axis.length} != {len(batch.sample_rows)}",
            )
        self.tensor_refs.clear()
        for segment_key, segment in batch.segments.items():
            self._validate_segment(segment_key, segment)

        trainable_segments = {
            name for name, segment in batch.segments.items() if segment.trainable
        }
        if trainable_segments:
            if batch.primary_segment is None:
                self._fail(
                    "TrajectoryBatch.primary_segment is required when trainable segments exist",
                )
            if batch.primary_segment not in batch.segments:
                self._fail(
                    f"TrajectoryBatch.primary_segment={batch.primary_segment!r} is unknown",
                )
            if batch.primary_segment not in trainable_segments:
                self._fail(
                    f"TrajectoryBatch.primary_segment={batch.primary_segment!r} "
                    "must reference a trainable segment",
                )
        elif batch.primary_segment is not None:
            self._fail(
                "TrajectoryBatch.primary_segment must be None when no trainable segments exist",
            )

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
                    f"trainable segment {segment.name!r} has multiple {tensor.role!r} tensors",
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
                    self._fail(f"ReplayInput {replay.name!r} references unknown tensor {ref!r}")
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
                    f"tensor {segment_name}.{tensor.name} references unknown axis {axis_name!r}",
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


def require_shape_prefix(name: str, value: Any, expected: tuple[int, ...]) -> None:
    """Require ``value.shape`` to start with the ``expected`` leading dimensions."""

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < len(expected):
        raise ValueError(f"{name} must have leading dimensions {expected}")
    actual = tuple(int(length) for length in shape[: len(expected)])
    if actual != expected:
        raise ValueError(f"{name} has leading dimensions {actual}, expected {expected}")


__all__ = [
    "TrajectoryValidationError",
    "TrajectoryValidator",
    "require_shape_prefix",
    "tensor_ref",
]
