"""Typed resolution helpers for trajectory training views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.engine.trajectory.types import (
    ReplayInput,
    TensorRole,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.engine.trajectory.validation import (
    tensor_ref,
    validate_training_view,
    validate_trajectory_batch,
)
from vrl.engine.trajectory.views import LossUnit, TrainingView


class TrajectoryResolverError(ValueError):
    """Raised when a training view cannot be resolved to trajectory facts."""


@dataclass(frozen=True, slots=True)
class ResolvedTrajectoryTensor:
    """A trajectory tensor after loss-unit axis slicing has been applied."""

    ref: str
    value: Any
    axes: tuple[str, ...]
    role: TensorRole
    source: TrajectoryTensor


@dataclass(frozen=True, slots=True)
class ResolvedContextValue:
    """A serializable trajectory context value required by a replay input."""

    ref: str
    value: Any


@dataclass(frozen=True, slots=True)
class ResolvedReplayInput:
    """A replay recipe with all tensor and context refs resolved."""

    ref: str
    replay_input: ReplayInput
    tensors: tuple[ResolvedTrajectoryTensor, ...]
    context_values: tuple[ResolvedContextValue, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedLossUnit:
    """Resolved tensors for one logical training loss unit."""

    loss_unit: LossUnit
    segment: TrajectorySegment
    action: ResolvedTrajectoryTensor
    old_log_prob: ResolvedTrajectoryTensor
    mask: ResolvedTrajectoryTensor
    replay_inputs: tuple[ResolvedReplayInput, ...]


@dataclass(frozen=True, slots=True)
class ResolvedTrainingView:
    """Resolved loss units for a training view."""

    trajectory: TrajectoryBatch
    view: TrainingView
    loss_units: tuple[ResolvedLossUnit, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryResolver:
    """Object entry point for reading and resolving trajectory-backed batches."""

    trajectory: TrajectoryBatch
    training_view: TrainingView | None = None

    def __post_init__(self) -> None:
        validate_trajectory_batch(self.trajectory)

    @classmethod
    def from_batch(cls, batch: Any) -> TrajectoryResolver:
        """Create a resolver from a trainer batch that carries a trajectory."""

        trajectory = getattr(batch, "trajectory", None)
        if not isinstance(trajectory, TrajectoryBatch):
            _fail("RolloutBatch is missing first-class TrajectoryBatch")
        training_view = getattr(batch, "training_view", None)
        return cls(
            trajectory=trajectory,
            training_view=training_view if isinstance(training_view, TrainingView) else None,
        )

    def primary_trainable_segment_name(self, fallback: str | None = None) -> str:
        """Resolve the primary trainable segment for replay."""

        if self.training_view is not None and self.training_view.primary_segment:
            return self.training_view.primary_segment
        if fallback is not None:
            return fallback
        for name, segment in self.trajectory.segments.items():
            if segment.trainable:
                return name
        _fail("TrajectoryBatch has no trainable segment")

    def tensor(self, segment_name: str, tensor_name: str) -> TrajectoryTensor:
        """Read one named tensor from a trajectory segment."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            _fail(f"unknown trajectory segment {segment_name!r}")
        tensor = segment.tensors.get(tensor_name)
        if tensor is None:
            _fail(f"segment {segment_name!r} is missing tensor {tensor_name!r}")
        return tensor

    def tensor_value(self, segment_name: str, tensor_name: str) -> Any:
        """Read one named tensor value from a trajectory segment."""

        return self.tensor(segment_name, tensor_name).value

    def role_tensor(self, segment_name: str, role: TensorRole) -> TrajectoryTensor:
        """Read the unique tensor with ``role`` from a trajectory segment."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            _fail(f"unknown trajectory segment {segment_name!r}")
        matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
        if len(matches) != 1:
            _fail(
                f"segment {segment_name!r} requires exactly one role {role!r}, "
                f"found {len(matches)}",
            )
        return matches[0]

    def role_value(self, segment_name: str, role: TensorRole) -> Any:
        """Read the unique tensor value with ``role`` from a trajectory segment."""

        return self.role_tensor(segment_name, role).value

    def replay_tensor_dict(
        self,
        segment_name: str | None = None,
        *,
        replay_input_name: str = "logprob",
    ) -> dict[str, Any]:
        """Return named replay tensors declared by a segment ReplayInput."""

        name = segment_name or self.primary_trainable_segment_name()
        segment = self.trajectory.segments.get(name)
        if segment is None:
            _fail(f"unknown trajectory segment {name!r}")
        replay = segment.replay_inputs.get(replay_input_name)
        if replay is None:
            _fail(f"segment {name!r} is missing replay input {replay_input_name!r}")
        out: dict[str, Any] = {}
        for ref in replay.tensor_refs:
            segment_ref, tensor_name = _split_ref(
                self._canonical_tensor_ref(name, ref),
                "tensor",
            )
            if segment_ref != name:
                _fail(
                    f"replay input {name}.{replay_input_name} crosses segment boundary "
                    f"with tensor ref {ref!r}",
                )
            out[tensor_name] = self.tensor_value(segment_ref, tensor_name)
        return out

    def resolve_training_view(self, view: TrainingView) -> ResolvedTrainingView:
        """Resolve a training view into typed tensors owned by the trajectory."""

        validate_training_view(self.trajectory, view)
        resolved_units = tuple(self._resolve_loss_unit(unit) for unit in view.loss_units)
        return ResolvedTrainingView(
            trajectory=self.trajectory,
            view=view,
            loss_units=resolved_units,
        )

    def resolve_loss_unit(self, unit: LossUnit) -> ResolvedLossUnit:
        """Resolve a single loss unit after validating it against the trajectory."""

        validate_training_view(
            self.trajectory,
            TrainingView(loss_units=(unit,), primary_segment=unit.segment),
        )
        return self._resolve_loss_unit(unit)

    def _resolve_loss_unit(self, unit: LossUnit) -> ResolvedLossUnit:
        segment = self.trajectory.segments.get(unit.segment)
        if segment is None:
            _fail(f"LossUnit.segment={unit.segment!r} is unknown")

        action = self._resolve_tensor(
            unit,
            unit.action_ref,
            expected_role="action",
            require_loss_axis=True,
        )
        old_log_prob = self._resolve_tensor(
            unit,
            unit.old_log_prob_ref,
            expected_role="old_log_prob",
            require_loss_axis=True,
        )
        mask = self._resolve_tensor(
            unit,
            unit.mask_ref,
            expected_role="mask",
            require_loss_axis=True,
        )
        _require_core_shapes_match(unit, action, old_log_prob, mask)
        replay_inputs = tuple(
            self._resolve_replay_input(unit, replay_ref)
            for replay_ref in unit.replay_input_refs
        )
        return ResolvedLossUnit(
            loss_unit=unit,
            segment=segment,
            action=action,
            old_log_prob=old_log_prob,
            mask=mask,
            replay_inputs=replay_inputs,
        )

    def _resolve_replay_input(
        self,
        unit: LossUnit,
        ref: str,
    ) -> ResolvedReplayInput:
        segment_name, replay_name = _split_ref(ref, "replay input")
        if segment_name != unit.segment:
            _fail(
                f"LossUnit replay input {ref!r} must reference segment "
                f"{unit.segment!r}",
            )
        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            _fail(f"LossUnit references unknown replay input {ref!r}")
        replay_input = segment.replay_inputs.get(replay_name)
        if replay_input is None:
            _fail(f"LossUnit references unknown replay input {ref!r}")

        resolved_tensors = tuple(
            self._resolve_tensor(
                unit,
                self._canonical_tensor_ref(segment_name, tensor_ref_value),
                expected_role=None,
                require_loss_axis=False,
            )
            for tensor_ref_value in replay_input.tensor_refs
        )
        context_values = tuple(
            self._resolve_context_value(context_ref)
            for context_ref in replay_input.context_refs
        )
        return ResolvedReplayInput(
            ref=ref,
            replay_input=replay_input,
            tensors=resolved_tensors,
            context_values=context_values,
        )

    def _resolve_tensor(
        self,
        unit: LossUnit,
        ref: str,
        *,
        expected_role: TensorRole | None,
        require_loss_axis: bool,
    ) -> ResolvedTrajectoryTensor:
        source = self._lookup_tensor(ref)
        if expected_role is not None and source.role != expected_role:
            _fail(f"tensor {ref!r} must have role {expected_role!r}, got {source.role!r}")
        if require_loss_axis and unit.axis not in source.axes:
            _fail(f"tensor {ref!r} does not contain loss axis {unit.axis!r}")

        axes = source.axes
        value = source.value
        if unit.axis_index is not None and unit.axis in axes:
            axis_dim = axes.index(unit.axis)
            value = _slice_axis(value, ref, axis_dim, unit.axis_index)
            axes = axes[:axis_dim] + axes[axis_dim + 1 :]

        _require_axis_shape(self.trajectory, ref, value, axes)
        return ResolvedTrajectoryTensor(
            ref=ref,
            value=value,
            axes=axes,
            role=source.role,
            source=source,
        )

    def _lookup_tensor(self, ref: str) -> TrajectoryTensor:
        segment_name, tensor_name = _split_ref(ref, "tensor")
        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            _fail(f"unknown tensor ref {ref!r}: segment {segment_name!r} is unknown")
        tensor = segment.tensors.get(tensor_name)
        if tensor is None:
            _fail(f"unknown tensor ref {ref!r}: tensor {tensor_name!r} is unknown")
        return tensor

    def _resolve_context_value(self, ref: str) -> ResolvedContextValue:
        if ref not in self.trajectory.context:
            _fail(f"ReplayInput references unknown context value {ref!r}")
        return ResolvedContextValue(ref=ref, value=self.trajectory.context[ref])

    def _canonical_tensor_ref(self, segment_name: str, ref: str) -> str:
        if "." in ref:
            return ref
        return tensor_ref(segment_name, ref)


def _split_ref(ref: str, kind: str) -> tuple[str, str]:
    if "." not in ref:
        _fail(f"{kind} ref {ref!r} must be 'segment.name'")
    segment_name, name = ref.split(".", 1)
    if not segment_name or not name:
        _fail(f"{kind} ref {ref!r} must be 'segment.name'")
    return segment_name, name


def _slice_axis(value: Any, ref: str, axis_dim: int, axis_index: int) -> Any:
    shape = _shape(value)
    if shape is not None:
        if axis_dim >= len(shape):
            _fail(
                f"tensor {ref!r} rank {len(shape)} cannot slice axis dim {axis_dim}",
            )
        axis_length = shape[axis_dim]
        if axis_index >= axis_length:
            _fail(
                f"tensor {ref!r} axis index {axis_index} is out of range "
                f"for length {axis_length}",
            )
    select = getattr(value, "select", None)
    if callable(select):
        try:
            return select(axis_dim, axis_index)
        except Exception as exc:  # pragma: no cover - defensive for tensor-like objects.
            _fail(f"failed to slice tensor {ref!r}: {exc}")
    try:
        key = [slice(None)] * (axis_dim + 1)
        key[axis_dim] = axis_index
        return value[tuple(key)]
    except Exception:
        return _slice_sequence_axis(value, axis_dim, axis_index, ref)


def _slice_sequence_axis(value: Any, axis_dim: int, axis_index: int, ref: str) -> Any:
    try:
        if axis_dim == 0:
            return value[axis_index]
        return [
            _slice_sequence_axis(item, axis_dim - 1, axis_index, ref)
            for item in value
        ]
    except Exception as exc:  # pragma: no cover - defensive for non-indexable values.
        _fail(f"failed to slice tensor {ref!r}: {exc}")


def _require_core_shapes_match(
    unit: LossUnit,
    action: ResolvedTrajectoryTensor,
    old_log_prob: ResolvedTrajectoryTensor,
    mask: ResolvedTrajectoryTensor,
) -> None:
    baseline_axes = action.axes
    for resolved in (old_log_prob, mask):
        if resolved.axes != baseline_axes:
            _fail(
                f"LossUnit for segment {unit.segment!r} has axis mismatch: "
                f"{resolved.ref!r} axes {resolved.axes!r} != {baseline_axes!r}",
            )

    baseline_shape = _logical_shape(action)
    for resolved in (old_log_prob, mask):
        shape = _logical_shape(resolved)
        if baseline_shape is not None and shape is not None and shape != baseline_shape:
            _fail(
                f"LossUnit for segment {unit.segment!r} has shape mismatch: "
                f"{resolved.ref!r} logical shape {shape!r} != {baseline_shape!r}",
            )

    old_shape = _shape(old_log_prob.value)
    mask_shape = _shape(mask.value)
    if old_shape is not None and mask_shape is not None and old_shape != mask_shape:
        _fail(
            f"LossUnit for segment {unit.segment!r} has shape mismatch: "
            f"old_log_prob shape {old_shape!r} != mask shape {mask_shape!r}",
        )


def _require_axis_shape(
    trajectory: TrajectoryBatch,
    ref: str,
    value: Any,
    axes: tuple[str, ...],
) -> None:
    shape = _shape(value)
    if shape is None:
        return
    if len(shape) < len(axes):
        _fail(
            f"tensor {ref!r} rank {len(shape)} is smaller than "
            f"resolved axes {axes!r}",
        )
    for dim, axis_name in enumerate(axes):
        axis = trajectory.axes.get(axis_name)
        if axis is None:
            _fail(f"tensor {ref!r} references unknown axis {axis_name!r}")
        expected = axis.length
        if expected is not None and shape[dim] != expected:
            _fail(
                f"tensor {ref!r} axis {axis_name!r} has shape {shape[dim]}, "
                f"expected {expected}",
            )


def _logical_shape(resolved: ResolvedTrajectoryTensor) -> tuple[int, ...] | None:
    shape = _shape(resolved.value)
    if shape is None:
        return None
    if len(shape) < len(resolved.axes):
        _fail(
            f"tensor {resolved.ref!r} rank {len(shape)} is smaller than "
            f"resolved axes {resolved.axes!r}",
        )
    return shape[: len(resolved.axes)]


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    try:
        return (len(value),)
    except TypeError:
        return None


def _fail(message: str) -> None:
    raise TrajectoryResolverError(message)


__all__ = [
    "ResolvedContextValue",
    "ResolvedLossUnit",
    "ResolvedReplayInput",
    "ResolvedTrainingView",
    "ResolvedTrajectoryTensor",
    "TrajectoryResolver",
    "TrajectoryResolverError",
]
