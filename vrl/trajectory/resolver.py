"""Typed resolution helpers for trajectory training views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.trajectory.device import move_value_to_device
from vrl.trajectory.types import (
    TensorRole,
    TrajectoryBatch,
    TrajectoryTensor,
)
from vrl.trajectory.validation import (
    TrajectoryValidator,
    tensor_ref,
)
from vrl.trajectory.views import TrainingView
from vrl.trajectory.views import role_tensor as _role_tensor_of_segment


class TrajectoryResolverError(ValueError):
    """Raised when a training view cannot be resolved to trajectory facts."""


@dataclass(frozen=True, slots=True)
class TrajectoryResolver:
    """Object entry point for reading and resolving trajectory-backed batches."""

    trajectory: TrajectoryBatch
    training_view: TrainingView | None = None

    def __post_init__(self) -> None:
        TrajectoryValidator(self.trajectory).validate_batch()

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
        return _role_tensor_of_segment(segment, role)

    def role_value(self, segment_name: str, role: TensorRole) -> Any:
        """Read the unique tensor value with ``role`` from a trajectory segment."""

        return self.role_tensor(segment_name, role).value

    def replay_tensor_dict(
        self,
        segment_name: str | None = None,
        *,
        replay_input_name: str = "logprob",
        axis: str | None = None,
        axis_index: int | None = None,
        device: Any | None = None,
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
            tensor = self.tensor(segment_ref, tensor_name)
            value = tensor.value
            if axis is not None and axis_index is not None and axis in tensor.axes:
                value = _slice_axis(
                    value,
                    tensor_ref(segment_ref, tensor_name),
                    tensor.axes.index(axis),
                    axis_index,
                )
            out[tensor_name] = move_value_to_device(value, device)
        return out

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
    "TrajectoryResolver",
    "TrajectoryResolverError",
]
