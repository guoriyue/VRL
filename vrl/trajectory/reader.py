"""Read trajectory tensors and select named replay axes."""

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
from vrl.utils.validation import require_int


class TrajectoryReaderError(ValueError):
    """Raised when a training view cannot be resolved to trajectory facts."""


@dataclass(frozen=True, slots=True)
class TrajectoryReader:
    """Read named tensors, roles, and replay slices from one trajectory."""

    trajectory: TrajectoryBatch

    def __post_init__(self) -> None:
        TrajectoryValidator(self.trajectory).validate_batch()

    @classmethod
    def from_batch(cls, batch: Any) -> TrajectoryReader:
        """Create a reader from a trainer batch that carries a trajectory."""

        trajectory = getattr(batch, "trajectory", None)
        if not isinstance(trajectory, TrajectoryBatch):
            raise TrajectoryReaderError("RolloutBatch is missing first-class TrajectoryBatch")
        return cls(trajectory=trajectory)

    def primary_trainable_segment_name(self) -> str:
        """Resolve the primary trainable segment for replay."""

        if self.trajectory.primary_segment is not None:
            return self.trajectory.primary_segment
        raise TrajectoryReaderError("TrajectoryBatch has no trainable segment")

    def tensor(self, segment_name: str, tensor_name: str) -> TrajectoryTensor:
        """Read one named tensor from a trajectory segment."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            raise TrajectoryReaderError(f"unknown trajectory segment {segment_name!r}")
        tensor = segment.tensors.get(tensor_name)
        if tensor is None:
            raise TrajectoryReaderError(
                f"segment {segment_name!r} is missing tensor {tensor_name!r}"
            )
        return tensor

    def tensor_value(self, segment_name: str, tensor_name: str) -> Any:
        """Read one named tensor value from a trajectory segment."""

        return self.tensor(segment_name, tensor_name).value

    def role_tensor(self, segment_name: str, role: TensorRole) -> TrajectoryTensor:
        """Read the unique tensor with ``role`` from a trajectory segment."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            raise TrajectoryReaderError(f"unknown trajectory segment {segment_name!r}")
        return segment.role_tensor(role)

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
        """Return replay tensors, optionally selecting one explicitly named axis.

        Axis and index must be provided together. Tensors without that axis
        retain their full value, such as static prompt embeddings during replay.
        Only ``segment_name=None`` selects the primary trainable segment.
        """

        if (axis is None) != (axis_index is None):
            raise TrajectoryReaderError("replay axis and axis_index must be provided together")
        if axis is not None:
            if axis not in self.trajectory.axes:
                raise TrajectoryReaderError(f"unknown replay axis {axis!r}")
            axis_index = require_int(axis_index, path="replay.axis_index", minimum=0)

        name = self.primary_trainable_segment_name() if segment_name is None else segment_name
        segment = self.trajectory.segments.get(name)
        if segment is None:
            raise TrajectoryReaderError(f"unknown trajectory segment {name!r}")
        replay = segment.replay_inputs.get(replay_input_name)
        if replay is None:
            raise TrajectoryReaderError(
                f"segment {name!r} is missing replay input {replay_input_name!r}"
            )
        out: dict[str, Any] = {}
        for ref in replay.tensor_refs:
            canonical_ref = ref if "." in ref else tensor_ref(name, ref)
            segment_ref, tensor_name = canonical_ref.split(".", 1)
            if not segment_ref or not tensor_name:
                raise TrajectoryReaderError(
                    f"tensor ref {canonical_ref!r} must be 'segment.name'",
                )
            if segment_ref != name:
                raise TrajectoryReaderError(
                    f"replay input {name}.{replay_input_name} crosses segment boundary "
                    f"with tensor ref {ref!r}",
                )
            tensor = self.tensor(segment_ref, tensor_name)
            value = tensor.value
            if axis is not None and axis_index is not None and axis in tensor.axes:
                value = self._slice_axis(
                    value,
                    canonical_ref,
                    tensor.axes.index(axis),
                    axis_index,
                )
            out[tensor_name] = move_value_to_device(value, device)
        return out

    @classmethod
    def _slice_axis(cls, value: Any, ref: str, axis_dim: int, axis_index: int) -> Any:
        """Select one position on a declared tensor axis, including nested sequences."""

        if isinstance(value, (list, tuple)):
            try:
                if axis_dim == 0:
                    return value[axis_index]
                return [cls._slice_axis(item, ref, axis_dim - 1, axis_index) for item in value]
            except IndexError as exc:
                raise TrajectoryReaderError(f"failed to slice tensor {ref!r}: {exc}") from exc

        shape = getattr(value, "shape", None)
        if shape is not None:
            if axis_dim >= len(shape):
                raise TrajectoryReaderError(
                    f"tensor {ref!r} rank {len(shape)} cannot slice axis dim {axis_dim}",
                )
            axis_length = shape[axis_dim]
            if axis_index >= axis_length:
                raise TrajectoryReaderError(
                    f"tensor {ref!r} axis index {axis_index} is out of range for length {axis_length}",
                )
        try:
            select = getattr(value, "select", None)
            if callable(select):
                return select(axis_dim, axis_index)
            key = [slice(None)] * (axis_dim + 1)
            key[axis_dim] = axis_index
            return value[tuple(key)]
        except Exception as exc:
            raise TrajectoryReaderError(f"failed to slice tensor {ref!r}: {exc}") from exc


__all__ = [
    "TrajectoryReader",
    "TrajectoryReaderError",
]
