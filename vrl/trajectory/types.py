"""Serializable trajectory record types.

These dataclasses describe rollout facts needed by reward, replay, and
training. They intentionally do not own runtime state such as KV-cache handles,
Ray actors, model modules, or scheduler objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, get_args

from vrl.generation.types import GenerationSampleRow
from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.trajectory.views import RewardInputSpec

AxisKind = Literal[
    "sample",
    "discrete_token",
    "continuous_token",
    "denoise_step",
    "temporal_chunk",
    "denoise_transition",
    "text_token",
    "segment",
    "frame",
    "media",
    "custom",
]
TensorRole = Literal[
    "observation",
    "action",
    "old_log_prob",
    "mask",
    "replay_input",
]
SegmentModality = Literal["image", "video", "text", "latent", "mixed", "unknown"]
DistributionKind = Literal[
    "categorical",
    "gaussian",
    "flow_matching",
    "deterministic",
    "custom",
]


def validate_string_tuple(name: str, values: tuple[str, ...]) -> None:
    """Raise ValueError if any element of ``values`` is not a non-empty string.

    Guards the ``tensor_refs`` tuples that name tensors inside a segment — here
    for ``ReplayInput`` and in ``vrl.trajectory.views`` for ``RewardInputSpec``. An
    empty or non-string ref would only fail much later, at resolve time, with no
    pointer back to the record that declared it.
    """

    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class TrajectoryAxis:
    """Named logical axis used by trajectory tensors."""

    name: str
    kind: AxisKind
    length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TrajectoryAxis.name must be a non-empty string")
        if self.kind not in get_args(AxisKind):
            raise ValueError(f"unknown TrajectoryAxis.kind {self.kind!r}")
        if self.length is not None:
            require_int(self.length, path="TrajectoryAxis.length", minimum=0)


@dataclass(slots=True)
class TrajectoryTensor:
    """One tensor-like trajectory fact with explicit logical axes."""

    name: str
    value: Any
    axes: tuple[str, ...]
    role: TensorRole

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectoryTensor.name must be non-empty")
        if not self.axes:
            raise ValueError("TrajectoryTensor.axes must be non-empty")
        if self.role not in get_args(TensorRole):
            raise ValueError(f"unknown TrajectoryTensor.role {self.role!r}")


@dataclass(frozen=True, slots=True)
class ReplayInput:
    """Serializable replay recipe for re-computing training signals."""

    name: str
    tensor_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ReplayInput.name must be non-empty")
        validate_string_tuple("ReplayInput.tensor_refs", self.tensor_refs)


@dataclass(slots=True)
class TrajectorySegment:
    """A trainable or non-trainable segment inside a trajectory."""

    name: str
    modality: SegmentModality
    trainable: bool
    distribution: DistributionKind
    tensors: dict[str, TrajectoryTensor]
    reward_view: str | None = None
    replay_inputs: dict[str, ReplayInput] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectorySegment.name must be non-empty")
        if self.modality not in get_args(SegmentModality):
            raise ValueError(f"unknown TrajectorySegment.modality {self.modality!r}")
        if self.distribution not in get_args(DistributionKind):
            raise ValueError(f"unknown TrajectorySegment.distribution {self.distribution!r}")
        if self.reward_view is not None and not self.reward_view:
            raise ValueError("TrajectorySegment.reward_view must be non-empty when set")

    def role_tensor(self, role: str) -> TrajectoryTensor:
        matches = [tensor for tensor in self.tensors.values() if tensor.role == role]
        if len(matches) != 1:
            raise RuntimeError(
                f"segment {self.name!r} requires exactly one role {role!r}, found {len(matches)}",
            )
        return matches[0]


@dataclass(slots=True)
class TrajectoryBatch:
    """First-class trajectory record emitted by generation runtimes."""

    request_id: str
    # display/provenance-only: the request identity this record was built from.
    # Behavior keys off ``request_id`` (batch joins) and the segments/views; the
    # family/task tokens are carried so a serialized record names its origin.
    family: str
    task: str
    sample_rows: list[GenerationSampleRow]
    axes: dict[str, TrajectoryAxis]
    segments: dict[str, TrajectorySegment]
    primary_segment: str | None = None
    reward_views: dict[str, RewardInputSpec] = field(default_factory=dict)
    # Batch-shared replay metadata; sample-aligned values belong in segment tensors.
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("TrajectoryBatch.request_id must be non-empty")
        if not self.family:
            raise ValueError("TrajectoryBatch.family must be non-empty")
        if not self.task:
            raise ValueError("TrajectoryBatch.task must be non-empty")

    def select_samples(self, selector: Any) -> TrajectoryBatch:
        """Select TrajectoryBatch rows by boolean mask or integer indices."""

        positions = self._sample_positions(selector)
        count = len(positions)
        return self._rebuild(
            sample_rows=[self.sample_rows[i] for i in positions],
            tensor_value_fn=lambda tensor: (
                self._select_sample_values(tensor.value, positions, tensor.axes.index("sample"))
                if "sample" in tensor.axes
                else tensor.value
            ),
            axes_sample_length=count,
            context=dict(self.context),
        )

    def to_device(self, device: Any) -> TrajectoryBatch:
        """Move tensor leaves in a TrajectoryBatch to a target device."""

        from vrl.trajectory.device import move_value_to_device

        return self._rebuild(
            sample_rows=list(self.sample_rows),
            tensor_value_fn=lambda tensor: move_value_to_device(tensor.value, device),
            axes_sample_length=self.axes["sample"].length,
            context=move_value_to_device(self.context, device),
        )

    def _rebuild(
        self,
        *,
        sample_rows: list[GenerationSampleRow],
        tensor_value_fn: Callable[[TrajectoryTensor], Any],
        axes_sample_length: int | None,
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        from vrl.trajectory.validation import TrajectoryValidator

        axes = {
            name: replace(axis, length=axes_sample_length) if name == "sample" else axis
            for name, axis in self.axes.items()
        }
        segments = {
            name: replace(
                segment,
                tensors={
                    tensor_name: replace(tensor, value=tensor_value_fn(tensor))
                    for tensor_name, tensor in segment.tensors.items()
                },
                replay_inputs=dict(segment.replay_inputs),
                metadata=dict(segment.metadata),
            )
            for name, segment in self.segments.items()
        }

        out = replace(
            self,
            sample_rows=sample_rows,
            axes=axes,
            segments=segments,
            reward_views=dict(self.reward_views),
            context=context,
        )
        return TrajectoryValidator(out).validate_batch()

    @classmethod
    def _select_sample_values(cls, value: Any, positions: list[int], axis_dim: int) -> Any:
        """Select the declared sample dimension, including nested Python payloads."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            selected = (
                [value[i] for i in positions]
                if axis_dim == 0
                else [cls._select_sample_values(inner, positions, axis_dim - 1) for inner in value]
            )
            return tuple(selected) if isinstance(value, tuple) else selected
        if isinstance(value, dict):
            return {
                key: cls._select_sample_values(inner, positions, axis_dim)
                for key, inner in value.items()
            }
        key = [slice(None)] * (axis_dim + 1)
        key[axis_dim] = positions
        return value[tuple(key)]

    def _sample_positions(self, selector: Any) -> list[int]:
        """Normalize one-dimensional masks/indices once for every sample-aligned value."""
        is_boolean_mask = False
        if hasattr(selector, "detach"):
            if selector.is_floating_point() or selector.is_complex():
                raise ValueError("trajectory selector must contain only booleans or only integers")
            is_boolean_mask = str(selector.dtype) == "torch.bool"
            if selector.ndim != 1:
                raise ValueError("trajectory selector must be one-dimensional")
            values = selector.detach().cpu().tolist()
        else:
            values = list(selector)
        if is_boolean_mask or (values and all(isinstance(value, bool) for value in values)):
            if len(values) != len(self.sample_rows):
                raise ValueError("trajectory boolean selector must match the sample count")
            return [index for index, selected in enumerate(values) if selected]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("trajectory selector must contain only booleans or only integers")
        return values


__all__ = [
    "AxisKind",
    "DistributionKind",
    "ReplayInput",
    "SegmentModality",
    "TensorRole",
    "TrajectoryAxis",
    "TrajectoryBatch",
    "TrajectorySegment",
    "TrajectoryTensor",
]
