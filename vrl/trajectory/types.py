"""Serializable trajectory record types.

These dataclasses describe rollout facts needed by reward, replay, and
training. They intentionally do not own runtime state such as KV-cache handles,
Ray actors, model modules, or scheduler objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vrl.generation.types import GenerationSampleRow
from vrl.utils.validation import require_string_tuple

if TYPE_CHECKING:
    from vrl.trajectory.views import RewardView

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
AdvantageScope = Literal["sample", "segment", "axis"]


@dataclass(frozen=True, slots=True)
class TrajectoryAxis:
    """Named logical axis used by trajectory tensors."""

    name: str
    kind: AxisKind
    length: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectoryAxis.name must be non-empty")
        if self.length is not None and self.length < 0:
            raise ValueError("TrajectoryAxis.length must be >= 0 when set")


@dataclass(slots=True)
class TrajectoryTensor:
    """One tensor-like trajectory fact with explicit logical axes."""

    name: str
    value: Any
    axes: tuple[str, ...]
    role: TensorRole
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectoryTensor.name must be non-empty")
        if not self.axes:
            raise ValueError("TrajectoryTensor.axes must be non-empty")


@dataclass(frozen=True, slots=True)
class ReplayInput:
    """Serializable replay recipe for re-computing training signals."""

    name: str
    tensor_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ReplayInput.name must be non-empty")
        require_string_tuple("ReplayInput.tensor_refs", self.tensor_refs)


@dataclass(slots=True)
class TrajectorySegment:
    """A trainable or non-trainable segment inside a trajectory."""

    name: str
    modality: SegmentModality
    trainable: bool
    distribution: DistributionKind
    tensors: dict[str, TrajectoryTensor]
    reward_view: str | None = None
    advantage_scope: AdvantageScope = "sample"
    replay_inputs: dict[str, ReplayInput] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectorySegment.name must be non-empty")
        if self.reward_view is not None and not self.reward_view:
            raise ValueError("TrajectorySegment.reward_view must be non-empty when set")


@dataclass(slots=True)
class TrajectoryMetrics:
    """Serializable statistics about the trajectory record itself."""

    num_samples: int | None = None
    axis_lengths: dict[str, int] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_samples is not None and self.num_samples < 0:
            raise ValueError("TrajectoryMetrics.num_samples must be >= 0 when set")
        for name, length in self.axis_lengths.items():
            if not name:
                raise ValueError("TrajectoryMetrics.axis_lengths keys must be non-empty")
            if int(length) < 0:
                raise ValueError("TrajectoryMetrics.axis_lengths values must be >= 0")


@dataclass(slots=True)
class TrajectoryBatch:
    """First-class trajectory record emitted by generation runtimes."""

    request_id: str
    family: str
    task: str
    sample_rows: list[GenerationSampleRow]
    group_ids: Any
    axes: dict[str, TrajectoryAxis]
    segments: dict[str, TrajectorySegment]
    reward_views: dict[str, RewardView] = field(default_factory=dict)
    metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("TrajectoryBatch.request_id must be non-empty")
        if not self.family:
            raise ValueError("TrajectoryBatch.family must be non-empty")
        if not self.task:
            raise ValueError("TrajectoryBatch.task must be non-empty")


__all__ = [
    "AdvantageScope",
    "AxisKind",
    "DistributionKind",
    "ReplayInput",
    "SegmentModality",
    "TensorRole",
    "TrajectoryAxis",
    "TrajectoryBatch",
    "TrajectoryMetrics",
    "TrajectorySegment",
    "TrajectoryTensor",
]
