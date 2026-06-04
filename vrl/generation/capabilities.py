"""Typed capability contract for engine planning.

Family routing still lives in the rollout family registry and backend-specific
flags still live in ``RuntimeBundle.runtime_caps``. This module only provides
the normalized view that the engine planner can consume. It is not user config:
configs describe what a run wants, while capabilities describe what a family
executor can safely support.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vrl.utils.validation import require_string_tuple

TrajectoryKind = Literal[
    "diffusion",
    "ar_discrete",
    "ar_continuous",
    "multisegment",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class AxisCapability:
    """One logical trajectory axis visible to the engine planner."""

    name: str
    kind: str
    batchable: bool = False
    chunkable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AxisCapability.name must be non-empty")
        if not self.kind:
            raise ValueError("AxisCapability.kind must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "batchable": self.batchable,
            "chunkable": self.chunkable,
        }

    @classmethod
    def from_value(cls, value: AxisCapability | Mapping[str, Any]) -> AxisCapability:
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            batchable=bool(value.get("batchable", False)),
            chunkable=bool(value.get("chunkable", False)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionStageCapability:
    """Planner-visible execution stage, not a full pipeline graph node."""

    name: str
    segment: str | None = None
    axis: str | None = None
    cache_read: bool = False
    cache_write: bool = False
    profiler_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExecutionStageCapability.name must be non-empty")
        if self.profiler_name is not None and not self.profiler_name:
            raise ValueError(
                "ExecutionStageCapability.profiler_name must be non-empty when set"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "segment": self.segment,
            "axis": self.axis,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "profiler_name": self.profiler_name,
            "metadata": dict(self.metadata),
        }

    @property
    def profiler_label(self) -> str:
        return self.profiler_name or f"engine.{self.name}"

    @classmethod
    def from_value(
        cls,
        value: ExecutionStageCapability | Mapping[str, Any],
    ) -> ExecutionStageCapability:
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value["name"]),
            segment=None if value.get("segment") is None else str(value.get("segment")),
            axis=None if value.get("axis") is None else str(value.get("axis")),
            cache_read=bool(value.get("cache_read", False)),
            cache_write=bool(value.get("cache_write", False)),
            profiler_name=None if value.get("profiler_name") is None else str(value.get("profiler_name")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class FamilyCapability:
    """Planner-facing capability snapshot for one family/task pair."""

    family: str
    task: str
    trajectory_kind: TrajectoryKind
    expected_axes: tuple[AxisCapability, ...]
    execution_stages: tuple[ExecutionStageCapability, ...]
    trainable_segments: tuple[str, ...] = ()
    reward_views: tuple[str, ...] = ()
    supports_batched_requests: bool = True
    supports_chunked_execution: bool = True
    supports_batched_forward: bool = True
    supports_stepwise: bool = False
    supports_cfg: bool = False
    supports_batched_decode: bool = False
    supports_reference_conditioning: bool = False
    supports_token_logprobs: bool = False
    supports_kv_decode: bool = False
    supports_prefill_decode_split: bool = False
    supports_resident_rollout_state: bool = False
    supports_torch_compile: bool = False
    supports_cuda_graph: bool = False
    cache_kinds: tuple[str, ...] = ()
    default_max_samples_per_chunk: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("FamilyCapability.family must be non-empty")
        if not self.task:
            raise ValueError("FamilyCapability.task must be non-empty")
        if not self.expected_axes:
            raise ValueError("FamilyCapability.expected_axes must be non-empty")
        if not self.execution_stages:
            raise ValueError("FamilyCapability.execution_stages must be non-empty")
        require_string_tuple("FamilyCapability.trainable_segments", self.trainable_segments)
        require_string_tuple("FamilyCapability.reward_views", self.reward_views)
        require_string_tuple("FamilyCapability.cache_kinds", self.cache_kinds)
        if (
            self.default_max_samples_per_chunk is not None
            and self.default_max_samples_per_chunk < 1
        ):
            raise ValueError(
                "FamilyCapability.default_max_samples_per_chunk must be >= 1"
            )

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.expected_axes)

    @property
    def batchable_axes(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.expected_axes if axis.batchable)

    @property
    def chunkable_axes(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.expected_axes if axis.chunkable)

    @property
    def profiler_labels(self) -> tuple[str, ...]:
        return tuple(stage.profiler_label for stage in self.execution_stages)

    def batch_signature(self) -> tuple[Any, ...]:
        """Return the capability portion of a request batching key."""

        return (
            self.trajectory_kind,
            self.axis_names,
            self.batchable_axes,
            self.supports_batched_requests,
            self.supports_batched_forward,
        )

    def with_runtime_caps(self, runtime_caps: Mapping[str, Any] | None) -> FamilyCapability:
        """Merge backend-loaded flags without changing static trajectory facts."""

        if not runtime_caps:
            return self
        updates: dict[str, Any] = {}
        bool_fields = (
            "supports_batched_requests",
            "supports_chunked_execution",
            "supports_batched_forward",
            "supports_stepwise",
            "supports_cfg",
            "supports_batched_decode",
            "supports_reference_conditioning",
            "supports_token_logprobs",
            "supports_kv_decode",
            "supports_prefill_decode_split",
            "supports_resident_rollout_state",
            "supports_torch_compile",
            "supports_cuda_graph",
        )
        for field_name in bool_fields:
            if field_name in runtime_caps:
                updates[field_name] = bool(runtime_caps[field_name])
        if "cache_kinds" in runtime_caps:
            updates["cache_kinds"] = tuple(str(item) for item in runtime_caps["cache_kinds"])
        if "default_max_samples_per_chunk" in runtime_caps:
            value = runtime_caps["default_max_samples_per_chunk"]
            updates["default_max_samples_per_chunk"] = (
                None if value is None else int(value)
            )
        if "family_capability" in runtime_caps:
            dynamic = family_capability_from_value(runtime_caps["family_capability"])
            if dynamic is not None:
                updates.update(
                    {
                        "trajectory_kind": dynamic.trajectory_kind,
                        "expected_axes": dynamic.expected_axes,
                        "execution_stages": dynamic.execution_stages,
                        "trainable_segments": dynamic.trainable_segments,
                        "reward_views": dynamic.reward_views,
                    }
                )
        if not updates:
            return self
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task": self.task,
            "trajectory_kind": self.trajectory_kind,
            "expected_axes": [axis.to_dict() for axis in self.expected_axes],
            "execution_stages": [stage.to_dict() for stage in self.execution_stages],
            "trainable_segments": list(self.trainable_segments),
            "reward_views": list(self.reward_views),
            "supports_batched_requests": self.supports_batched_requests,
            "supports_chunked_execution": self.supports_chunked_execution,
            "supports_batched_forward": self.supports_batched_forward,
            "supports_stepwise": self.supports_stepwise,
            "supports_cfg": self.supports_cfg,
            "supports_batched_decode": self.supports_batched_decode,
            "supports_reference_conditioning": self.supports_reference_conditioning,
            "supports_token_logprobs": self.supports_token_logprobs,
            "supports_kv_decode": self.supports_kv_decode,
            "supports_prefill_decode_split": self.supports_prefill_decode_split,
            "supports_resident_rollout_state": self.supports_resident_rollout_state,
            "supports_torch_compile": self.supports_torch_compile,
            "supports_cuda_graph": self.supports_cuda_graph,
            "cache_kinds": list(self.cache_kinds),
            "default_max_samples_per_chunk": self.default_max_samples_per_chunk,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(
        cls,
        value: FamilyCapability | Mapping[str, Any],
    ) -> FamilyCapability:
        if isinstance(value, cls):
            return value
        return cls(
            family=str(value["family"]),
            task=str(value["task"]),
            trajectory_kind=_trajectory_kind(value.get("trajectory_kind", "unknown")),
            expected_axes=tuple(
                AxisCapability.from_value(axis)
                for axis in value.get("expected_axes", ())
            ),
            execution_stages=tuple(
                ExecutionStageCapability.from_value(stage)
                for stage in value.get("execution_stages", ())
            ),
            trainable_segments=tuple(str(item) for item in value.get("trainable_segments", ())),
            reward_views=tuple(str(item) for item in value.get("reward_views", ())),
            supports_batched_requests=bool(value.get("supports_batched_requests", True)),
            supports_chunked_execution=bool(value.get("supports_chunked_execution", True)),
            supports_batched_forward=bool(value.get("supports_batched_forward", True)),
            supports_stepwise=bool(value.get("supports_stepwise", False)),
            supports_cfg=bool(value.get("supports_cfg", False)),
            supports_batched_decode=bool(value.get("supports_batched_decode", False)),
            supports_reference_conditioning=bool(
                value.get("supports_reference_conditioning", False)
            ),
            supports_token_logprobs=bool(value.get("supports_token_logprobs", False)),
            supports_kv_decode=bool(value.get("supports_kv_decode", False)),
            supports_prefill_decode_split=bool(
                value.get("supports_prefill_decode_split", False)
            ),
            supports_resident_rollout_state=bool(
                value.get("supports_resident_rollout_state", False)
            ),
            supports_torch_compile=bool(value.get("supports_torch_compile", False)),
            supports_cuda_graph=bool(value.get("supports_cuda_graph", False)),
            cache_kinds=tuple(str(item) for item in value.get("cache_kinds", ())),
            default_max_samples_per_chunk=None if value.get("default_max_samples_per_chunk") is None else int(value.get("default_max_samples_per_chunk")),
            metadata=dict(value.get("metadata") or {}),
        )


def family_capability_from_value(value: Any) -> FamilyCapability | None:
    """Normalize a serialized or typed capability value."""

    if value is None:
        return None
    if isinstance(value, FamilyCapability):
        return value
    if isinstance(value, Mapping):
        return FamilyCapability.from_value(value)
    raise TypeError(
        "family capability must be a FamilyCapability, mapping, or None; "
        f"got {type(value).__name__}"
    )


def _trajectory_kind(value: Any) -> TrajectoryKind:
    text = str(value)
    if text in {"diffusion", "ar_discrete", "ar_continuous", "multisegment", "unknown"}:
        return text  # type: ignore[return-value]
    raise ValueError(f"unsupported trajectory_kind: {value!r}")


__all__ = [
    "AxisCapability",
    "ExecutionStageCapability",
    "FamilyCapability",
    "TrajectoryKind",
    "family_capability_from_value",
]
