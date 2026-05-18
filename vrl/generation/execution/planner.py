"""Engine planning contract shared by local and distributed runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.capabilities import (
    AxisCapability,
    FamilyCapability,
    family_capability_from_value,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    build_prompt_chunk_schedule,
)
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    WorkloadSignature,
)


@dataclass(frozen=True, slots=True)
class ResolvedAxis:
    """Resolved axis metadata for one request."""

    name: str
    kind: str
    length: int | None
    batchable: bool = False
    chunkable: bool = False

    @classmethod
    def from_capability(
        cls,
        axis: AxisCapability,
        *,
        length: int | None,
    ) -> ResolvedAxis:
        return cls(
            name=axis.name,
            kind=axis.kind,
            length=length,
            batchable=axis.batchable,
            chunkable=axis.chunkable,
        )


@dataclass(frozen=True, slots=True)
class ExecutionStage:
    """One planner-visible execution stage and profiler label."""

    name: str
    stage_id: str = ""
    segment: str | None = None
    axis: str | None = None
    axis_index: int | None = None
    prompt_index: int | None = None
    sample_start: int | None = None
    sample_count: int | None = None
    batch_group_key: tuple[Any, ...] = ()
    cache_read: bool = False
    cache_write: bool = False
    profiler_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExecutionStage.name must be non-empty")
        if not self.stage_id:
            object.__setattr__(self, "stage_id", self.name)
        if not self.profiler_name:
            object.__setattr__(self, "profiler_name", f"engine.{self.name}")
        if self.axis_index is not None and self.axis_index < 0:
            raise ValueError("ExecutionStage.axis_index must be >= 0 when set")
        if self.prompt_index is not None and self.prompt_index < 0:
            raise ValueError("ExecutionStage.prompt_index must be >= 0 when set")
        if self.sample_start is not None and self.sample_start < 0:
            raise ValueError("ExecutionStage.sample_start must be >= 0 when set")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("ExecutionStage.sample_count must be >= 1 when set")

    @property
    def chunk_key(self) -> str | None:
        if (
            self.prompt_index is None
            or self.sample_start is None
            or self.sample_count is None
        ):
            return None
        sample_end = self.sample_start + self.sample_count
        return f"prompt:{self.prompt_index}:samples:{self.sample_start}:{sample_end}"


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Request-level planning envelope for engine execution."""

    request_id: str
    family: str
    task: str
    sample_rows: tuple[GenerationSampleRow, ...]
    workload: WorkloadSignature
    capability: FamilyCapability
    trajectory_kind: str
    expected_axes: dict[str, ResolvedAxis]
    chunks: tuple[SampleChunk, ...]
    execution_stages: tuple[ExecutionStage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def profiler_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(stage.profiler_name for stage in self.execution_stages))

    @property
    def primary_chunk_stage(self) -> ExecutionStage:
        for preferred in ("forward_chunk", "forward"):
            for stage in self.execution_stages:
                if stage.name == preferred:
                    return stage
        return self.execution_stages[0]

    @property
    def chunk_execution_stages(self) -> tuple[ExecutionStage, ...]:
        return tuple(
            stage for stage in self.execution_stages if stage.name == "forward_chunk"
        )

    def chunk_stage_for(self, sample_chunk: SampleChunk) -> ExecutionStage:
        """Return the materialized execution stage for one sample chunk."""

        for stage in self.chunk_execution_stages:
            if stage.chunk_key == sample_chunk.chunk_key:
                return stage
        raise KeyError(f"no execution stage found for chunk {sample_chunk.chunk_key!r}")

    def profiler_label(self, stage_name: str) -> str:
        for stage in self.execution_stages:
            if stage.name == stage_name:
                return stage.profiler_name
        return f"engine.{stage_name}"

    def summary(self) -> dict[str, Any]:
        """Return a lightweight, serializable plan summary for logs/metrics."""

        return {
            "request_id": self.request_id,
            "family": self.family,
            "task": self.task,
            "trajectory_kind": self.trajectory_kind,
            "profiler_labels": list(self.profiler_labels),
            "capability": self.capability.to_dict(),
            "axes": {
                name: {
                    "kind": axis.kind,
                    "length": axis.length,
                    "batchable": axis.batchable,
                    "chunkable": axis.chunkable,
                }
                for name, axis in self.expected_axes.items()
            },
            "chunks": [
                {
                    "prompt_index": sample_chunk.prompt_index,
                    "chunk_key": sample_chunk.chunk_key,
                    "sample_start": sample_chunk.sample_start,
                    "sample_count": sample_chunk.sample_count,
                }
                for sample_chunk in self.chunks
            ],
            "execution_stages": [
                {
                    "stage_id": stage.stage_id,
                    "name": stage.name,
                    "segment": stage.segment,
                    "axis": stage.axis,
                    "axis_index": stage.axis_index,
                    "prompt_index": stage.prompt_index,
                    "sample_start": stage.sample_start,
                    "sample_count": stage.sample_count,
                    "chunk_key": stage.chunk_key,
                    "profiler_name": stage.profiler_name,
                    "cache_read": stage.cache_read,
                    "cache_write": stage.cache_write,
                    "metadata": dict(stage.metadata),
                }
                for stage in self.execution_stages
            ],
        }


@dataclass(frozen=True, slots=True)
class EnginePlanner:
    """Build an EnginePlan for one request and resolved family capability."""

    request: GenerationRequest
    capability: FamilyCapability
    sample_rows: tuple[GenerationSampleRow, ...] = ()
    max_samples_per_chunk: int | None = None

    def build(self) -> EnginePlan:
        """Build the immutable execution plan."""

        from vrl.utils.profiling import record_function

        with record_function("engine.plan"):
            return self._build()

    def _build(self) -> EnginePlan:
        chunk_schedule = build_prompt_chunk_schedule(
            self.request.prompts,
            samples_per_prompt=self.request.samples_per_prompt,
            max_samples_per_chunk=self._chunk_size(),
            capability=self.capability,
        )
        resolved_axes = self._resolved_axes()
        execution_stages = self._execution_stages(
            resolved_axes,
            chunk_schedule.chunks,
        )
        return EnginePlan(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=self.sample_rows,
            workload=WorkloadSignature.from_request_and_capability(
                self.request,
                self.capability,
            ),
            capability=self.capability,
            trajectory_kind=self.capability.trajectory_kind,
            expected_axes=resolved_axes,
            chunks=chunk_schedule.chunks,
            execution_stages=execution_stages,
            metadata={
                "samples_per_prompt": self.request.samples_per_prompt,
                "num_prompts": len(self.request.prompts),
            },
        )

    def _chunk_size(self) -> int:
        if self.max_samples_per_chunk is not None:
            return max(1, int(self.max_samples_per_chunk))
        if self.capability.default_max_samples_per_chunk is not None:
            return self.capability.default_max_samples_per_chunk
        return max(
            1,
            int(
                self.request.sampling.get(
                    "sample_batch_size",
                    self.request.samples_per_prompt,
                )
            ),
        )

    def _resolved_axes(self) -> dict[str, ResolvedAxis]:
        return {
            axis.name: ResolvedAxis.from_capability(
                axis,
                length=self._axis_length(axis.name),
            )
            for axis in self.capability.expected_axes
        }

    def _axis_length(self, axis_name: str) -> int | None:
        sampling = self.request.sampling
        if axis_name == "sample":
            return (
                len(self.sample_rows)
                if self.sample_rows
                else len(self.request.prompts) * self.request.samples_per_prompt
            )
        if axis_name == "timestep":
            value = sampling.get("num_steps", sampling.get("num_inference_steps"))
            return None if value is None else int(value)
        if axis_name == "token":
            value = sampling.get(
                "image_token_num",
                sampling.get("max_new_image_tokens", sampling.get("max_new_tokens")),
            )
            return None if value is None else int(value)
        if axis_name == "segment":
            return None
        return None

    def _execution_stages(
        self,
        resolved_axes: dict[str, ResolvedAxis],
        sample_chunks: tuple[SampleChunk, ...],
    ) -> tuple[ExecutionStage, ...]:
        units = [
            ExecutionStage(
                name="plan",
                stage_id=f"{self.request.request_id}:stage:plan",
                profiler_name="engine.plan",
                batch_group_key=(
                    self.request.family,
                    self.request.task,
                    self.capability.trajectory_kind,
                ),
            ),
        ]
        units.extend(self._chunk_stages(sample_chunks))
        for stage in self.capability.execution_stages:
            axis = resolved_axes.get(stage.axis or "")
            units.append(
                ExecutionStage(
                    name=stage.name,
                    stage_id=f"{self.request.request_id}:stage:{stage.name}",
                    segment=stage.segment,
                    axis=stage.axis,
                    axis_index=None,
                    batch_group_key=self._batch_group_key(axis),
                    cache_read=stage.cache_read,
                    cache_write=stage.cache_write,
                    profiler_name=stage.profiler_label,
                    metadata=dict(stage.metadata),
                )
            )
            if stage.cache_read:
                units.append(
                    ExecutionStage(
                        name="cache_read",
                        stage_id=(
                            f"{self.request.request_id}:stage:{stage.name}:cache_read"
                        ),
                        segment=stage.segment,
                        axis=stage.axis,
                        profiler_name="engine.cache_read",
                    )
                )
            if stage.cache_write:
                units.append(
                    ExecutionStage(
                        name="cache_write",
                        stage_id=(
                            f"{self.request.request_id}:stage:{stage.name}:cache_write"
                        ),
                        segment=stage.segment,
                        axis=stage.axis,
                        profiler_name="engine.cache_write",
                    )
                )
        return tuple(units)

    def _chunk_stages(
        self,
        sample_chunks: tuple[SampleChunk, ...],
    ) -> list[ExecutionStage]:
        units: list[ExecutionStage] = []
        for index, sample_chunk in enumerate(sample_chunks):
            units.append(
                ExecutionStage(
                    name="forward_chunk",
                    stage_id=(
                        f"{self.request.request_id}:chunk:"
                        f"p{sample_chunk.prompt_index}:"
                        f"s{sample_chunk.sample_start}:"
                        f"n{sample_chunk.sample_count}"
                    ),
                    axis="sample",
                    axis_index=sample_chunk.sample_start,
                    prompt_index=sample_chunk.prompt_index,
                    sample_start=sample_chunk.sample_start,
                    sample_count=sample_chunk.sample_count,
                    batch_group_key=(
                        self.request.family,
                        self.request.task,
                        self.capability.trajectory_kind,
                    ),
                    profiler_name="engine.forward_chunk",
                    metadata={
                        "stage_kind": "chunk",
                        "chunk_index": index,
                        "chunk_key": sample_chunk.chunk_key,
                    },
                )
            )
        return units

    def _batch_group_key(self, axis: ResolvedAxis | None) -> tuple[Any, ...]:
        return (
            self.request.family,
            self.request.task,
            self.capability.trajectory_kind,
            None if axis is None else axis.name,
            None if axis is None else axis.kind,
        )


def build_engine_plan(
    request: GenerationRequest,
    sample_rows: Sequence[GenerationSampleRow] | None = None,
    *,
    capability: FamilyCapability | Mapping[str, Any] | None = None,
    max_samples_per_chunk: int | None = None,
) -> EnginePlan:
    """Build a request-level plan from sample rows and family capability metadata."""

    resolved_capability = family_capability_from_value(capability)
    if resolved_capability is None:
        raise ValueError(
            "build_engine_plan requires an explicit FamilyCapability; "
            f"got None for {request.family}/{request.task}",
        )
    return EnginePlanner(
        request=request,
        capability=resolved_capability,
        sample_rows=tuple(sample_rows or ()),
        max_samples_per_chunk=max_samples_per_chunk,
    ).build()


def resolve_executor_capability(
    executor: Any,
    request: GenerationRequest,
) -> FamilyCapability:
    """Resolve the explicit planning capability declared by an executor."""

    method = getattr(executor, "capability", None)
    if callable(method):
        return _merge_runtime_caps(method(), executor, request)
    for attr_name in ("family_capability", "capability_metadata"):
        value = getattr(executor, attr_name, None)
        if value is not None:
            return _merge_runtime_caps(value, executor, request)
    raise ValueError(
        f"{type(executor).__name__} must declare FamilyCapability for "
        f"{request.family}/{request.task}",
    )


def attach_engine_plan(output: GenerationOutput, plan: EnginePlan) -> GenerationOutput:
    """Attach a plan to a GenerationOutput without changing decoded artifacts."""

    output.engine_plan = plan
    if output.metrics is None:
        output.metrics = GenerationMetrics(
            num_prompts=len(output.prompts),
            num_samples=len(output.sample_rows),
        )
    if output.metrics is not None:
        output.metrics.trajectory_kind = plan.trajectory_kind
        output.metrics.execution_stages = plan.profiler_labels
        output.metrics.engine_plan_id = plan.request_id
    output.extra["engine_plan"] = plan.summary()
    return output


def _merge_runtime_caps(
    value: Any,
    executor: Any,
    request: GenerationRequest,
) -> FamilyCapability:
    capability = family_capability_from_value(value)
    if capability is None:
        raise ValueError(
            f"{type(executor).__name__} declared an empty FamilyCapability for "
            f"{request.family}/{request.task}",
        )
    runtime_caps = getattr(executor, "runtime_caps", None)
    if isinstance(runtime_caps, Mapping):
        return capability.with_runtime_caps(runtime_caps)
    return capability


__all__ = [
    "EnginePlan",
    "EnginePlanner",
    "ExecutionStage",
    "ResolvedAxis",
    "attach_engine_plan",
    "build_engine_plan",
    "resolve_executor_capability",
]
