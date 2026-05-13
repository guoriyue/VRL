"""Engine planning contract shared by local and distributed runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.engine.core.capabilities import (
    AxisCapability,
    FamilyCapability,
    family_capability_from_value,
)
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.execution.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
)


@dataclass(frozen=True, slots=True)
class AxisPlan:
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
    ) -> AxisPlan:
        return cls(
            name=axis.name,
            kind=axis.kind,
            length=length,
            batchable=axis.batchable,
            chunkable=axis.chunkable,
        )


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    """One logical engine execution region and profiler label."""

    name: str
    unit_id: str = ""
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
            raise ValueError("ExecutionUnit.name must be non-empty")
        if not self.unit_id:
            object.__setattr__(self, "unit_id", self.name)
        if not self.profiler_name:
            object.__setattr__(self, "profiler_name", f"engine.{self.name}")
        if self.axis_index is not None and self.axis_index < 0:
            raise ValueError("ExecutionUnit.axis_index must be >= 0 when set")
        if self.prompt_index is not None and self.prompt_index < 0:
            raise ValueError("ExecutionUnit.prompt_index must be >= 0 when set")
        if self.sample_start is not None and self.sample_start < 0:
            raise ValueError("ExecutionUnit.sample_start must be >= 0 when set")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("ExecutionUnit.sample_count must be >= 1 when set")

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
    sample_specs: tuple[GenerationSampleSpec, ...]
    workload: WorkloadSignature
    capability: FamilyCapability
    trajectory_kind: str
    expected_axes: dict[str, AxisPlan]
    micro_batches: tuple[MicroBatchPlan, ...]
    execution_units: tuple[ExecutionUnit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def profiler_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(unit.profiler_name for unit in self.execution_units))

    @property
    def primary_chunk_unit(self) -> ExecutionUnit:
        for preferred in ("forward_chunk", "forward"):
            for unit in self.execution_units:
                if unit.name == preferred:
                    return unit
        return self.execution_units[0]

    @property
    def chunk_execution_units(self) -> tuple[ExecutionUnit, ...]:
        return tuple(unit for unit in self.execution_units if unit.name == "forward_chunk")

    def chunk_unit_for(self, chunk: MicroBatchPlan) -> ExecutionUnit:
        """Return the materialized execution unit for one micro-batch."""

        for unit in self.chunk_execution_units:
            if unit.chunk_key == chunk.chunk_key:
                return unit
        raise KeyError(f"no execution unit found for chunk {chunk.chunk_key!r}")

    def profiler_label(self, unit_name: str) -> str:
        return profiler_label_for_unit(self, unit_name)

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
            "micro_batches": [
                {
                    "prompt_index": chunk.prompt_index,
                    "chunk_key": chunk.chunk_key,
                    "sample_start": chunk.sample_start,
                    "sample_count": chunk.sample_count,
                }
                for chunk in self.micro_batches
            ],
            "execution_units": [
                {
                    "unit_id": unit.unit_id,
                    "name": unit.name,
                    "segment": unit.segment,
                    "axis": unit.axis,
                    "axis_index": unit.axis_index,
                    "prompt_index": unit.prompt_index,
                    "sample_start": unit.sample_start,
                    "sample_count": unit.sample_count,
                    "chunk_key": unit.chunk_key,
                    "profiler_name": unit.profiler_name,
                    "cache_read": unit.cache_read,
                    "cache_write": unit.cache_write,
                    "metadata": dict(unit.metadata),
                }
                for unit in self.execution_units
            ],
        }


def build_engine_plan(
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec] | None = None,
    *,
    capability: FamilyCapability | Mapping[str, Any] | None = None,
    max_samples_per_microbatch: int | None = None,
) -> EnginePlan:
    """Build a request plan while reusing the existing MicroBatchPlan contract."""

    from vrl.trainers.profiling import record_function

    with record_function("engine.plan"):
        resolved_capability = family_capability_from_value(capability)
        if resolved_capability is None:
            raise ValueError(
                "build_engine_plan requires an explicit FamilyCapability; "
                f"got None for {request.family}/{request.task}",
            )
        sample_specs_tuple = tuple(sample_specs or ())
        chunk_size = _resolve_microbatch_size(
            request,
            resolved_capability,
            max_samples_per_microbatch=max_samples_per_microbatch,
        )
        execution_plan = plan_prompt_group_microbatches(
            request.prompts,
            samples_per_prompt=request.samples_per_prompt,
            max_samples_per_microbatch=chunk_size,
            capability=resolved_capability,
        )
        axis_plans = {
            axis.name: AxisPlan.from_capability(
                axis,
                length=_axis_length(axis.name, request, sample_specs_tuple),
            )
            for axis in resolved_capability.expected_axes
        }
        execution_units = _build_execution_units(
            resolved_capability,
            request,
            axis_plans,
            execution_plan.micro_batches,
        )
        return EnginePlan(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_specs=sample_specs_tuple,
            workload=WorkloadSignature.from_request_and_capability(
                request,
                resolved_capability,
            ),
            capability=resolved_capability,
            trajectory_kind=resolved_capability.trajectory_kind,
            expected_axes=axis_plans,
            micro_batches=execution_plan.micro_batches,
            execution_units=execution_units,
            metadata={
                "samples_per_prompt": request.samples_per_prompt,
                "num_prompts": len(request.prompts),
            },
        )


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


def attach_engine_plan(output: OutputBatch, plan: EnginePlan) -> OutputBatch:
    """Attach a plan to an OutputBatch without changing decoded artifacts."""

    output.engine_plan = plan
    if output.metrics is None:
        output.metrics = GenerationMetrics(
            num_prompts=len(output.prompts),
            num_samples=len(output.sample_specs),
        )
    if output.metrics is not None:
        output.metrics.trajectory_kind = plan.trajectory_kind
        output.metrics.execution_units = plan.profiler_labels
        output.metrics.engine_plan_id = plan.request_id
    output.extra["engine_plan"] = plan.summary()
    return output


def profiler_label_for_unit(plan: EnginePlan, unit_name: str) -> str:
    """Return the profiler label for a named unit in a plan."""

    for unit in plan.execution_units:
        if unit.name == unit_name:
            return unit.profiler_name
    return f"engine.{unit_name}"


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


def _resolve_microbatch_size(
    request: GenerationRequest,
    capability: FamilyCapability,
    *,
    max_samples_per_microbatch: int | None,
) -> int:
    if max_samples_per_microbatch is not None:
        return max(1, int(max_samples_per_microbatch))
    if capability.default_max_samples_per_microbatch is not None:
        return capability.default_max_samples_per_microbatch
    return max(1, int(request.sampling.get("sample_batch_size", request.samples_per_prompt)))


def _axis_length(
    axis_name: str,
    request: GenerationRequest,
    sample_specs: tuple[GenerationSampleSpec, ...],
) -> int | None:
    sampling = request.sampling
    if axis_name == "sample":
        return len(sample_specs) if sample_specs else len(request.prompts) * request.samples_per_prompt
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


def _build_execution_units(
    capability: FamilyCapability,
    request: GenerationRequest,
    axis_plans: dict[str, AxisPlan],
    micro_batches: tuple[MicroBatchPlan, ...],
) -> tuple[ExecutionUnit, ...]:
    units = [
        ExecutionUnit(
            name="plan",
            unit_id=f"{request.request_id}:plan",
            profiler_name="engine.plan",
            batch_group_key=(request.family, request.task, capability.trajectory_kind),
        ),
    ]
    units.extend(_build_chunk_units(capability, request, micro_batches))
    for unit in capability.execution_units:
        axis_plan = axis_plans.get(unit.axis or "")
        units.append(
            ExecutionUnit(
                name=unit.name,
                unit_id=f"{request.request_id}:unit:{unit.name}",
                segment=unit.segment,
                axis=unit.axis,
                axis_index=None,
                batch_group_key=_batch_group_key(request, capability, axis_plan),
                cache_read=unit.cache_read,
                cache_write=unit.cache_write,
                profiler_name=unit.profiler_label,
                metadata=dict(unit.metadata),
            )
        )
        if unit.cache_read:
            units.append(
                ExecutionUnit(
                    name="cache_read",
                    unit_id=f"{request.request_id}:unit:{unit.name}:cache_read",
                    segment=unit.segment,
                    axis=unit.axis,
                    profiler_name="engine.cache_read",
                )
            )
        if unit.cache_write:
            units.append(
                ExecutionUnit(
                    name="cache_write",
                    unit_id=f"{request.request_id}:unit:{unit.name}:cache_write",
                    segment=unit.segment,
                    axis=unit.axis,
                    profiler_name="engine.cache_write",
                )
            )
    return tuple(units)


def _build_chunk_units(
    capability: FamilyCapability,
    request: GenerationRequest,
    micro_batches: tuple[MicroBatchPlan, ...],
) -> list[ExecutionUnit]:
    units: list[ExecutionUnit] = []
    for index, chunk in enumerate(micro_batches):
        units.append(
            ExecutionUnit(
                name="forward_chunk",
                unit_id=(
                    f"{request.request_id}:chunk:"
                    f"p{chunk.prompt_index}:s{chunk.sample_start}:n{chunk.sample_count}"
                ),
                axis="sample",
                axis_index=chunk.sample_start,
                prompt_index=chunk.prompt_index,
                sample_start=chunk.sample_start,
                sample_count=chunk.sample_count,
                batch_group_key=(request.family, request.task, capability.trajectory_kind),
                profiler_name="engine.forward_chunk",
                metadata={
                    "unit_kind": "chunk",
                    "chunk_index": index,
                    "chunk_key": chunk.chunk_key,
                },
            )
        )
    return units


def _batch_group_key(
    request: GenerationRequest,
    capability: FamilyCapability,
    axis_plan: AxisPlan | None,
) -> tuple[Any, ...]:
    return (
        request.family,
        request.task,
        capability.trajectory_kind,
        None if axis_plan is None else axis_plan.name,
        None if axis_plan is None else axis_plan.kind,
    )


__all__ = [
    "AxisPlan",
    "EnginePlan",
    "ExecutionUnit",
    "attach_engine_plan",
    "build_engine_plan",
    "profiler_label_for_unit",
    "resolve_executor_capability",
]
