"""Generation pipeline executor protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.engine.core.capabilities import FamilyCapability
    from vrl.engine.core.types import (
        GenerationRequest,
        GenerationSampleSpec,
        OutputBatch,
        WorkloadSignature,
    )
    from vrl.engine.execution.microbatching import MicroBatchPlan
    from vrl.engine.execution.planner import EnginePlan, ExecutionUnit


class PipelineChunkResult(Protocol):
    """Family-specific chunk payload returned before final OutputBatch gather."""


@runtime_checkable
class FamilyPipelineExecutor(Protocol):
    """Family-specific model execution unit."""

    family: str
    task: str

    def workload_signature(
        self,
        request: GenerationRequest,
    ) -> WorkloadSignature: ...

@runtime_checkable
class CapabilityAwareFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Optional extension for engines that expose typed planning capability."""

    def capability(self) -> FamilyCapability: ...

    def plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> EnginePlan: ...


@runtime_checkable
class PlanAwareFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Optional extension for executors that consume the authoritative plan."""

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        plan: EnginePlan,
    ) -> OutputBatch: ...


@runtime_checkable
class PlanAwareBatchedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Optional batched executor extension that receives per-request plans."""

    def forward_batch_plan(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
        engine_plans_by_request: dict[str, EnginePlan],
    ) -> dict[str, OutputBatch]: ...


@runtime_checkable
class ChunkedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Distributed chunk executor that receives its EnginePlan envelope."""

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
        execution_unit: ExecutionUnit,
        plan_summary: Mapping[str, object],
    ) -> PipelineChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch: ...
