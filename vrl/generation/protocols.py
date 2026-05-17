"""Generation pipeline executor protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.execution.microbatching import MicroBatchSample
    from vrl.generation.execution.planner import ExecutionStage
    from vrl.generation.types import (
        GenerationOutput,
        GenerationRequest,
        GenerationSampleRow,
        WorkloadSignature,
    )


class PipelineChunkResult(Protocol):
    """Family-specific chunk payload returned before final GenerationOutput gather."""


@runtime_checkable
class ChunkGatherer(Protocol):
    """Pure chunk gather contract that does not require an executor/model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> GenerationOutput: ...


class GenerationRuntime(Protocol):
    """Generation runtime consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...


@runtime_checkable
class FamilyPipelineExecutor(Protocol):
    """Family-specific model executor."""

    family: str
    task: str

    def workload_signature(
        self,
        request: GenerationRequest,
    ) -> WorkloadSignature: ...


@runtime_checkable
class ChunkedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Distributed chunk executor that receives its EnginePlan envelope."""

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: MicroBatchSample,
        execution_stage: ExecutionStage,
        plan_summary: Mapping[str, object],
    ) -> PipelineChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> GenerationOutput: ...


__all__ = [
    "ChunkGatherer",
    "ChunkedFamilyPipelineExecutor",
    "FamilyPipelineExecutor",
    "GenerationRuntime",
    "PipelineChunkResult",
]
