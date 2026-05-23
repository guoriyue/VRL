"""Generation pipeline executor protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.execution.chunks import SampleChunk
    from vrl.generation.execution.planner import ExecutionStage
    from vrl.generation.types import (
        GenerationOutput,
        GenerationRequest,
        GenerationSampleRow,
        WorkloadSignature,
    )


ChunkResult = Any


@runtime_checkable
class ChunkGatherer(Protocol):
    """Pure chunk gather contract that does not require an executor/model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput: ...


class GenerationRuntime(Protocol):
    """Generation runtime consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...


@runtime_checkable
class PipelineExecutor(Protocol):
    """Family-specific distributed chunk executor."""

    family: str
    task: str

    def workload_signature(
        self,
        request: GenerationRequest,
    ) -> WorkloadSignature: ...

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: ExecutionStage,
        plan_summary: Mapping[str, object],
    ) -> ChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput: ...


__all__ = [
    "ChunkGatherer",
    "ChunkResult",
    "GenerationRuntime",
    "PipelineExecutor",
]
