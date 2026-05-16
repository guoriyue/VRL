"""Generation pipeline executor protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.engine.core.types import (
        GenerationRequest,
        GenerationSampleRow,
        OutputBatch,
        WorkloadSignature,
    )
    from vrl.engine.execution.microbatching import MicroBatchSample
    from vrl.engine.execution.planner import ExecutionUnit


class PipelineChunkResult(Protocol):
    """Family-specific chunk payload returned before final OutputBatch gather."""


@runtime_checkable
class ChunkGatherer(Protocol):
    """Pure chunk gather contract that does not require an executor/model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch: ...


class RolloutBackend(Protocol):
    """Generation backend consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> OutputBatch: ...


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
class ChunkedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Distributed chunk executor that receives its EnginePlan envelope."""

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: MicroBatchSample,
        execution_unit: ExecutionUnit,
        plan_summary: Mapping[str, object],
    ) -> PipelineChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch: ...
