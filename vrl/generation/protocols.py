"""Generation runtime and chunk executor protocols."""

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


@runtime_checkable
class PolicyVersionProvider(Protocol):
    """Anything that can report the policy version its weights correspond to.

    Orchestration asks providers (collector runtime, weight syncer) for the
    version through this contract instead of reaching into their internals.
    ``None`` means "I do not track a version".
    """

    current_policy_version: int | None


class GenerationRuntime(Protocol):
    """Generation runtime consumed by rollout collectors."""

    current_policy_version: int | None

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...

    def should_release_memory_before_reward(self) -> bool:
        """Whether rollout memory must be dropped before reward-model scoring.

        Owned by the runtime because only it knows its release mode and GPU
        sharing; callers must not probe runtime internals for this decision.
        """
        ...

    def is_colocated(self) -> bool:
        """Whether the trainer and rollout share a GPU.

        Owned by the runtime for the same reason as the release decision.
        """
        ...


@runtime_checkable
class GenerationChunkExecutor(Protocol):
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
    "GenerationChunkExecutor",
    "GenerationRuntime",
    "PolicyVersionProvider",
]
