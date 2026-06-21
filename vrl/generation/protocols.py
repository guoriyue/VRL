"""Generation runtime and chunk executor protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.execution.chunks import SampleChunk
    from vrl.generation.execution.planner import ExecutionStage
    from vrl.generation.types import (
        GenerationOutput,
        GenerationRequest,
        GenerationSampleRow,
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
    """Generation runtime consumed by rollout collectors.

    The runtime is a transport boundary: it generates and, on request, releases
    its actor lease. Whether to release before reward scoring is no longer the
    runtime's decision — that is derived once from GPU topology into the
    ``RayLifecyclePlan`` and read by the collector (see vrl/ray/resources.py).
    """

    current_policy_version: int | None

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...

    async def release(self) -> None:
        """Release the runtime's actor lease when the lifecycle plan asks for it.

        Resident runtimes may implement this as a no-op; on-demand runtimes drop
        their actors and reacquire them on the next generate().
        """
        ...

    def is_colocated(self) -> bool:
        """Whether the trainer and rollout share a GPU.

        Still owned by the runtime: this drives whether the driver model is
        offloaded to CPU during rollout, a different axis from the actor lease.
        """
        ...


@runtime_checkable
class GenerationChunkExecutor(Protocol):
    """Family-specific distributed chunk executor."""

    family: str
    task: str

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: ExecutionStage,
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
