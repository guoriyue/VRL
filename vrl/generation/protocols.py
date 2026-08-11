"""Generation runtime and chunk executor protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.execution.chunks import SampleChunk
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
class GenerationRuntime(Protocol):
    """Generation runtime consumed by rollout collectors.

    The runtime is a transport boundary: schedules explicitly activate it before
    generation and offload it at a shared-GPU handoff. Whether to offload before
    reward scoring is no longer the runtime's decision — that is derived once
    from GPU topology into the
    ``RayLifecyclePlan`` and read by the collector (see vrl/ray/resources.py).
    """

    current_policy_version: int | None

    @property
    def requires_driver_model_offload(self) -> bool:
        """Whether shared ownership requires parking trainer state for generation."""
        ...

    async def preflight(self) -> None:
        """Fail fast on an unhealthy engine before the schedule starts.

        The generation twin of ``RewardRuntime.preflight``: a live fleet must
        answer one bounded health probe per worker; deferred sessions have
        nothing to probe yet (their launch validates itself).
        """
        ...

    async def activate(self) -> None:
        """Make generation ready and complete any policy install staged while inactive."""
        ...

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...

    async def offload(self) -> None:
        """Yield GPU memory after the schedule has drained generation.

        Resident runtimes may implement this as a no-op. Shared-GPU runtimes park
        physical GPU memory at a handoff and restore it during the next activate;
        terminal actor destruction remains the responsibility of shutdown().
        """
        ...

    async def shutdown(self) -> None:
        """Close admission and release every runtime-owned worker/resource."""
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
    ) -> ChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput: ...


@runtime_checkable
class DiffusionChunkProbeExecutor(Protocol):
    """Diffusion executor capability used by automatic chunk-size sizing."""

    def forward_probe_chunk(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        *,
        execute_steps: int,
    ) -> ChunkResult: ...


__all__ = [
    "ChunkGatherer",
    "ChunkResult",
    "DiffusionChunkProbeExecutor",
    "GenerationChunkExecutor",
    "GenerationRuntime",
]
