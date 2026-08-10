"""Generation runtime and chunk executor protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.bindings.full_sequence_denoise.executor import (
        DiffusionChunkResult,
        DiffusionDenoisedStageOutput,
        DiffusionPreparedStageOutput,
        DiffusionPromptStageInput,
        DiffusionPromptStageOutput,
    )
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

    def is_colocated(self) -> bool:
        """Whether the trainer and rollout share a GPU.

        Still owned by the runtime: this drives whether the driver model is
        offloaded to CPU during rollout, a different axis from worker residency.
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
    ) -> ChunkResult: ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput: ...


@runtime_checkable
class DiffusionStagedChunkExecutor(Protocol):
    """Executor that exposes the five fused diffusion chunk stages.

    ``GenerationWorkerCore.probe_chunk_size`` drives these stage methods directly
    (truncated to a couple of denoise steps) to size ``samples_per_chunk``. The
    Protocol is the ``samples_per_chunk: auto`` diffusion-only gate: a
    non-diffusion executor that lacks the stages fails the ``isinstance`` check
    with a typed error instead of an ``AttributeError`` mid-probe.
    """

    def build_prompt_stage_input(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> DiffusionPromptStageInput: ...

    def run_prompt_encode_stage(
        self,
        payload: DiffusionPromptStageInput,
        *,
        stage_durations: dict[str, float],
        record_function: Any,
    ) -> DiffusionPromptStageOutput: ...

    def run_prepare_stage(
        self,
        payload: DiffusionPromptStageOutput,
        *,
        stage_durations: dict[str, float],
    ) -> DiffusionPreparedStageOutput: ...

    def run_denoise_stage(
        self,
        payload: DiffusionPreparedStageOutput,
        *,
        stage_durations: dict[str, float],
    ) -> DiffusionDenoisedStageOutput: ...

    def run_decode_stage(
        self,
        payload: DiffusionDenoisedStageOutput,
    ) -> DiffusionChunkResult: ...


__all__ = [
    "ChunkGatherer",
    "ChunkResult",
    "DiffusionStagedChunkExecutor",
    "GenerationChunkExecutor",
    "GenerationRuntime",
]
