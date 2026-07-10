"""Request-level sample chunk planning."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.generation.execution.chunks import SampleChunk, build_prompt_chunks
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Public execution-plan envelope shared by direct and Ray runtimes."""

    chunks: tuple[SampleChunk, ...]


@dataclass(frozen=True, slots=True)
class EnginePlanner:
    """Split one generation request into prompt-major sample chunks."""

    request: GenerationRequest
    max_samples_per_chunk: int | None = None

    def build(self) -> EnginePlan:
        """Build the immutable execution plan."""

        from vrl.utils.profiling import record_function

        with record_function("engine.plan"):
            return EnginePlan(
                chunks=build_prompt_chunks(
                    self.request.prompts,
                    samples_per_prompt=self.request.samples_per_prompt,
                    max_samples_per_chunk=self._chunk_size(),
                ),
            )

    def _chunk_size(self) -> int:
        if self.max_samples_per_chunk is not None:
            return max(1, int(self.max_samples_per_chunk))
        return max(
            1,
            int(
                self.request.sampling.get(
                    "samples_per_chunk",
                    self.request.samples_per_prompt,
                ),
            ),
        )


def build_engine_plan(
    request: GenerationRequest,
    *,
    max_samples_per_chunk: int | None = None,
) -> EnginePlan:
    """Build the chunk plan consumed by direct and distributed executors."""

    return EnginePlanner(
        request=request,
        max_samples_per_chunk=max_samples_per_chunk,
    ).build()


__all__ = ["EnginePlan", "EnginePlanner", "build_engine_plan"]
