"""Request-level sample chunk planning."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.generation.execution.chunks import SampleChunk, build_prompt_chunks
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Public execution-plan envelope shared by direct and Ray runtimes."""

    chunks: tuple[SampleChunk, ...]


def build_engine_plan(
    request: GenerationRequest,
    *,
    max_samples_per_chunk: int | None = None,
) -> EnginePlan:
    """Build the chunk plan consumed by direct and distributed executors.

    Chunk size precedence: explicit ``max_samples_per_chunk`` argument, then
    the request's ``sampling["samples_per_chunk"]``, then ``samples_per_prompt``.
    """

    from vrl.utils.profiling import profile_range

    if max_samples_per_chunk is not None:
        chunk_size = max(1, int(max_samples_per_chunk))
    else:
        chunk_size = max(
            1,
            int(
                request.sampling.get(
                    "samples_per_chunk",
                    request.samples_per_prompt,
                ),
            ),
        )
    with profile_range("engine.plan"):
        return EnginePlan(
            chunks=build_prompt_chunks(
                request.prompts,
                samples_per_prompt=request.samples_per_prompt,
                max_samples_per_chunk=chunk_size,
            ),
        )


__all__ = ["EnginePlan", "build_engine_plan"]
