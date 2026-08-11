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

    THE single chunk-width resolution: explicit ``max_samples_per_chunk``
    argument, then the request's ``sampling["samples_per_chunk"]``, then
    ``samples_per_prompt`` (the whole group in one chunk). Every planner —
    Ray placement and in-process alike — goes through this one fallback.
    """

    from vrl.utils.profiling import profile_range

    if max_samples_per_chunk is not None:
        chunk_size = max(1, int(max_samples_per_chunk))
    else:
        raw = request.sampling.get("samples_per_chunk", request.samples_per_prompt)
        if raw == "auto":
            # Resolved to an int by the Ray runtime's startup probe before a
            # request reaches planning; seeing it here means the request
            # bypassed that runtime (e.g. a local/direct executor).
            raise ValueError(
                "sampling.samples_per_chunk: auto requires the Ray generation "
                "runtime (startup chunk-size probe); set an explicit int here",
            )
        chunk_size = max(1, int(raw))
    with profile_range("engine.plan"):
        return EnginePlan(
            chunks=build_prompt_chunks(
                len(request.inputs),
                samples_per_prompt=request.samples_per_prompt,
                max_samples_per_chunk=chunk_size,
            ),
        )


__all__ = ["EnginePlan", "build_engine_plan"]
