"""Request-level chunk execution shared by every generation binding."""

from __future__ import annotations

from collections.abc import Sequence

from vrl.generation.execution.chunks import run_sample_chunks_with_oom_retry
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.protocols import ChunkGatherer, ChunkResult
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)


class ChunkExecutorBase:
    """Drive a planned request through the family chunk step and gather it.

    The three binding bases (full-sequence denoise, chunk-autoregressive
    denoise, token-autoregressive) differ in how ONE chunk is produced, never
    in how a request's chunks are driven or assembled, so that half lives here:

    - ``forward_plan`` is the in-process request entry (local tools, family
      tests, single-process e2e): the same ``forward_chunk_plan`` chunk step
      and the same gather as the Ray dispatch, with a local OOM-split retry.
      Production drives chunks through the Ray dispatcher instead; planning is
      shared via ``build_engine_plan``'s single width fallback.
    - ``gather_chunks`` delegates to the gatherer injected by the composition
      root that already owns the family registry entry. The neutral execution
      layer never looks family identity up again.

    Families own ``forward_chunk_plan``; overriding ``gather_chunks`` is only
    for payloads that never reach the registry (test doubles).
    """

    family: str

    def __init__(self, *, gatherer: ChunkGatherer | None = None) -> None:
        self._gatherer = gatherer

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        plan: EnginePlan,
    ) -> GenerationOutput:
        chunks = run_sample_chunks_with_oom_retry(
            plan.chunks,
            lambda chunk: self.forward_chunk_plan(request, chunk),
        )
        return self.gather_chunks(request, list(sample_rows), chunks)

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput:
        if self._gatherer is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an injected chunk gatherer "
                "for request-level execution",
            )
        return self._gatherer.gather_chunks(request, sample_rows, chunks)


__all__ = ["ChunkExecutorBase"]
