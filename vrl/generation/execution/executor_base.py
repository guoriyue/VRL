"""Request-level chunk execution shared by every generation binding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.generation.execution.chunks import run_sample_chunks_with_oom_retry
from vrl.generation.protocols import ChunkResult, GenerationChunkExecutor
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)


class ChunkExecutorBase(GenerationChunkExecutor):
    """Drive a planned request through the family chunk step and gather it.

    The three binding bases (full-sequence denoise, chunk-autoregressive
    denoise, token-autoregressive) differ in how ONE chunk is produced, never
    in how a request's chunks are driven or assembled, so that half lives here:

    - ``forward_plan`` is the in-process twin of the Ray dispatch — the same
      ``forward_chunk_plan`` calls with the same OOM-split retry, then the same
      gather — so a local run cannot drift from production.
    - ``gather_chunks`` resolves the gatherer through the family registry,
      which is where the executor -> gatherer binding is already declared
      (``ModelFamilyEntry.gatherer_cls``, also read by the Ray launcher). One
      binding, one construction site: an executor cannot pair itself with a
      gatherer the registry does not know about.

    Families own ``forward_chunk_plan``; overriding ``gather_chunks`` is only
    for payloads that never reach the registry (test doubles).
    """

    family: str

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        plan: Any,
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
        # Lazy: the registry pulls the config schema, and this package stays
        # importable from the neutral execution layer.
        from vrl.families.registry import get_model_family_entry

        gatherer = get_model_family_entry(self.family).new_gatherer()
        return gatherer.gather_chunks(request, sample_rows, chunks)


__all__ = ["ChunkExecutorBase"]
