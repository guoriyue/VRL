"""Helpers for gathering chunked generation executor outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from vrl.engine.core.protocols import ChunkedFamilyPipelineExecutor, PipelineChunkResult
from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleRow,
    OutputBatch,
)


@runtime_checkable
class ChunkGatherer(Protocol):
    """Pure chunk gather contract that does not require an executor/model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch: ...


def require_chunked_executor(executor: Any) -> ChunkedFamilyPipelineExecutor:
    """Return executor if it exposes the distributed chunk contract."""

    forward_chunk_plan = getattr(executor, "forward_chunk_plan", None)
    gather_chunks = getattr(executor, "gather_chunks", None)
    if not callable(forward_chunk_plan) or not callable(gather_chunks):
        raise TypeError(
            f"{type(executor).__name__} does not implement "
            "forward_chunk_plan(...) and gather_chunks(...)",
        )
    return executor


def require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    """Return gatherer if it exposes the pure chunk gather contract."""

    gather_chunks = getattr(gatherer, "gather_chunks", None)
    if not callable(gather_chunks):
        raise TypeError(
            f"{type(gatherer).__name__} does not implement gather_chunks(...)",
        )
    return gatherer


def gather_pipeline_chunks(
    gatherer: Any,
    request: GenerationRequest,
    sample_rows: Sequence[GenerationSampleRow],
    chunks: Sequence[PipelineChunkResult],
) -> OutputBatch:
    """Gather family-specific chunk payloads into one canonical OutputBatch."""

    chunk_gatherer = require_chunk_gatherer(gatherer)
    return chunk_gatherer.gather_chunks(request, sample_rows, chunks)


__all__ = [
    "ChunkGatherer",
    "gather_pipeline_chunks",
    "require_chunk_gatherer",
    "require_chunked_executor",
]
