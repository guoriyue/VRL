"""Ray wire protocol for observing single-worker pipeline progress."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.runtime_errors import TerminalRuntimeError


@dataclass(frozen=True, slots=True)
class PipelinedRequestProgress:
    """Cross-concurrency-group progress for one pipelined request.

    ``completed_chunks`` is the continuous prefix whose recorded device-side
    produce fences have completed. It never counts host-side CUDA enqueue as
    completion; final teardown and gather retain the last stall window.
    """

    request_id: str
    completed_chunks: int
    total_chunks: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("pipelined progress request_id must be non-empty")
        if self.total_chunks < 1:
            raise ValueError("pipelined progress total_chunks must be >= 1")
        if not 0 <= self.completed_chunks <= self.total_chunks:
            raise ValueError(
                "pipelined progress completed_chunks must be between 0 and total_chunks",
            )


class PipelinedProgressError(TerminalRuntimeError):
    """The worker's pipelined progress stream violated its wire contract."""


__all__ = ["PipelinedProgressError", "PipelinedRequestProgress"]
