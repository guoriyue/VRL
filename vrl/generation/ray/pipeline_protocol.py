"""Ray wire protocol for observing single-worker pipeline progress."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.runtime_errors import TerminalRuntimeError


@dataclass(frozen=True, slots=True)
class PipelinedRequestProgress:
    """Cross-concurrency-group progress for one pipelined request.

    ``completed_batches`` is the continuous prefix whose recorded device-side
    produce fences have completed. It never counts host-side CUDA enqueue as
    completion; final teardown and gather retain the last stall window.
    """

    request_id: str
    completed_batches: int
    total_batches: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("pipelined progress request_id must be non-empty")
        if self.total_batches < 1:
            raise ValueError("pipelined progress total_batches must be >= 1")
        if not 0 <= self.completed_batches <= self.total_batches:
            raise ValueError(
                "pipelined progress completed_batches must be between 0 and total_batches",
            )


class PipelinedProgressError(TerminalRuntimeError):
    """The worker's pipelined progress stream violated its wire contract."""


__all__ = ["PipelinedProgressError", "PipelinedRequestProgress"]
