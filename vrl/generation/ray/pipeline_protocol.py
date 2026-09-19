"""Ray wire protocol for observing a worker's request-batch progress."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.runtime_errors import TerminalRuntimeError


@dataclass(frozen=True, slots=True)
class RequestBatchProgress:
    """Cross-concurrency-group progress for one request's generation batches.

    ``completed_batches`` counts the contiguous prefix that has finished its
    CPU copy and staging. Enqueuing CUDA work alone does not count. The
    finalizer's merge is a separate operation with its own deadline.
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


__all__ = ["PipelinedProgressError", "RequestBatchProgress"]
