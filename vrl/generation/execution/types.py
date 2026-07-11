"""Types for distributed generation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import GenerationRequest


class StaleSlotDiscard(Exception):
    """A generation request outlived its worker's trainable-state slot window.

    Raised by the executor when a chunk comes back as a TYPED stale-slot result
    (``ChunkExecutionResult.stale_slot``) — the request's policy version was
    evicted under a non-draining weight sync, NOT a real generation failure. The
    continuous producer catches this distinct type and counts the group as a
    graceful stale discard (``discarded_stale_count``) instead of a collect error
    (``error_count``). See SPRINT_shadow_model_weight_sync.md §3.1 / §4.
    """


@dataclass(frozen=True, slots=True)
class DistributedWorkerHandle:
    """Scheduler-visible identity and actor handle for one generation worker."""

    worker_id: str
    actor: Any | None = None


@dataclass(frozen=True, slots=True)
class ChunkExecutionEnvelope:
    """Authoritative chunk execution payload sent from the driver to a worker."""

    request: GenerationRequest
    chunk: SampleChunk

    @property
    def chunk_key(self) -> str:
        return self.chunk.chunk_key


@dataclass(slots=True)
class ChunkExecutionResult:
    """Envelope returned by a generation worker for one generation chunk."""

    request_id: str
    worker_id: str
    chunk: SampleChunk
    output: ChunkResult | None
    metrics: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    error: str | None = None
    # Set when the chunk could not run because the worker no longer retains a
    # trainable-state slot for ``request.policy_version`` (the request outlived
    # the slot-retention window under a non-draining weight sync). Distinct from a
    # real generation failure so the caller counts it as a stale discard, not an
    # error — see SPRINT_shadow_model_weight_sync.md.
    stale_slot: bool = False


__all__ = [
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "DistributedWorkerHandle",
    "StaleSlotDiscard",
]
