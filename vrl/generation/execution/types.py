"""Types for distributed generation execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, get_args

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import GenerationRequest


class StaleSlotDiscard(Exception):
    """A generation request outlived its worker's trainable-state slot window.

    Raised by the executor when a chunk comes back as a TYPED stale-slot result
    (``ChunkExecutionResult.stale_slot``) — the request's policy version was
    evicted under a non-draining weight sync, NOT a real generation failure. The
    continuous finite-batch producer catches this distinct type and terminates
    the batch with the fixed-version cause instead of retrying it as a collect
    error. See SPRINT_shadow_model_weight_sync.md §3.1 / §4.
    """


@dataclass(frozen=True, slots=True)
class DistributedWorkerHandle:
    """Scheduler-visible identity and actor handle for one generation worker."""

    worker_id: str
    actor: Any | None = None

    def require_installed_policy_version(self, installed: Any, expected: int) -> None:
        """Require this worker's ACK for the expected installed policy version.

        ``installed`` crosses Ray as whatever the worker returned, so it is
        validated as an integer before it is compared.
        """

        try:
            installed_version = int(installed)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"worker {self.worker_id!r} returned invalid installed policy "
                f"version {installed!r}",
            ) from exc
        if installed_version != int(expected):
            raise RuntimeError(
                f"worker {self.worker_id!r} installed policy version {installed_version}, "
                f"expected {int(expected)}",
            )


ChunkPlacementStrategy = Literal["round_robin", "dynamic"]
ParkingBackend = Literal["cpu_only", "cpu_offload", "cumem"]


class QueryableCompletion(Protocol):
    """Non-blocking completion query implemented by device events."""

    def query(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ChunkProduceFence:
    """In-process fence for one chunk's device-side produce completion.

    ``event=None`` represents synchronous CPU execution. CUDA callers publish
    only events that have already been recorded; the Ray worker retains and
    queries them locally rather than putting CUDA objects on the wire.
    """

    completed_chunks: int
    event: QueryableCompletion | None

    def __post_init__(self) -> None:
        if self.completed_chunks < 1:
            raise ValueError("chunk produce fence completed_chunks must be >= 1")

    def query(self) -> bool:
        """Return without synchronizing the device."""

        return self.event is None or self.event.query()


ChunkCompletionCallback: TypeAlias = Callable[[ChunkProduceFence], None]


@dataclass(frozen=True, slots=True)
class WorkerMemoryParkingSnapshot:
    """Physical-memory evidence returned by one worker after parking.

    The baseline is captured before policy load, so an unavoidable CUDA context
    or library footprint is not mistaken for leaked model state. The runtime
    validates every field before handing a shared GPU to the trainer.
    """

    worker_id: str
    backend: ParkingBackend
    baseline_gpu_used_bytes: int
    # display/provenance-only: records the physical footprint before parking so
    # an incomplete-parking error shows what the worker attempted to release.
    loaded_gpu_used_bytes: int
    residual_gpu_used_bytes: int
    residual_bytes_limit: int = 0

    def validate(self) -> None:
        if not self.worker_id:
            raise ValueError("parking snapshot worker_id must be non-empty")
        if self.backend not in get_args(ParkingBackend):
            raise ValueError(f"unsupported parking backend: {self.backend!r}")
        values = {
            "baseline_gpu_used_bytes": self.baseline_gpu_used_bytes,
            "loaded_gpu_used_bytes": self.loaded_gpu_used_bytes,
            "residual_gpu_used_bytes": self.residual_gpu_used_bytes,
            "residual_bytes_limit": self.residual_bytes_limit,
        }
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"parking snapshot {name} must be >= 0")
        allowed = self.baseline_gpu_used_bytes + self.residual_bytes_limit
        if self.residual_gpu_used_bytes > allowed:
            raise RuntimeError(
                f"worker {self.worker_id!r} incomplete {self.backend} memory parking: "
                f"loaded={self.loaded_gpu_used_bytes} residual="
                f"{self.residual_gpu_used_bytes} baseline="
                f"{self.baseline_gpu_used_bytes} limit={self.residual_bytes_limit}",
            )


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


@dataclass(frozen=True, slots=True)
class PipelinedRequestOutOfMemory:
    """Typed worker response that asks the driver to retry through chunk admission.

    A whole-request pipeline can retain two chunks while overlapping teardown and
    compute. If that larger residency OOMs, the worker must first discard its
    partial request state, then return this response so the driver can use the
    normal per-chunk CPU handoff and split-on-OOM path.
    """

    request_id: str
    worker_id: str
    error: str


__all__ = [
    "ChunkCompletionCallback",
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "ChunkPlacementStrategy",
    "ChunkProduceFence",
    "DistributedWorkerHandle",
    "ParkingBackend",
    "PipelinedRequestOutOfMemory",
    "QueryableCompletion",
    "StaleSlotDiscard",
    "WorkerMemoryParkingSnapshot",
]
