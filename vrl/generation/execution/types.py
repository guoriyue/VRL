"""Types for distributed generation execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Protocol, TypeAlias, get_args

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import GenerationRequest
from vrl.utils.cuda_memory import validate_parking_residual


class StaleSlotDiscard(Exception):
    """A generation request outlived its worker's trainable-state slot window.

    Raised by the executor when a chunk comes back as a TYPED stale-slot result
    (``ChunkExecutionResult.stale_slot``) — the request's policy version was
    evicted under a non-draining weight sync, NOT a real generation failure. The
    continuous finite-batch producer catches this distinct type and terminates
    the batch with the fixed-version cause instead of retrying it as a collect
    error. See SPRINT_shadow_model_weight_sync.md §3.1 / §4.
    """


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


# Keep the exported alias as its historical runtime value; a ``type`` statement
# would replace it with a TypeAliasType and needlessly change public introspection.
ChunkCompletionCallback: TypeAlias = Callable[[ChunkProduceFence], None]  # noqa: UP040


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
        validate_parking_residual(
            residual_bytes=self.residual_gpu_used_bytes,
            baseline_bytes=self.baseline_gpu_used_bytes,
            limit_bytes=self.residual_bytes_limit,
            context=(
                f"{self.backend} memory parking (worker {self.worker_id!r}, "
                f"loaded={self.loaded_gpu_used_bytes})"
            ),
        )


@dataclass(frozen=True, slots=True)
class ChunkMemoryReading:
    """Measured CUDA memory footprint of one executed generation chunk."""

    # display/provenance-only: identifies the measured chunk width in runtime
    # debug telemetry; the source chunk is no longer available after aggregation.
    sample_count: int
    # display/provenance-only: captures the unrepeatable allocator baseline used
    # to interpret phase peaks in runtime debug telemetry.
    baseline_allocated_bytes: int
    denoise_peak_bytes: int
    decode_peak_bytes: int
    reserved_start_bytes: int
    free_start_bytes: int
    total_bytes: int

    @property
    def peak_bytes(self) -> int:
        return max(self.denoise_peak_bytes, self.decode_peak_bytes)

    @property
    def non_torch_bytes(self) -> int:
        """Device bytes torch cannot use: CUDA context plus other processes."""

        return max(0, (self.total_bytes - self.free_start_bytes) - self.reserved_start_bytes)

    @property
    def budget_bytes(self) -> int:
        """Bytes torch could occupy on this device."""

        return self.reserved_start_bytes + self.free_start_bytes

    @classmethod
    def from_metrics(cls, raw: Mapping[str, Any]) -> ChunkMemoryReading | None:
        """Normalize the binding-owned memory mapping once at the worker boundary."""

        names = tuple(item.name for item in fields(cls))
        if any(raw.get(name) is None for name in names):
            return None
        return cls(**{name: int(raw[name]) for name in names})


@dataclass(frozen=True, slots=True)
class ChunkSizeProbeTrial:
    """One worker-local probe verdict carried back across Ray."""

    n: int
    oom: bool
    # display/provenance-only: names which adaptive-probe branch produced the
    # trial so startup diagnostics remain interpretable.
    label: str
    peak_bytes: int | None = None
    non_torch_bytes: int | None = None
    wall_s: float | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("chunk-size probe trial n must be >= 1")
        if not self.label:
            raise ValueError("chunk-size probe trial label must be non-empty")
        measurements = (self.peak_bytes, self.non_torch_bytes, self.wall_s)
        if self.oom:
            if any(value is not None for value in measurements):
                raise ValueError("OOM chunk-size probe trials cannot carry measurements")
            return
        if any(value is None for value in measurements):
            raise ValueError("successful chunk-size probe trials require all measurements")
        assert self.peak_bytes is not None
        assert self.non_torch_bytes is not None
        assert self.wall_s is not None
        if self.peak_bytes < 0 or self.non_torch_bytes < 0 or self.wall_s < 0:
            raise ValueError("chunk-size probe trial measurements must be >= 0")

    @property
    def per_sample_s(self) -> float | None:
        """Derive throughput without duplicating it in the wire payload."""

        return None if self.wall_s is None else self.wall_s / self.n


@dataclass(frozen=True, slots=True)
class ChunkSizeProbeResult:
    """One worker's resolved chunk size and its probe provenance."""

    samples_per_chunk: int
    # display/provenance-only: the device budget observed during startup sizing.
    budget_bytes: int
    # display/provenance-only: the irreproducible measurements behind the chosen
    # size, retained for startup diagnostics.
    trials: tuple[ChunkSizeProbeTrial, ...]

    def __post_init__(self) -> None:
        if self.samples_per_chunk < 1:
            raise ValueError("probed samples_per_chunk must be >= 1")
        if self.budget_bytes < 0:
            raise ValueError("chunk-size probe budget_bytes must be >= 0")
        trials = tuple(self.trials)
        if any(not isinstance(trial, ChunkSizeProbeTrial) for trial in trials):
            raise TypeError("chunk-size probe trials must contain ChunkSizeProbeTrial")
        object.__setattr__(self, "trials", trials)


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
    # display/provenance-only: worker-local measurements folded into optional
    # runtime-debug telemetry after chunk execution.
    memory: ChunkMemoryReading | None = None
    # display/provenance-only: family/worker diagnostics requested explicitly by
    # GenerationRequest.runtime_debug.
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
    "ChunkMemoryReading",
    "ChunkPlacementStrategy",
    "ChunkProduceFence",
    "ChunkSizeProbeResult",
    "ChunkSizeProbeTrial",
    "ParkingBackend",
    "PipelinedRequestOutOfMemory",
    "QueryableCompletion",
    "StaleSlotDiscard",
    "WorkerMemoryParkingSnapshot",
]
