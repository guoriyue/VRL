"""Chunk placement planning for distributed generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, get_args

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import EnginePlan, build_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkPlacementStrategy,
    DistributedWorkerHandle,
)
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class ChunkPlacementPolicy:
    """How chunks bind to generation workers.

    ``round_robin`` binds at plan time (the historical baseline).
    ``dynamic`` leaves chunks unbound: the dispatch loop pulls the next
    pending chunk onto the first worker with a free slot, submitting by
    ``estimated_cost`` descending (LPT) so a large chunk never anchors the
    tail. With one worker the two strategies are equivalent.
    """

    strategy: ChunkPlacementStrategy = "round_robin"

    def __post_init__(self) -> None:
        allowed = get_args(ChunkPlacementStrategy)
        if self.strategy not in allowed:
            raise ValueError(
                "chunk placement strategy must be one of "
                f"{', '.join(allowed)}; got {self.strategy!r}",
            )


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one generation worker.

    ``worker_id`` is ``None`` under dynamic placement — binding then happens at
    dispatch time in the actor pool, not at plan time. The envelope is the wire
    payload and single source of truth for chunk identity.
    """

    worker_id: str | None
    envelope: ChunkExecutionEnvelope
    estimated_cost: float = 0.0

    @property
    def chunk(self) -> SampleChunk:
        """Return the chunk carried by the authoritative wire envelope."""

        return self.envelope.chunk


@dataclass(frozen=True, slots=True)
class DistributedGenerationPlan:
    """Driver-side plan plus worker placement for one generation request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


@dataclass(frozen=True, slots=True)
class ChunkMemoryReading:
    """Measured CUDA memory footprint of one executed chunk (byte admission L2).

    Produced worker-side around the two memory phases of a diffusion chunk
    (denoise = sustained plateau, decode = short spike) with per-phase
    ``reset_peak_memory_stats``, so peaks are chunk-scoped, not process-lifetime
    high-water marks. Shadow mode only records these next to the estimator's
    prediction; nothing here gates admission yet.
    """

    sample_count: int
    # Whole-chunk latent tensor bytes. Calibration provenance only: the
    # cross-shape fit (peak ~ f(latent bytes)) consumes it offline; the v0
    # same-shape estimator does not read it.
    latent_bytes: int
    # torch.cuda.memory_allocated() right before the denoise loop: weights +
    # encoder outputs + initial latents resident on the worker.
    baseline_allocated_bytes: int
    # Absolute allocated peaks per phase (baseline included, since resident
    # tensors stay allocated through both phases).
    denoise_peak_bytes: int
    decode_peak_bytes: int
    # Device-level occupancy at denoise start, for the budget split
    # total = usable-by-torch + non-torch (CUDA context + other processes).
    reserved_start_bytes: int
    free_start_bytes: int
    total_bytes: int

    @property
    def peak_bytes(self) -> int:
        return max(self.denoise_peak_bytes, self.decode_peak_bytes)

    @property
    def non_torch_bytes(self) -> int:
        """Device bytes torch cannot use: CUDA context + other processes."""

        return max(0, (self.total_bytes - self.free_start_bytes) - self.reserved_start_bytes)

    @property
    def budget_bytes(self) -> int:
        """Bytes torch could occupy on this device: current reservation + free."""

        return self.reserved_start_bytes + self.free_start_bytes

    @classmethod
    def from_metrics(cls, raw: Mapping[str, Any]) -> ChunkMemoryReading | None:
        names = [f.name for f in fields(cls)]
        if any(raw.get(name) is None for name in names):
            return None
        return cls(**{name: int(raw[name]) for name in names})


@dataclass(frozen=True, slots=True)
class AffinePeakFit:
    """Two-point affine fit of chunk peak bytes: peak(n) = intercept + n * slope.

    Memory DEMAND is affine in sample count (latents, activations, trajectory
    buffers all scale per sample); what is NOT affine is the allocator layer
    (segment rounding, fragmentation), which is why the startup probe must
    CONFIRM the fitted candidate with one real run instead of trusting the
    division — same reason vLLM profiles its worst-case shape rather than
    extrapolating (see SPRINT_chunk_size_probe.md).
    """

    slope_bytes_per_sample: float
    intercept_bytes: float

    @classmethod
    def from_trials(
        cls,
        n_low: int,
        peak_low: int,
        n_high: int,
        peak_high: int,
    ) -> AffinePeakFit:
        if n_high <= n_low:
            raise ValueError(f"affine fit needs two distinct n, got {n_low} and {n_high}")
        slope = (peak_high - peak_low) / (n_high - n_low)
        return cls(
            slope_bytes_per_sample=slope,
            intercept_bytes=peak_low - slope * n_low,
        )

    def max_samples_within(self, budget_bytes: int) -> int:
        """Largest n with predicted peak <= budget (0 = not even the intercept fits)."""

        headroom = budget_bytes - self.intercept_bytes
        if headroom < 0:
            return 0
        if self.slope_bytes_per_sample <= 0:
            # A flat/degenerate fit carries no per-sample signal; the caller's
            # ceiling (samples_per_prompt) applies, the confirm run still guards.
            return _FLAT_FIT_UNBOUNDED
        return int(headroom // self.slope_bytes_per_sample)


# Sentinel ceiling for a degenerate flat fit; callers always min() against the
# request ceiling so any large value works. Named to keep call sites readable.
_FLAT_FIT_UNBOUNDED = 1 << 20


def build_chunk_memory_shadow(
    chunk_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Raw per-chunk memory readings for drift monitoring (no estimation).

    One row per executed chunk that carried a ``chunk_memory`` reading; rows
    without one (AR chunks, CPU runs) are skipped, and no reading at all ->
    empty list so callers emit nothing. These rows are the calibration record
    the startup chunk-size probe is checked against: a steady-state peak that
    drifts far from the probe's accepted trial means the probe verdict is
    stale (e.g. the colocated trainer's phase footprint changed).
    """

    rows: list[dict[str, Any]] = []
    for metrics in chunk_metrics:
        raw = metrics.get("chunk_memory")
        if not isinstance(raw, Mapping):
            continue
        reading = ChunkMemoryReading.from_metrics(raw)
        if reading is None:
            continue
        rows.append(
            {
                "chunk_key": metrics.get("chunk_key"),
                "sample_count": reading.sample_count,
                "peak_bytes": reading.peak_bytes,
                "baseline_allocated_bytes": reading.baseline_allocated_bytes,
                "denoise_peak_bytes": reading.denoise_peak_bytes,
                "decode_peak_bytes": reading.decode_peak_bytes,
                "latent_bytes": reading.latent_bytes,
                "non_torch_bytes": reading.non_torch_bytes,
                "budget_bytes": reading.budget_bytes,
            },
        )
    return rows


class DistributedExecutionPlanner:
    """Plan chunk placement across generation workers."""

    def __init__(
        self,
        *,
        policy: ChunkPlacementPolicy | None = None,
    ) -> None:
        self.policy = policy or ChunkPlacementPolicy()

    def plan_with_engine(
        self,
        request: GenerationRequest,
        workers: list[DistributedWorkerHandle],
    ) -> DistributedGenerationPlan:
        if not workers:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        raw_samples = request.sampling.get("samples_per_chunk", request.samples_per_prompt)
        if raw_samples == "auto":
            # "auto" is resolved to an int by the Ray runtime's startup probe
            # before the request reaches planning; seeing it here means the
            # request bypassed that runtime (e.g. a local/direct executor).
            raise ValueError(
                "sampling.samples_per_chunk: auto requires the Ray generation "
                "runtime (startup chunk-size probe); set an explicit int here",
            )
        max_samples = int(raw_samples)
        engine_plan = build_engine_plan(
            request,
            max_samples_per_chunk=max(1, max_samples),
        )
        bind_at_plan_time = self.policy.strategy == "round_robin"
        steps = request.sampling.get("num_steps") or request.sampling.get(
            "max_new_tokens",
        )
        cost_per_sample = max(1, int(steps or 1))
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.chunks):
            worker = workers[idx % len(workers)] if bind_at_plan_time else None
            envelope = ChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker.worker_id if worker else None,
                    envelope=envelope,
                    estimated_cost=float(chunk.sample_count * cost_per_sample),
                ),
            )
        return DistributedGenerationPlan(
            engine_plan=engine_plan,
            assignments=tuple(assignments),
        )


__all__ = [
    "AffinePeakFit",
    "ChunkMemoryReading",
    "ChunkPlacementPolicy",
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "build_chunk_memory_shadow",
]
