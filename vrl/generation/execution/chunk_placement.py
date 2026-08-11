"""Chunk placement planning for distributed generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, get_args

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import EnginePlan, build_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    ChunkMemoryReading,
    ChunkPlacementStrategy,
)
from vrl.generation.types import GenerationRequest


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


def cuda_occupancy_snapshot() -> dict[str, int] | None:
    """Device occupancy at chunk start, or None off CUDA.

    This is the half of a :class:`ChunkMemoryReading` that can only be measured
    before the denoise loop starts; the executor completes the record with the
    two per-phase peaks and the sample count, and ``from_metrics``
    reassembles it. It lives beside that dataclass rather than in the denoise
    loop because the key names are the byte-admission telemetry contract, not
    general CUDA semantics — and they are checked against the dataclass so a
    renamed field fails here instead of silently producing a reading that never
    reassembles.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        snapshot = {
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_start_bytes": int(torch.cuda.memory_reserved()),
            "free_start_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    except Exception:
        return None
    unknown = snapshot.keys() - {f.name for f in fields(ChunkMemoryReading)}
    if unknown:
        raise ValueError(
            f"chunk occupancy keys are not ChunkMemoryReading fields: {sorted(unknown)}",
        )
    return snapshot


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
    chunk_results: Sequence[ChunkExecutionResult],
) -> list[dict[str, Any]]:
    """Raw per-chunk memory readings for drift monitoring (no estimation).

    One row per executed chunk that carried a typed memory reading; rows
    without one (AR chunks, CPU runs) are skipped, and no reading at all ->
    empty list so callers emit nothing. These rows are the calibration record
    the startup chunk-size probe is checked against: a steady-state peak that
    drifts far from the probe's accepted trial means the probe verdict is
    stale (e.g. the colocated trainer's phase footprint changed).
    """

    rows: list[dict[str, Any]] = []
    for result in chunk_results:
        reading = result.memory
        if reading is None:
            continue
        rows.append(
            {
                "chunk_key": result.chunk.chunk_key,
                "sample_count": reading.sample_count,
                "peak_bytes": reading.peak_bytes,
                "baseline_allocated_bytes": reading.baseline_allocated_bytes,
                "denoise_peak_bytes": reading.denoise_peak_bytes,
                "decode_peak_bytes": reading.decode_peak_bytes,
                "non_torch_bytes": reading.non_torch_bytes,
                "budget_bytes": reading.budget_bytes,
            },
        )
    return rows


class DistributedExecutionPlanner:
    """Plan chunk placement across generation workers.

    ``round_robin`` binds at plan time. ``dynamic`` leaves chunks unbound so
    the dispatch loop can pull the highest-cost pending chunk onto a free worker.
    """

    def __init__(
        self,
        *,
        strategy: ChunkPlacementStrategy = "round_robin",
    ) -> None:
        allowed = get_args(ChunkPlacementStrategy)
        if strategy not in allowed:
            raise ValueError(
                f"chunk placement strategy must be one of {', '.join(allowed)}; got {strategy!r}",
            )
        self.strategy = strategy

    def plan_with_engine(
        self,
        request: GenerationRequest,
        worker_ids: Sequence[str],
    ) -> DistributedGenerationPlan:
        worker_ids = tuple(worker_ids)
        if not worker_ids:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        if any(not worker_id for worker_id in worker_ids):
            raise ValueError("DistributedExecutionPlanner worker IDs must be non-empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("DistributedExecutionPlanner worker IDs must be unique")
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
        bind_at_plan_time = self.strategy == "round_robin"
        steps = request.sampling.get("num_steps") or request.sampling.get(
            "max_new_tokens",
        )
        cost_per_sample = max(1, int(steps or 1))
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.chunks):
            worker_id = worker_ids[idx % len(worker_ids)] if bind_at_plan_time else None
            envelope = ChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker_id,
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
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "build_chunk_memory_shadow",
    "cuda_occupancy_snapshot",
]
