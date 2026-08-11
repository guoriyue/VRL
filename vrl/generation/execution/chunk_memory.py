"""Chunk memory sizing: measure, model, and monitor per-chunk CUDA peaks.

One topic across three moments of a chunk's life, split out of
``chunk_placement`` because sizing and placement only ever shared consumers,
never meaning: ``cuda_occupancy_snapshot`` captures the pre-loop half of a
``ChunkMemoryReading`` at chunk start (called by the regime-neutral denoise
loop), ``AffinePeakFit`` turns two probe trials into the startup chunk-width
proposal (the worker's ``samples_per_chunk: auto`` probe), and
``build_chunk_memory_shadow`` flattens executed-chunk readings into the
drift-calibration rows driver telemetry checks against the probe's verdict.
The reading dataclass itself stays in ``types.py`` — it rides the Ray wire
inside chunk results; this module owns producing and interpreting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from vrl.generation.execution.types import (
    ChunkExecutionResult,
    ChunkMemoryReading,
)


def cuda_occupancy_snapshot() -> dict[str, int] | None:
    """Device occupancy at chunk start, or None off CUDA.

    This is the half of a :class:`ChunkMemoryReading` that can only be measured
    before the denoise loop starts; the executor completes the record with the
    two per-phase peaks and the sample count, and ``from_metrics``
    reassembles it. It lives beside the fit/shadow consumers rather than in the
    denoise loop because the key names are the byte-admission telemetry
    contract, not general CUDA semantics — and they are checked against the
    dataclass so a renamed field fails here instead of silently producing a
    reading that never reassembles.
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


__all__ = [
    "AffinePeakFit",
    "build_chunk_memory_shadow",
    "cuda_occupancy_snapshot",
]
