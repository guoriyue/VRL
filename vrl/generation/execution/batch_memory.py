"""CUDA occupancy capture and the affine model used for startup batch sizing.

BatchMemoryReading is the worker wire record in types.py. This module owns
pre-loop occupancy sampling and fitting; the executor logs completed readings
without an intermediate telemetry representation.
"""

from __future__ import annotations

from dataclasses import dataclass


def cuda_occupancy_snapshot() -> dict[str, int] | None:
    """Device occupancy at batch start, or None off CUDA.

    This is the half of a :class:`BatchMemoryReading` that can only be measured
    before the denoise loop starts; the executor completes the record with the
    two per-phase peaks and the sample count, and ``from_metrics``
    reassembles it. The keys belong to the batch-memory wire record, while the
    values must be captured before the loop changes allocator occupancy.
    """

    import torch

    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "baseline_allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_start_bytes": int(torch.cuda.memory_reserved()),
        "free_start_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


@dataclass(frozen=True, slots=True)
class AffinePeakFit:
    """Two-point affine fit of batch peak bytes: peak(n) = intercept + n * slope.

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

    def max_samples_within(self, budget_bytes: int, *, max_samples: int) -> int:
        """Bound the fitted capacity by the request's actual sample ceiling.

        Return zero when the intercept exceeds the budget. A non-growing fit
        cannot predict a memory ceiling, so use the request ceiling and let the
        worker's real confirmation trial determine whether it fits.
        """
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        headroom = budget_bytes - self.intercept_bytes
        if headroom < 0:
            return 0
        if self.slope_bytes_per_sample <= 0:
            return max_samples
        return min(int(headroom // self.slope_bytes_per_sample), max_samples)


__all__ = ["AffinePeakFit", "cuda_occupancy_snapshot"]
