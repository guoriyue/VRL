"""Pure score distributions and reproducible bootstrap intervals shared by analyses."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Sequence


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    """Count/mean/median/std/stderr/min/max for one score column."""

    if not values:
        raise ValueError("cannot summarize an empty score distribution")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    schema: str,
    label: str,
    score_key: str,
    resamples: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap 95% interval for the mean."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be non-empty and finite")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap resamples must be a positive integer")
    seed_text = f"{schema}\0{label}\0{score_key}"
    if seed is not None:
        seed_text = f"{seed}\0{seed_text}"
    seed_bytes = hashlib.sha256(seed_text.encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(resamples)
    ]
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]
