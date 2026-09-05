"""Shared score aggregation for fixed-prompt checkpoint evaluations.

Every checkpoint eval answers the same statistical question — did this
checkpoint score better than the base arm on the SAME prompt/seed grid — so
the distribution, the paired-delta bootstrap, and the scores.jsonl/csv writer
belong in one place rather than being re-derived per family. ``score_keys`` and
``base_label`` name report fields rather than the caller, so these stay free
functions (AGENTS.md placement Rule 1).

The bootstrap RNG is seeded from the report schema plus the label and score
key, so a rerun over the same rows reproduces the interval exactly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BOOTSTRAP_RESAMPLES = 2_000


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
) -> tuple[float, float]:
    """Deterministic percentile bootstrap 95% interval for the mean."""

    seed_bytes = hashlib.sha256(f"{schema}\0{label}\0{score_key}".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def summarize_paired_scores(
    rows: Sequence[dict[str, Any]],
    *,
    score_keys: Sequence[str],
    schema: str,
    base_label: str = "base",
    cell_keys: Sequence[str] = ("prompt_index", "sample_index"),
    unpaired_labels: Sequence[str] = (),
) -> dict[str, Any]:
    """Absolute distributions plus per-cell deltas against the base arm.

    The paired statistic is the point of a fixed-prompt eval: comparing
    independent means across arms would drown the effect in prompt difficulty,
    which is exactly the variance a shared prompt/seed grid removes. Every arm
    must therefore cover the identical cell set, and a mismatch is an error
    rather than a silently smaller comparison.

    ``unpaired_labels`` names arms that are calibration anchors rather than
    checkpoints (a ground-truth clip set, for instance): they still get an
    absolute distribution, but they do not share the generated grid and so
    cannot be differenced against it.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint_label"]), []).append(row)
    if base_label not in grouped:
        raise ValueError(f"paired score summary requires {base_label!r} rows")

    absolute = {
        label: {key: distribution([float(row[f"r_{key}"]) for row in group]) for key in score_keys}
        for label, group in grouped.items()
    }

    def cell_of(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(int(row[name]) for name in cell_keys)

    base = {cell_of(row): row for row in grouped[base_label]}
    skip = {base_label, *unpaired_labels}
    paired: dict[str, Any] = {}
    for label, group in grouped.items():
        if label in skip:
            continue
        cells = {cell_of(row): row for row in group}
        if set(cells) != set(base):
            raise ValueError(f"paired score grid differs for {label}")
        paired[label] = {}
        for key in score_keys:
            deltas = [
                float(cells[cell][f"r_{key}"]) - float(base[cell][f"r_{key}"])
                for cell in sorted(base)
            ]
            lower, upper = bootstrap_mean_interval(
                deltas,
                schema=schema,
                label=label,
                score_key=key,
            )
            paired[label][key] = {
                **distribution(deltas),
                "win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
                "tie_rate": sum(delta == 0 for delta in deltas) / len(deltas),
                "bootstrap_95ci": [lower, upper],
                "clear_improvement": lower > 0.0,
                "clear_regression": upper < 0.0,
            }
    return {"absolute": absolute, "paired_delta_from_base": paired}


def write_scores(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Publish scores.jsonl and scores.csv side by side."""

    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "bootstrap_mean_interval",
    "distribution",
    "summarize_paired_scores",
    "write_scores",
]
