"""Model-independent score summaries and fixed-prompt checkpoint curves.

Every checkpoint eval answers the same statistical question — did this
checkpoint score better than the base arm on the SAME prompt/seed grid — so
the distribution, the paired-delta bootstrap, and the scores.jsonl/csv writer
belong in one place rather than being re-derived per family. ``score_keys`` and
``base_label`` name report fields rather than the caller, so these stay free
functions (AGENTS.md placement Rule 1).

The bootstrap RNG is seeded from the report schema plus the label and score
key, so a rerun over the same rows reproduces the interval exactly.

``write_curve_report`` averages repeated samples within each prompt before
bootstrapping: prompts, not correlated seeds of one prompt, are independent
units. The older ``summarize_paired_scores`` API deliberately retains its
per-cell video-report semantics. Neither entrypoint loads a generation or
reward model. Run ``python -m vrl.scripts.eval.score_report --help`` to report
existing JSONL scores with checkpoint_label, epoch, prompt_index, sample_index,
seed, prompt, and r_<component> columns. These are generic numeric measurements:
a clear increase is not a claim of improved quality or that higher is better.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any


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


def write_curve_report(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    score_keys: Sequence[str] | None = None,
    base_label: str = "base",
    tie_epsilon: float = 0.0,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate a paired grid and publish summary.json, curve.csv, and curve.png.

    Each arm must contain the same unique prompt/sample cells with matching
    prompts and seeds. Epochs belong to arms and cannot change within an arm.
    Both absolute means and paired differences give every prompt equal weight,
    even if prompts have different numbers of sampled images or videos.
    ``r_*`` columns may contain diagnostics as well as rewards; directional
    findings describe increases/decreases, never whether either is desirable.
    """

    if not rows:
        raise ValueError("curve report requires score rows")
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("tie_epsilon must be finite and non-negative")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    keys = (
        tuple(score_keys)
        if score_keys is not None
        else tuple(sorted({key[2:] for row in rows for key in row if key.startswith("r_")}))
    )
    if not keys or len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("score_keys must be non-empty and unique")

    grouped: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    epochs: dict[str, int] = {}
    prompts: dict[int, str] = {}
    required = {"checkpoint_label", "epoch", "prompt_index", "sample_index", "seed", "prompt"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"score row is missing fields: {sorted(missing)}")
        label = row["checkpoint_label"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("checkpoint_label must be a non-empty string")
        for field in ("epoch", "prompt_index", "sample_index", "seed"):
            if type(row[field]) is not int:
                raise ValueError(f"{label}: {field} must be an integer")
        if row["prompt_index"] < 0 or row["sample_index"] < 0:
            raise ValueError(f"{label}: prompt/sample indices must be non-negative")
        if not isinstance(row["prompt"], str):
            raise ValueError(f"{label}: prompt must be a string")
        epoch = row["epoch"]
        if label in epochs and epochs[label] != epoch:
            raise ValueError(f"conflicting epochs for {label}")
        epochs[label] = epoch
        cell = (row["prompt_index"], row["sample_index"])
        cells = grouped.setdefault(label, {})
        if cell in cells:
            raise ValueError(f"duplicate score cell for {label}: {cell}")
        if cell[0] in prompts and prompts[cell[0]] != row["prompt"]:
            raise ValueError(f"prompt mismatch at prompt_index={cell[0]}")
        prompts[cell[0]] = row["prompt"]
        for key in keys:
            try:
                value = float(row[f"r_{key}"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"missing or invalid r_{key} for {label}: {cell}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite r_{key} for {label}: {cell}")
        cells[cell] = row
    if base_label not in grouped:
        raise ValueError(f"paired curve report requires {base_label!r} rows")
    base = grouped[base_label]
    for label, cells in grouped.items():
        if cells.keys() != base.keys():
            raise ValueError(f"paired score grid differs for {label}")
        for cell, row in cells.items():
            if row["seed"] != base[cell]["seed"]:
                raise ValueError(f"seed mismatch for {label}: {cell}")

    # Collapse seeds before bootstrapping; otherwise repeated observations of
    # one prompt would make uncertainty look artificially small.
    prompt_means = {}
    for label, cells in grouped.items():
        prompt_means[label] = {
            key: {
                prompt_index: statistics.fmean(
                    float(row[f"r_{key}"])
                    for cell, row in cells.items()
                    if cell[0] == prompt_index
                )
                for prompt_index in sorted(prompts)
            }
            for key in keys
        }
    schema = "vrl.score_curve/v1"
    arms: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    for label in sorted(grouped, key=lambda name: (epochs[name], name != base_label, name)):
        scores = {}
        for key in keys:
            values = list(prompt_means[label][key].values())
            deltas = [
                value - prompt_means[base_label][key][index]
                for index, value in prompt_means[label][key].items()
            ]
            absolute = distribution(values)
            absolute["bootstrap_95ci"] = list(
                bootstrap_mean_interval(
                    values,
                    schema=schema,
                    label=label,
                    score_key=f"absolute:{key}",
                    resamples=bootstrap_resamples,
                    seed=seed,
                ),
            )
            lower, upper = bootstrap_mean_interval(
                deltas,
                schema=schema,
                label=label,
                score_key=f"delta:{key}",
                resamples=bootstrap_resamples,
                seed=seed,
            )
            paired = {
                **distribution(deltas),
                "bootstrap_95ci": [lower, upper],
                "win_rate": sum(value > tie_epsilon for value in deltas) / len(deltas),
                "tie_rate": sum(abs(value) <= tie_epsilon for value in deltas) / len(deltas),
                "clear_increase": lower > tie_epsilon,
                "clear_decrease": upper < -tie_epsilon,
            }
            scores[key] = {"absolute": absolute, "paired_delta_from_base": paired}
            curve_rows.append(
                {
                    "checkpoint_label": label,
                    "epoch": epochs[label],
                    "score_key": key,
                    "prompt_count": len(prompts),
                    "sample_count": len(grouped[label]),
                    "mean": absolute["mean"],
                    "ci_low": absolute["bootstrap_95ci"][0],
                    "ci_high": absolute["bootstrap_95ci"][1],
                    "paired_delta": paired["mean"],
                    "paired_ci_low": lower,
                    "paired_ci_high": upper,
                    "win_rate": paired["win_rate"],
                    "tie_rate": paired["tie_rate"],
                },
            )
        arms[label] = {
            "epoch": epochs[label],
            "prompt_count": len(prompts),
            "sample_count": len(grouped[label]),
            "scores": scores,
        }
    summary = {
        "schema": schema,
        "base_label": base_label,
        "statistical_unit": "prompt",
        "tie_epsilon": tie_epsilon,
        "bootstrap": {"resamples": bootstrap_resamples, "seed": seed},
        "arms": arms,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_curve_plot(curve_rows, keys, output_dir / "curve.png")
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_curve_plot(rows: list[dict[str, Any]], score_keys: Sequence[str], path: Path) -> None:
    """Draw reward and paired-delta panels using the existing core Pillow dependency."""

    from PIL import Image, ImageDraw, ImageFont

    panel_width, panel_height = 560, 280
    canvas = Image.new("RGB", (panel_width * 2, panel_height * len(score_keys)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for key_index, key in enumerate(score_keys):
        points = [row for row in rows if row["score_key"] == key]
        epochs = [row["epoch"] for row in points]
        x_min, x_max = min(epochs), max(epochs)
        for column, (value_key, low_key, high_key, title) in enumerate(
            (
                ("mean", "ci_low", "ci_high", "Absolute score"),
                ("paired_delta", "paired_ci_low", "paired_ci_high", "Paired delta from base"),
            ),
        ):
            left, right = column * panel_width + 70, (column + 1) * panel_width - 25
            top, bottom = key_index * panel_height + 45, (key_index + 1) * panel_height - 60
            bounds = [row[name] for row in points for name in (low_key, high_key)]
            if column:
                bounds.append(0.0)
            y_min, y_max = min(bounds), max(bounds)
            margin = max((y_max - y_min) * 0.12, 0.01)
            y_min, y_max = y_min - margin, y_max + margin
            draw.text((left, top - 30), f"{key}: {title} (95% CI)", fill="black", font=font)
            draw.line((left, top, left, bottom, right, bottom), fill="black", width=1)
            for tick in range(5):
                value = y_min + (y_max - y_min) * tick / 4
                y = bottom - (bottom - top) * tick / 4
                draw.line((left, y, right, y), fill="#e4e4e4")
                draw.text((left - 65, y - 7), f"{value:.3g}", fill="black", font=font)
            coordinates = []
            for row in points:
                x = left + (right - left) * (
                    (row["epoch"] - x_min) / (x_max - x_min) if x_max != x_min else 0.5
                )
                y = bottom - (row[value_key] - y_min) / (y_max - y_min) * (bottom - top)
                lo = bottom - (row[low_key] - y_min) / (y_max - y_min) * (bottom - top)
                hi = bottom - (row[high_key] - y_min) / (y_max - y_min) * (bottom - top)
                coordinates.append((x, y))
                draw.line((x, lo, x, hi), fill="#2471a3", width=2)
                draw.line((x - 4, lo, x + 4, lo), fill="#2471a3", width=2)
                draw.line((x - 4, hi, x + 4, hi), fill="#2471a3", width=2)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#2471a3")
            if len(coordinates) > 1:
                draw.line(coordinates, fill="#2471a3", width=2)
            # A fixed number of ticks keeps a long checkpoint sweep readable;
            # exact labels and every epoch remain in curve.csv/summary.json.
            tick_epochs = sorted(set(epochs))
            if len(tick_epochs) > 6:
                tick_epochs = [
                    tick_epochs[index * (len(tick_epochs) - 1) // 5] for index in range(6)
                ]
            for epoch in tick_epochs:
                x = left + (right - left) * (
                    (epoch - x_min) / (x_max - x_min) if x_max != x_min else 0.5
                )
                draw.text((x, bottom + 8), str(epoch), fill="black", font=font, anchor="mt")
            draw.text(
                ((left + right) / 2, bottom + 28), "Epoch", fill="black", font=font, anchor="mt"
            )
    canvas.save(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Existing score rows in JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-key", action="append", dest="score_keys")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    with args.scores.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary = write_curve_report(
        rows,
        args.output_dir,
        score_keys=args.score_keys,
        base_label=args.base_label,
        tie_epsilon=args.tie_epsilon,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "bootstrap_mean_interval",
    "distribution",
    "summarize_paired_scores",
    "write_curve_report",
    "write_scores",
]


if __name__ == "__main__":
    main()
