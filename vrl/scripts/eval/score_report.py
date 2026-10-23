"""Model-independent fixed-prompt checkpoint curves.

Every checkpoint eval answers the same statistical question — did this
checkpoint score better than the base arm on the SAME prompt/seed grid — so
the distribution and the paired-delta bootstrap belong in one place rather
than being re-derived per family.

The bootstrap RNG is seeded from the report schema plus the label and score
key, so a rerun over the same rows reproduces the interval exactly.

``write_curve_report`` averages repeated samples within each prompt before
bootstrapping: prompts, not correlated seeds of one prompt, are independent
units. It loads no generation or reward model. Run
``python -m vrl.scripts.eval.score_report --help`` to report existing JSONL
scores with checkpoint_label, epoch, prompt_index, sample_index, seed, prompt,
and r_<component> columns; the comparison arm is labelled ``base``. These are
generic numeric measurements: a clear increase is not a claim of improved
quality or that higher is better. Repeated ``--joint-min COMPONENT=MIN`` adds a
``joint`` score: the fraction of samples on which every listed component
reaches its minimum at once (all constraints pass and quality >= tau),
reported with the same CIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.utils.json_files import read_jsonl, write_json
from vrl.utils.score_statistics import bootstrap_mean_interval, distribution

BASE_LABEL = "base"


def write_curve_report(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    score_keys: Sequence[str] | None = None,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
    joint_min: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Validate a paired grid and publish summary.json, curve.csv, and curve.png.

    The ``base`` arm is the paired comparison point. Each arm must contain the same unique prompt/sample cells with matching
    prompts and seeds. Epochs belong to arms and cannot change within an arm.
    Both absolute means and paired differences give every prompt equal weight,
    even if prompts have different numbers of sampled images or videos.
    ``r_*`` columns may contain diagnostics as well as rewards; directional
    findings describe increases/decreases, never whether either is desirable.

    ``joint_min`` maps components to minimum scores and adds the ``joint``
    score: 1 for a sample whose every listed component reaches its minimum
    (each constraint passes AND quality >= tau), else 0. It goes through the
    same prompt-level bootstrap as every other key, so its absolute mean is
    the passing fraction with a CI and its paired delta compares arms on it.
    """

    if not rows:
        raise ValueError("curve report requires score rows")
    keys = (
        tuple(score_keys)
        if score_keys is not None
        else tuple(sorted({key[2:] for row in rows for key in row if key.startswith("r_")}))
    )
    joint_min = dict(joint_min or {})
    if not all(math.isfinite(threshold) for threshold in joint_min.values()):
        raise ValueError("joint_min thresholds must be finite")
    # Joint components are read from every row, so they are validated like
    # score keys even when they are not reported on their own.
    measured = (*keys, *(key for key in joint_min if key not in keys))
    if joint_min:
        keys = (*keys, "joint")
    if not keys or len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("score_keys must be non-empty and unique")

    def score(row: dict[str, Any], key: str) -> float:
        if key == "joint" and joint_min:
            return float(
                all(float(row[f"r_{name}"]) >= minimum for name, minimum in joint_min.items())
            )
        return float(row[f"r_{key}"])

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
        for key in measured:
            try:
                value = float(row[f"r_{key}"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"missing or invalid r_{key} for {label}: {cell}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite r_{key} for {label}: {cell}")
        cells[cell] = row
    if BASE_LABEL not in grouped:
        raise ValueError(f"paired curve report requires {BASE_LABEL!r} rows")
    base = grouped[BASE_LABEL]
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
                    score(row, key) for cell, row in cells.items() if cell[0] == prompt_index
                )
                for prompt_index in sorted(prompts)
            }
            for key in keys
        }
    schema = "vrl.score_curve/v1"
    arms: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    for label in sorted(grouped, key=lambda name: (epochs[name], name != BASE_LABEL, name)):
        scores = {}
        for key in keys:
            values = list(prompt_means[label][key].values())
            deltas = [
                value - prompt_means[BASE_LABEL][key][index]
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
                "win_rate": sum(value > 0 for value in deltas) / len(deltas),
                "tie_rate": sum(value == 0 for value in deltas) / len(deltas),
                "clear_increase": lower > 0,
                "clear_decrease": upper < 0,
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
        "base_label": BASE_LABEL,
        "statistical_unit": "prompt",
        "bootstrap": {"resamples": bootstrap_resamples, "seed": seed},
        "arms": arms,
    }
    if joint_min:
        summary["joint_min"] = joint_min
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_curve_plot(curve_rows, keys, output_dir / "curve.png")
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    write_json(output_dir / "summary.json", summary)
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
    parser.add_argument(
        "--joint-min",
        action="append",
        default=[],
        metavar="COMPONENT=MIN",
        help=(
            "Repeat once per constraint and once for quality, e.g. "
            "--joint-min nsfw_safety=0 --joint-min pickscore=21.5: adds the 'joint' "
            "score, the fraction of samples whose every listed r_<component> >= MIN."
        ),
    )
    args = parser.parse_args(argv)
    joint_min: dict[str, float] = {}
    for item in args.joint_min:
        name, separator, minimum = item.rpartition("=")
        if not separator or not name or name in joint_min:
            parser.error(f"--joint-min expects distinct COMPONENT=MIN entries, got {item!r}")
        try:
            joint_min[name] = float(minimum)
        except ValueError:
            parser.error(f"--joint-min {item!r}: MIN must be a number")
    summary = write_curve_report(
        read_jsonl(args.scores),
        args.output_dir,
        score_keys=args.score_keys,
        joint_min=joint_min,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = ["write_curve_report"]


if __name__ == "__main__":
    main()
