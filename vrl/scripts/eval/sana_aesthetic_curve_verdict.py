"""Apply the SANA trustworthy-curve verdict.

Online training no longer produces ``eval_metrics.csv``. New runs consume the
provenance-bound report from ``sana_aesthetic_checkpoint_eval``; the old CSV is
accepted only for historical runs that already contain the inline-eval artifact.

The run protocol is documented in
``docs/sprints/SPRINT_sana_aesthetic_trustworthy_curve.md``. This program keeps
the numerical decision mechanical: it reads only the completed standalone report
and training CSV and emits a JSON PASS/FAIL record. Qualitative contact-sheet review
is a separate mandatory gate and is therefore reported as pending unless the
caller explicitly supplies its pre-recorded result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"metrics file has no data rows: {path}")
    return rows


def _read_eval_rows(run_dir: Path) -> list[dict[str, float]]:
    from vrl.scripts.eval.sana_aesthetic_checkpoint_eval import (
        REPORT_RELATIVE_PATH,
        load_report_metrics,
    )

    report_path = run_dir / REPORT_RELATIVE_PATH
    if report_path.is_file():
        return load_report_metrics(run_dir)
    legacy_path = run_dir / "eval_metrics.csv"
    if legacy_path.is_file():
        from omegaconf import OmegaConf

        config_path = run_dir / "resolved_config.yaml"
        if config_path.is_file():
            legacy_cfg = OmegaConf.load(config_path)
            # ``trainer.eval`` was intentionally removed from the current schema.
            # Inspect the archived resolved mapping as legacy data instead of
            # resurrecting it as a live config path in the runtime-key registry.
            legacy_plain = OmegaConf.to_container(legacy_cfg, resolve=True)
            legacy_trainer = (
                legacy_plain.get("trainer", {}) if isinstance(legacy_plain, dict) else {}
            )
            legacy_eval = (
                legacy_trainer.get("eval", {}) if isinstance(legacy_trainer, dict) else {}
            )
            if isinstance(legacy_eval, dict) and bool(legacy_eval.get("enabled", False)):
                return _read_rows(legacy_path)
    raise FileNotFoundError(
        f"standalone SANA evaluation report not found: {report_path}. Run "
        "`python -m vrl.scripts.eval.sana_aesthetic_checkpoint_eval "
        f"--run-dir {run_dir}` after checkpoints have been saved.",
    )


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return statistics.fmean(row[key] for row in rows)


def _slope(rows: list[dict[str, float]], key: str) -> float:
    xs = [row["epoch"] for row in rows]
    ys = [row[key] for row in rows]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return (
        0.0
        if denom == 0
        else sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denom
    )


def evaluate(
    eval_rows: list[dict[str, float]],
    train_rows: list[dict[str, float]],
    *,
    expected_updates: int = 300,
    min_post_eval_points: int = 12,
    endpoint_points: int = 3,
    min_aesthetic_gain: float = 0.10,
    min_gain_z: float = 2.0,
    max_pickscore_relative_drop: float = 0.02,
    max_pre_update_logprob_abs_diff: float = 1.0e-6,
    qualitative_audit: str = "pending",
) -> dict[str, Any]:
    failures: list[str] = []
    baseline_rows = [row for row in eval_rows if int(row["epoch"]) == -1]
    post_rows = [row for row in eval_rows if int(row["epoch"]) >= 0]
    if len(baseline_rows) != 1:
        failures.append("fixed eval must contain exactly one epoch=-1 baseline")
    if len(post_rows) < min_post_eval_points:
        failures.append(
            f"fixed eval needs at least {min_post_eval_points} post-training points",
        )
    if len(train_rows) != expected_updates:
        failures.append(
            f"run depth changed or incomplete: {len(train_rows)} != {expected_updates} updates",
        )

    required_eval = {
        "epoch",
        "eval_reward_stderr",
        "r_aesthetic",
        "r_pickscore",
    }
    required_train = {
        "epoch",
        "reward_std",
        "grad_norm",
        "trained_prompt_num",
        "active_clip_fraction",
        "pre_update_logprob_abs_diff_max",
        "pre_update_clip_fraction",
        "pre_update_active_clip_fraction",
    }
    for label, rows, keys in (
        ("eval", eval_rows, required_eval),
        ("train", train_rows, required_train),
    ):
        missing = keys - set(rows[0])
        if missing:
            failures.append(f"{label} metrics missing columns: {sorted(missing)}")
        for row in rows:
            for key, value in row.items():
                if not math.isfinite(value):
                    failures.append(f"{label} metrics contain non-finite {key}")
                    break

    diagnostics: dict[str, float | int | str] = {
        "updates": len(train_rows),
        "post_eval_points": len(post_rows),
        "qualitative_audit": qualitative_audit,
    }
    if (
        baseline_rows
        and len(post_rows) >= endpoint_points
        and not (required_eval - set(eval_rows[0]))
    ):
        baseline = baseline_rows[0]
        endpoint = post_rows[-endpoint_points:]
        aesthetic_end = _mean(endpoint, "r_aesthetic")
        aesthetic_gain = aesthetic_end - baseline["r_aesthetic"]
        endpoint_stderr = (
            math.sqrt(
                sum(row["eval_reward_stderr"] ** 2 for row in endpoint),
            )
            / endpoint_points
        )
        combined_stderr = math.hypot(baseline["eval_reward_stderr"], endpoint_stderr)
        gain_z = aesthetic_gain / combined_stderr if combined_stderr > 0 else math.inf
        aesthetic_slope = _slope(post_rows, "r_aesthetic")
        pickscore_end = _mean(endpoint, "r_pickscore")
        pickscore_floor = baseline["r_pickscore"] * (1.0 - max_pickscore_relative_drop)
        diagnostics.update(
            {
                "baseline_aesthetic": baseline["r_aesthetic"],
                "endpoint_aesthetic": aesthetic_end,
                "aesthetic_gain": aesthetic_gain,
                "aesthetic_gain_z": gain_z,
                "aesthetic_slope_per_epoch": aesthetic_slope,
                "baseline_pickscore": baseline["r_pickscore"],
                "endpoint_pickscore": pickscore_end,
                "pickscore_floor": pickscore_floor,
            }
        )
        if aesthetic_gain < min_aesthetic_gain:
            failures.append(
                f"aesthetic gain {aesthetic_gain:.6f} < {min_aesthetic_gain:.6f}",
            )
        if gain_z <= min_gain_z:
            failures.append(f"aesthetic gain z={gain_z:.6f} <= {min_gain_z:.6f}")
        if aesthetic_slope <= 0:
            failures.append(f"fixed-eval aesthetic slope {aesthetic_slope:.6f} is not positive")
        if pickscore_end < pickscore_floor:
            failures.append(
                f"PickScore regressed: {pickscore_end:.6f} < floor {pickscore_floor:.6f}",
            )

    if not (required_train - set(train_rows[0])):
        window = min(32, len(train_rows))
        early_std = statistics.median(row["reward_std"] for row in train_rows[:window])
        late_std = statistics.median(row["reward_std"] for row in train_rows[-window:])
        max_parity_error = max(row["pre_update_logprob_abs_diff_max"] for row in train_rows)
        max_pre_update_clip = max(row["pre_update_clip_fraction"] for row in train_rows)
        max_pre_update_active_clip = max(
            row["pre_update_active_clip_fraction"] for row in train_rows
        )
        max_policy_active_clip = max(row["active_clip_fraction"] for row in train_rows)
        diagnostics.update(
            {
                "early_reward_std_median": early_std,
                "late_reward_std_median": late_std,
                "max_pre_update_logprob_abs_diff": max_parity_error,
                "max_pre_update_clip_fraction": max_pre_update_clip,
                "max_pre_update_active_clip_fraction": max_pre_update_active_clip,
                "max_active_policy_clip_fraction": max_policy_active_clip,
            }
        )
        if late_std <= 1e-4 or late_std < 0.25 * early_std:
            failures.append("late reward diversity collapsed")
        if max_parity_error > max_pre_update_logprob_abs_diff:
            failures.append(
                "pre-update logprob parity error "
                f"{max_parity_error:.6f} > {max_pre_update_logprob_abs_diff:.6f}",
            )
        if max_pre_update_clip != 0 or max_pre_update_active_clip != 0:
            failures.append("pre-update policy ratio or active clip is non-zero")
        if not any(row["grad_norm"] > 0 for row in train_rows):
            failures.append("all gradient norms are zero")
        if any(row["trained_prompt_num"] <= 0 for row in train_rows):
            failures.append("an update trained zero prompts")
        # This full-parameter protocol has one PPO pass. It replays the rollout
        # policy before the sole optimizer update, so surrogate clipping would
        # indicate backend drift rather than healthy later-pass policy movement.
        if max_policy_active_clip != 0:
            failures.append(
                "single-pass full-parameter policy unexpectedly engaged surrogate clipping",
            )

    if qualitative_audit not in {"pass", "fail"}:
        failures.append("qualitative baseline/endpoint audit is not recorded")
    elif qualitative_audit == "fail":
        failures.append("qualitative reward-hacking audit failed")

    return {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "diagnostics": diagnostics,
        "criteria": {
            "expected_updates": expected_updates,
            "min_post_eval_points": min_post_eval_points,
            "endpoint_points": endpoint_points,
            "min_aesthetic_gain": min_aesthetic_gain,
            "min_gain_z": min_gain_z,
            "max_pickscore_relative_drop": max_pickscore_relative_drop,
            "max_pre_update_logprob_abs_diff": max_pre_update_logprob_abs_diff,
            "expected_active_clip_fraction": 0.0,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--qualitative-audit", choices=("pass", "fail", "pending"), default="pending"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        eval_rows = _read_eval_rows(args.run_dir)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    result = evaluate(
        eval_rows,
        _read_rows(args.run_dir / "metrics.csv"),
        qualitative_audit=args.qualitative_audit,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
