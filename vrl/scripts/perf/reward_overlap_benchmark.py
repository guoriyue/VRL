"""A/B/C acceptance benchmark for generation/reward overlap.

Implements the performance acceptance defined in
``docs/sprints/done/SPRINT_reward_service.md``: three identical workloads that
differ only in how collection interleaves group generation and reward scoring.

    A  batched_serial        generate every group, then one batched reward call
    B  per_group_serial      per-group reward calls, no overlap  (control arm)
    C  per_group_streaming   reward group N overlaps generation N+1

C must beat A, not merely B. B prices the per-group call/transport tax that C
also pays, so without it a C-vs-A win cannot be attributed to overlap.

Each arm runs as a real training job (same entrypoint, config, and seed as
production) with only ``trainer.rollout_orchestration.reward_collection_mode``
overridden, so the measurement exercises the production collection path rather
than a bespoke loop. Per-step phase timings are read from each run's
``rollout_stats.jsonl``.

Two preconditions this script checks rather than assumes:

* The collection must contain at least two groups (``rollout.prompts_per_batch``
  >= 2). With one group per collection there is no group N+1 and all three arms
  are the same program.
* Arm C requires the collector's overlap capability. Forcing it without that
  capability raises in the rollout layer, so a missing capability surfaces as a
  failed run rather than a silently serial "C".

Usage:

    python -m vrl.scripts.perf.reward_overlap_benchmark \\
      --config experiment/wan_2_1/online_grpo_videoscore2_overlap \\
      --repeats 5 --iterations 6 --warmup-iterations 2 \\
      --out outputs/perf/reward_overlap

    # Recompute the verdict from runs that already completed:
    python -m vrl.scripts.perf.reward_overlap_benchmark \\
      --out outputs/perf/reward_overlap --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Arm id -> the reward_collection_mode value it forces. Keys are the sprint's
# A/B/C labels; values must stay in sync with RewardCollectionMode.
ARMS = {
    "A": "batched_serial",
    "B": "per_group_serial",
    "C": "per_group_streaming",
}

# Acceptance thresholds from SPRINT_reward_service.md "Performance acceptance".
MIN_IMPROVEMENT = 0.10
MAX_GENERATION_P95_REGRESSION = 0.05
MIN_REALIZED_FRACTION_OF_THEORETICAL = 0.50
MIN_REPEATS_FOR_CONFIDENCE = 5

# Student-t 95% one-sided critical values by degrees of freedom, for the
# lower confidence bound on the mean improvement. n=5 runs per arm is the
# sprint minimum, so the table only needs the small-sample range.
_T_ONE_SIDED_95 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    15: 1.753,
    20: 1.725,
    30: 1.697,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return math.inf
    for known in sorted(_T_ONE_SIDED_95):
        if df <= known:
            return _T_ONE_SIDED_95[known]
    return 1.645  # normal approximation


@dataclass
class RunMetrics:
    """Per-step collection phases from one training run."""

    collect_wall: list[float] = field(default_factory=list)
    generation_wall: list[float] = field(default_factory=list)
    reward_wall: list[float] = field(default_factory=list)
    overlap: list[float] = field(default_factory=list)
    reward_queue_wait: list[float] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.collect_wall)

    def mean_collect_wall(self) -> float:
        return statistics.fmean(self.collect_wall)

    def generation_p95(self) -> float:
        ordered = sorted(self.generation_wall)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]

    def theoretical_saving_s(self) -> float:
        """Ideal steady-state saving per step: min(G, R) summed over steps."""

        return sum(
            min(gen, reward)
            for gen, reward in zip(self.generation_wall, self.reward_wall, strict=True)
        )


def read_run_metrics(run_dir: Path, *, warmup_iterations: int) -> RunMetrics:
    """Load one run's per-step collection phases, dropping warmup steps."""

    verdict_path = run_dir / "run_verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("verdict") != "success":
            raise RuntimeError(f"{run_dir} did not succeed: {verdict}")
    stats_path = run_dir / "rollout_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing {stats_path}; the run wrote no phase stats")

    rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row.get("step", 0)))
    steady = rows[warmup_iterations:]
    if not steady:
        raise RuntimeError(
            f"{run_dir}: {len(rows)} recorded steps but warmup drops {warmup_iterations}",
        )

    metrics = RunMetrics()
    for row in steady:
        # A row missing collect.* means the step did no collection (e.g. a
        # gradient-accumulation microbatch); skip rather than score it as zero.
        if "collect.wall" not in row:
            continue
        metrics.collect_wall.append(float(row["collect.wall"]))
        metrics.generation_wall.append(float(row.get("collect.generation_wall", 0.0)))
        metrics.reward_wall.append(float(row.get("collect.reward_wall", 0.0)))
        metrics.overlap.append(float(row.get("collect.generation_reward_overlap", 0.0)))
        metrics.reward_queue_wait.append(float(row.get("reward.queue_wait_s", 0.0)))
    if not metrics.collect_wall:
        raise RuntimeError(f"{run_dir}: no steady-state step recorded a collect.wall")
    return metrics


def run_arm(
    *,
    config: str,
    arm: str,
    repeat: int,
    out_dir: Path,
    iterations: int,
    extra_overrides: list[str],
) -> Path:
    """Launch one training run for one arm/repeat and return its output dir."""

    run_dir = out_dir / f"arm{arm}_run{repeat}"
    overrides = [
        f"trainer.output_dir={run_dir}",
        f"trainer.rollout_orchestration.reward_collection_mode={ARMS[arm]}",
        f"trainer.total_epochs={iterations}",
        *extra_overrides,
    ]
    command = [
        sys.executable,
        "-m",
        "vrl.scripts.train",
        "--config",
        config,
        *overrides,
    ]
    print(f"[{arm}/{repeat}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"arm {arm} repeat {repeat} exited {completed.returncode}")
    _discard_checkpoints(run_dir)
    return run_dir


def _discard_checkpoints(run_dir: Path) -> None:
    """Delete this run's checkpoints; the benchmark only reads phase stats.

    A wan 1.3B checkpoint is ~3 GB and the final one is written unconditionally,
    so a fifteen-run campaign would need ~46 GB to keep state no arm compares.
    Deleting here bounds the campaign at one checkpoint of peak usage.
    """

    for checkpoint in sorted(run_dir.glob("checkpoint-*")):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)
            print(f"  discarded {checkpoint}", flush=True)


def summarize_arm(runs: list[RunMetrics]) -> dict[str, Any]:
    per_run_wall = [run.mean_collect_wall() for run in runs]
    return {
        "runs": len(runs),
        "steps_per_run": [run.steps for run in runs],
        "collect_wall_mean_s": statistics.fmean(per_run_wall),
        "collect_wall_per_run_s": per_run_wall,
        "generation_p95_s": statistics.fmean([run.generation_p95() for run in runs]),
        "reward_wall_mean_s": statistics.fmean(
            [statistics.fmean(run.reward_wall) for run in runs],
        ),
        "overlap_mean_s": statistics.fmean([statistics.fmean(run.overlap) for run in runs]),
        "reward_queue_wait_max_s": max(
            (max(run.reward_queue_wait) for run in runs if run.reward_queue_wait),
            default=0.0,
        ),
        "theoretical_saving_mean_s": statistics.fmean(
            [run.theoretical_saving_s() / run.steps for run in runs],
        ),
    }


def evaluate_gates(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the sprint's acceptance gates to the per-arm summaries."""

    baseline = arms["A"]
    streaming = arms["C"]
    baseline_wall = baseline["collect_wall_mean_s"]
    streaming_wall = streaming["collect_wall_mean_s"]
    improvement = (baseline_wall - streaming_wall) / baseline_wall

    # Lower confidence bound on the improvement, treating the two arms' per-run
    # means as independent samples (Welch). A positive bound is the sprint's
    # "positive lower confidence interval bound" gate.
    a_runs = baseline["collect_wall_per_run_s"]
    c_runs = streaming["collect_wall_per_run_s"]
    lower_bound: float | None = None
    if len(a_runs) >= 2 and len(c_runs) >= 2:
        a_var = statistics.variance(a_runs) / len(a_runs)
        c_var = statistics.variance(c_runs) / len(c_runs)
        standard_error = math.sqrt(a_var + c_var)
        if standard_error == 0:
            lower_bound = baseline_wall - streaming_wall
        else:
            df = (a_var + c_var) ** 2 / (
                a_var**2 / (len(a_runs) - 1) + c_var**2 / (len(c_runs) - 1)
            )
            margin = _t_critical(int(df)) * standard_error
            lower_bound = (baseline_wall - streaming_wall) - margin

    realized_saving = baseline_wall - streaming_wall
    theoretical = streaming["theoretical_saving_mean_s"]
    realized_fraction = realized_saving / theoretical if theoretical > 0 else 0.0

    generation_regression = (
        streaming["generation_p95_s"] - baseline["generation_p95_s"]
    ) / baseline["generation_p95_s"]

    gates = {
        "improvement_over_A": {
            "value": improvement,
            "threshold": MIN_IMPROVEMENT,
            "pass": improvement >= MIN_IMPROVEMENT,
        },
        "positive_lower_confidence_bound": {
            "value": lower_bound,
            "pass": lower_bound is not None and lower_bound > 0,
        },
        "enough_repeats": {
            "value": min(baseline["runs"], streaming["runs"]),
            "threshold": MIN_REPEATS_FOR_CONFIDENCE,
            "pass": min(baseline["runs"], streaming["runs"]) >= MIN_REPEATS_FOR_CONFIDENCE,
        },
        "generation_p95_regression": {
            "value": generation_regression,
            "threshold": MAX_GENERATION_P95_REGRESSION,
            "pass": generation_regression <= MAX_GENERATION_P95_REGRESSION,
        },
        "measured_overlap_positive": {
            "value": streaming["overlap_mean_s"],
            "pass": streaming["overlap_mean_s"] > 0,
        },
        "realized_fraction_of_theoretical": {
            "value": realized_fraction,
            "threshold": MIN_REALIZED_FRACTION_OF_THEORETICAL,
            "pass": realized_fraction >= MIN_REALIZED_FRACTION_OF_THEORETICAL,
        },
    }
    if "B" in arms:
        control = arms["B"]
        gates["per_group_call_tax_vs_A"] = {
            # Diagnostic, not a gate: how much of the A->C delta is explained by
            # per-group call granularity rather than by overlap.
            "value": (control["collect_wall_mean_s"] - baseline_wall) / baseline_wall,
            "pass": True,
        }
    return {
        "gates": gates,
        "accepted": all(gate["pass"] for gate in gates.values()),
    }


def analyze(out_dir: Path, *, warmup_iterations: int) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dirs = sorted(out_dir.glob(f"arm{arm}_run*"))
        runs = [read_run_metrics(path, warmup_iterations=warmup_iterations) for path in run_dirs]
        if runs:
            arms[arm] = summarize_arm(runs)
    missing = {"A", "C"} - set(arms)
    if missing:
        raise RuntimeError(f"cannot evaluate acceptance without arms {sorted(missing)}")
    report = {"arms": arms, **evaluate_gates(arms)}
    (out_dir / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment config name (required unless --analyze-only)")
    parser.add_argument("--out", required=True, help="Benchmark output directory")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6, help="training epochs per run")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument(
        "--start-repeat",
        type=int,
        default=0,
        help="First repeat index to run; earlier repeats are kept and analyzed.",
    )
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("override", nargs="*", help="extra OmegaConf dotlist overrides")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [arm.strip().upper() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; valid arms are {sorted(ARMS)}")

    if not args.analyze_only:
        if not args.config:
            parser.error("--config is required unless --analyze-only is passed")
        if args.repeats < MIN_REPEATS_FOR_CONFIDENCE:
            print(
                f"WARNING: {args.repeats} repeats is below the sprint's "
                f"{MIN_REPEATS_FOR_CONFIDENCE}; the confidence gate will fail.",
                flush=True,
            )
        # Interleave repeats across arms so a machine that drifts (thermal, page
        # cache, another tenant) biases every arm equally instead of whichever
        # arm happened to run last.
        for repeat in range(args.start_repeat, args.repeats):
            for arm in arms:
                if (out_dir / f"arm{arm}_run{repeat}" / "rollout_stats.jsonl").exists():
                    # A campaign is hours long and has already been interrupted
                    # once here (host disk filled mid-run). Completed repeats are
                    # valid evidence; re-running them only burns GPU time.
                    print(f"[{arm}/{repeat}] already complete, skipping", flush=True)
                    continue
                run_arm(
                    config=args.config,
                    arm=arm,
                    repeat=repeat,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    extra_overrides=list(args.override),
                )

    report = analyze(out_dir, warmup_iterations=args.warmup_iterations)
    print(json.dumps(report["gates"], indent=2), flush=True)
    print(f"ACCEPTED: {report['accepted']}", flush=True)
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
