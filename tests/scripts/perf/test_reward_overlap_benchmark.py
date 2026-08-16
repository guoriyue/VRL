"""Gate math for the A/B/C generation/reward overlap acceptance benchmark.

The GPU campaign is expensive and runs unattended, so the verdict logic is
tested against synthetic run directories: a benchmark that silently accepts a
regression is worse than no benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.scripts.perf.reward_overlap_benchmark import (
    analyze,
    evaluate_gates,
    read_run_metrics,
    summarize_arm,
)


def _write_run(
    out_dir: Path,
    arm: str,
    repeat: int,
    *,
    steps: int = 6,
    collect_wall: float,
    generation_wall: float,
    reward_wall: float,
    overlap: float,
    verdict: str = "success",
) -> Path:
    run_dir = out_dir / f"arm{arm}_run{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_verdict.json").write_text(json.dumps({"verdict": verdict}))
    lines = []
    for step in range(steps):
        lines.append(
            json.dumps(
                {
                    "step": step,
                    "collect.wall": collect_wall,
                    "collect.generation_wall": generation_wall,
                    "collect.reward_wall": reward_wall,
                    "collect.generation_reward_overlap": overlap,
                },
            ),
        )
    (run_dir / "rollout_stats.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir


def _passing_campaign(out_dir: Path, repeats: int = 5) -> None:
    """A/B serial at 100s, C overlapping reward (60s) into generation (40s).

    Serial wall 100 = G 40 + R 60; ideal overlap wall is max(G, R) = 60, so the
    theoretical saving is min(G, R) = 40. C lands at 70, realizing 30 of 40.
    """

    for repeat in range(repeats):
        _write_run(
            out_dir,
            "A",
            repeat,
            collect_wall=100.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )
        _write_run(
            out_dir,
            "B",
            repeat,
            collect_wall=102.0,
            generation_wall=40.0,
            reward_wall=62.0,
            overlap=0.0,
        )
        _write_run(
            out_dir,
            "C",
            repeat,
            collect_wall=70.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=30.0,
        )


def test_accepts_a_campaign_that_meets_every_gate(tmp_path: Path) -> None:
    _passing_campaign(tmp_path)

    report = analyze(tmp_path, warmup_iterations=2)

    gates = report["gates"]
    assert report["accepted"] is True
    assert gates["improvement_over_A"]["value"] == pytest.approx(0.30)
    assert gates["measured_overlap_positive"]["value"] == pytest.approx(30.0)
    # Realized 30s of the theoretical 40s min(G, R).
    assert gates["realized_fraction_of_theoretical"]["value"] == pytest.approx(0.75)
    assert gates["positive_lower_confidence_bound"]["pass"] is True
    # The control arm's per-group tax is reported as a diagnostic.
    assert gates["per_group_call_tax_vs_A"]["value"] == pytest.approx(0.02)
    assert json.loads((tmp_path / "acceptance.json").read_text())["accepted"] is True


def test_rejects_improvement_below_the_ten_percent_floor(tmp_path: Path) -> None:
    """A 5% win is the sprint's kill condition, not a pass."""
    for repeat in range(5):
        _write_run(
            tmp_path,
            "A",
            repeat,
            collect_wall=100.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )
        _write_run(
            tmp_path,
            "C",
            repeat,
            collect_wall=95.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=5.0,
        )

    report = analyze(tmp_path, warmup_iterations=2)

    assert report["accepted"] is False
    assert report["gates"]["improvement_over_A"]["pass"] is False


def test_rejects_generation_p95_regression_beyond_five_percent(tmp_path: Path) -> None:
    """Overlap that steals generation throughput is not a win."""
    for repeat in range(5):
        _write_run(
            tmp_path,
            "A",
            repeat,
            collect_wall=100.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )
        _write_run(
            tmp_path,
            "C",
            repeat,
            collect_wall=80.0,
            generation_wall=44.0,
            reward_wall=60.0,
            overlap=25.0,
        )

    report = analyze(tmp_path, warmup_iterations=2)

    assert report["accepted"] is False
    assert report["gates"]["generation_p95_regression"]["value"] == pytest.approx(0.10)
    assert report["gates"]["generation_p95_regression"]["pass"] is False


def test_rejects_when_realized_saving_is_under_half_the_theoretical(tmp_path: Path) -> None:
    """Queue/hash/network overhead eating the saving is a kill condition."""
    for repeat in range(5):
        _write_run(
            tmp_path,
            "A",
            repeat,
            collect_wall=100.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )
        # 12s realized of a theoretical 40s = 0.30.
        _write_run(
            tmp_path,
            "C",
            repeat,
            collect_wall=88.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=12.0,
        )

    report = analyze(tmp_path, warmup_iterations=2)

    assert report["accepted"] is False
    assert report["gates"]["realized_fraction_of_theoretical"]["value"] == pytest.approx(0.30)
    assert report["gates"]["realized_fraction_of_theoretical"]["pass"] is False


def test_rejects_fewer_than_five_repeats(tmp_path: Path) -> None:
    """A big win from two runs still fails the confidence requirement."""
    _passing_campaign(tmp_path, repeats=2)

    report = analyze(tmp_path, warmup_iterations=2)

    assert report["accepted"] is False
    assert report["gates"]["enough_repeats"]["pass"] is False


def test_noisy_arms_fail_the_confidence_bound_despite_a_mean_win(tmp_path: Path) -> None:
    """Run-to-run variance wider than the effect must not be accepted."""
    a_walls = [60.0, 140.0, 60.0, 140.0, 100.0]
    c_walls = [55.0, 130.0, 55.0, 130.0, 60.0]
    for repeat, (a_wall, c_wall) in enumerate(zip(a_walls, c_walls, strict=True)):
        _write_run(
            tmp_path,
            "A",
            repeat,
            collect_wall=a_wall,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )
        _write_run(
            tmp_path,
            "C",
            repeat,
            collect_wall=c_wall,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=20.0,
        )

    report = analyze(tmp_path, warmup_iterations=2)

    assert report["gates"]["positive_lower_confidence_bound"]["value"] < 0
    assert report["accepted"] is False


def test_failed_run_is_not_silently_averaged_in(tmp_path: Path) -> None:
    """A crashed run must abort the verdict, not shrink the sample."""
    _passing_campaign(tmp_path)
    _write_run(
        tmp_path,
        "C",
        9,
        collect_wall=10.0,
        generation_wall=40.0,
        reward_wall=60.0,
        overlap=30.0,
        verdict="failed",
    )

    with pytest.raises(RuntimeError, match="did not succeed"):
        analyze(tmp_path, warmup_iterations=2)


def test_warmup_steps_are_dropped_before_averaging(tmp_path: Path) -> None:
    """Cold-start steps must not enter the steady-state mean."""
    run_dir = tmp_path / "armA_run0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_verdict.json").write_text(json.dumps({"verdict": "success"}))
    rows = [
        {
            "step": 0,
            "collect.wall": 500.0,
            "collect.generation_wall": 1.0,
            "collect.reward_wall": 1.0,
            "collect.generation_reward_overlap": 0.0,
        },
        {
            "step": 1,
            "collect.wall": 300.0,
            "collect.generation_wall": 1.0,
            "collect.reward_wall": 1.0,
            "collect.generation_reward_overlap": 0.0,
        },
        {
            "step": 2,
            "collect.wall": 100.0,
            "collect.generation_wall": 1.0,
            "collect.reward_wall": 1.0,
            "collect.generation_reward_overlap": 0.0,
        },
        {
            "step": 3,
            "collect.wall": 100.0,
            "collect.generation_wall": 1.0,
            "collect.reward_wall": 1.0,
            "collect.generation_reward_overlap": 0.0,
        },
    ]
    (run_dir / "rollout_stats.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
    )

    metrics = read_run_metrics(run_dir, warmup_iterations=2)

    assert metrics.steps == 2
    assert metrics.mean_collect_wall() == pytest.approx(100.0)


def test_steps_without_collection_are_skipped_not_scored_as_zero(tmp_path: Path) -> None:
    """A gradient-accumulation step that collected nothing must not dilute the mean."""
    run_dir = tmp_path / "armA_run0"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "step": 0,
            "collect.wall": 100.0,
            "collect.generation_wall": 40.0,
            "collect.reward_wall": 60.0,
            "collect.generation_reward_overlap": 0.0,
        },
        {"step": 1, "backward": 3.0},
        {
            "step": 2,
            "collect.wall": 100.0,
            "collect.generation_wall": 40.0,
            "collect.reward_wall": 60.0,
            "collect.generation_reward_overlap": 0.0,
        },
    ]
    (run_dir / "rollout_stats.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
    )

    metrics = read_run_metrics(run_dir, warmup_iterations=0)

    assert metrics.steps == 2
    assert metrics.mean_collect_wall() == pytest.approx(100.0)


def test_analysis_requires_both_the_baseline_and_streaming_arms(tmp_path: Path) -> None:
    """B alone cannot produce a verdict: C must beat A, not merely B."""
    for repeat in range(5):
        _write_run(
            tmp_path,
            "B",
            repeat,
            collect_wall=100.0,
            generation_wall=40.0,
            reward_wall=60.0,
            overlap=0.0,
        )

    with pytest.raises(RuntimeError, match=r"\['A', 'C'\]"):
        analyze(tmp_path, warmup_iterations=2)


def test_theoretical_saving_uses_min_of_generation_and_reward(tmp_path: Path) -> None:
    """The ideal saving is min(G, R); the bottleneck stage cannot be hidden."""
    run_dir = _write_run(
        tmp_path,
        "C",
        0,
        steps=1,
        collect_wall=70.0,
        generation_wall=40.0,
        reward_wall=60.0,
        overlap=30.0,
    )

    metrics = read_run_metrics(run_dir, warmup_iterations=0)

    assert metrics.theoretical_saving_s() == pytest.approx(40.0)


def test_zero_variance_arms_still_yield_a_positive_bound(tmp_path: Path) -> None:
    """Identical repeats are a degenerate but real case; do not divide by zero."""
    _passing_campaign(tmp_path)

    arms = {
        arm: summarize_arm(
            [
                read_run_metrics(path, warmup_iterations=2)
                for path in sorted(tmp_path.glob(f"arm{arm}_run*"))
            ],
        )
        for arm in ("A", "C")
    }
    result = evaluate_gates(arms)

    assert result["gates"]["positive_lower_confidence_bound"]["value"] == pytest.approx(30.0)
    assert result["gates"]["positive_lower_confidence_bound"]["pass"] is True
