from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.scripts.eval import score_report
from vrl.scripts.eval import wan_hpsv3_checkpoint_eval as checkpoint_eval


def _score_row(label: str, prompt: int, sample: int, value: float) -> dict[str, object]:
    return {
        "checkpoint_label": label,
        "prompt_index": prompt,
        "sample_index": sample,
        "r_top_frame_mean": value,
        "r_frame_mean": value - 0.5,
        "r_frame_min": value - 1.0,
    }


def _grid(label: str, values: list[float]) -> list[dict[str, object]]:
    return [_score_row(label, index // 2, index % 2, value) for index, value in enumerate(values)]


# --- paired summary ----------------------------------------------------------


def test_paired_delta_detects_improvement_and_regression() -> None:
    rows = [
        *_grid("base", [1.0, 2.0, 3.0, 4.0]),
        *_grid("better", [2.0, 3.0, 4.0, 5.0]),
        *_grid("worse", [0.0, 1.0, 2.0, 3.0]),
    ]

    summary = score_report.summarize_paired_scores(
        rows,
        score_keys=checkpoint_eval.SCORE_KEYS,
        schema=checkpoint_eval.REPORT_SCHEMA,
    )

    better = summary["paired_delta_from_base"]["better"]["top_frame_mean"]
    worse = summary["paired_delta_from_base"]["worse"]["top_frame_mean"]
    assert better["mean"] == pytest.approx(1.0)
    assert better["win_rate"] == 1.0
    assert better["clear_improvement"] is True
    assert better["clear_regression"] is False
    assert worse["mean"] == pytest.approx(-1.0)
    assert worse["clear_regression"] is True
    assert summary["absolute"]["base"]["top_frame_mean"]["mean"] == pytest.approx(2.5)


def test_paired_summary_reports_every_score_key() -> None:
    """frame_mean/frame_min ride along so reward hacking stays visible."""

    rows = [*_grid("base", [1.0, 2.0]), *_grid("later", [3.0, 4.0])]

    summary = score_report.summarize_paired_scores(
        rows,
        score_keys=checkpoint_eval.SCORE_KEYS,
        schema=checkpoint_eval.REPORT_SCHEMA,
    )

    assert set(summary["paired_delta_from_base"]["later"]) == set(checkpoint_eval.SCORE_KEYS)


def test_paired_summary_refuses_a_mismatched_grid() -> None:
    rows = [*_grid("base", [1.0, 2.0]), *_grid("short", [1.0])]

    with pytest.raises(ValueError, match="paired score grid differs"):
        score_report.summarize_paired_scores(
            rows,
            score_keys=("top_frame_mean",),
            schema=checkpoint_eval.REPORT_SCHEMA,
        )


def test_paired_summary_requires_the_base_arm() -> None:
    with pytest.raises(ValueError, match="requires 'base' rows"):
        score_report.summarize_paired_scores(
            _grid("only", [1.0, 2.0]),
            score_keys=("top_frame_mean",),
            schema=checkpoint_eval.REPORT_SCHEMA,
        )


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1, 0.4, -0.2, 0.9, 0.3, 0.5]
    first = score_report.bootstrap_mean_interval(
        values,
        schema="s",
        label="ckpt",
        score_key="top_frame_mean",
    )
    second = score_report.bootstrap_mean_interval(
        values,
        schema="s",
        label="ckpt",
        score_key="top_frame_mean",
    )

    assert first == second
    assert first[0] < sum(values) / len(values) < first[1]


def test_write_scores_publishes_jsonl_and_csv(tmp_path: Path) -> None:
    rows = _grid("base", [1.0, 2.0])

    score_report.write_scores(rows, tmp_path)

    written = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert written == rows
    assert "r_top_frame_mean" in (tmp_path / "scores.csv").read_text().splitlines()[0]


# --- run config --------------------------------------------------------------


def _write_run(tmp_path: Path, **model_overrides: object) -> Path:
    model = {
        "family": "wan",
        "path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "use_lora": True,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "target_modules": ["to_q"],
            "path": "donor/lora_weights",
        },
        "torch_compile": {"enable": True},
        **model_overrides,
    }
    OmegaConf.save(OmegaConf.create({"model": model}), tmp_path / "resolved_config.yaml")
    return tmp_path


def test_run_config_clears_the_warm_start_adapter_and_compile(tmp_path: Path) -> None:
    """A warm-started run records its donor adapter; the base arm must not inherit it."""

    cfg = checkpoint_eval._load_run_config(_write_run(tmp_path))

    assert OmegaConf.select(cfg, "model.lora.path") is None
    assert OmegaConf.select(cfg, "model.torch_compile.enable") is False
    assert OmegaConf.select(cfg, "model.use_lora") is True


def test_run_config_refuses_another_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a wan_2_1 run"):
        checkpoint_eval._load_run_config(_write_run(tmp_path, family="sana"))


def test_run_config_requires_a_resolved_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"resolved_config\.yaml"):
        checkpoint_eval._load_run_config(tmp_path)


# --- reward worker config ----------------------------------------------------


def test_score_grid_builds_hpsv3_from_the_runs_own_reward_block(
    monkeypatch, tmp_path: Path
) -> None:
    """``score`` projects the run's ``reward.kwargs.hpsv3`` block (plus the eval
    device) into the HPSv3 worker config and merges every score key into the
    generated rows."""

    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    # A run's resolved config always carries components; the shared projection
    # validates the whole reward section rather than selecting one subtree.
    cfg.reward = {
        "components": {"hpsv3": 1.0},
        "kwargs": {
            "hpsv3": {
                "reward_name": "MizzenAI/HPSv3@main",
                "worker_config": {"model_path": "/models/HPSv3", "dtype": "bfloat16"},
            },
        },
    }
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    video = output_dir / "base.mp4"
    video.write_bytes(b"not really a video")
    row = {
        "checkpoint_label": "base",
        "prompt_index": 0,
        "sample_index": 0,
        "seed": 1,
        "prompt": "a prompt",
        "path": str(video),
        "size_bytes": video.stat().st_size,
        "sha256": checkpoint_eval.sha256_file(video),
    }
    (output_dir / "generated.jsonl").write_text(json.dumps(row) + "\n")
    built: list[dict[str, object]] = []

    class FakeHPSv3Model:
        def __init__(self, worker_config: dict[str, object]) -> None:
            built.append(worker_config)

        def __call__(self, artifact) -> dict[str, float]:
            return {"top_frame_mean": 3.0, "frame_mean": 2.0, "frame_min": 1.0}

    monkeypatch.setattr("vrl.rewards.models.hpsv3.HPSv3Model", FakeHPSv3Model)

    report = checkpoint_eval.score_grid(
        checkpoint_eval.build_parser().parse_args(
            [
                "score",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
        ),
    )

    (worker_config,) = built
    assert worker_config["model_path"] == "/models/HPSv3"
    assert worker_config["dtype"] == "bfloat16"
    assert worker_config["reward_model_name"] == "MizzenAI/HPSv3@main"
    assert worker_config["device"] == "cpu"
    assert report["scored"] == 1
    scored = [json.loads(line) for line in (output_dir / "scores.jsonl").read_text().splitlines()]
    assert scored[0]["r_top_frame_mean"] == 3.0
    assert scored[0]["r_frame_min"] == 1.0
