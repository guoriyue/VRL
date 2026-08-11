"""Tests for deriving Wan T2V rows from target-backed Video2World data."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from omegaconf import OmegaConf

from vrl.scripts.data import bootstrap, setup
from vrl.scripts.data.derive_text_video_targets import derive_text_video_rows


def test_derive_text_video_rows_removes_unconsumed_first_frame() -> None:
    rows = derive_text_video_rows(
        [
            {
                "prompt": "Put the blue block in the bowl",
                "reference_image": "video_world/references/episode.png",
                "target_video": "video_world/targets/episode.mp4",
                "task_type": "video2world",
                "metadata": {
                    "source": "droid",
                    "source_episode": "000001",
                    "conditioning": "first_frame",
                },
            },
        ],
    )

    assert rows == [
        {
            "prompt": "Put the blue block in the bowl",
            "target_video": "video_world/targets/episode.mp4",
            "task_type": "text_to_video",
            "metadata": {
                "source": "droid",
                "source_episode": "000001",
                "source_conditioning": "first_frame",
                "conditioning": "text_only",
                "derived_from_task_type": "video2world",
            },
        },
    ]


def _source_row(*, target_video: str, source_episode: str) -> dict[str, object]:
    return {
        "prompt": f"Move the object in episode {source_episode}",
        "reference_image": f"video_world/references/{source_episode}.png",
        "target_video": target_video,
        "task_type": "video2world",
        "metadata": {
            "source": "droid",
            "source_repo": "lerobot/droid_1.0.1",
            "source_split": "train",
            "source_episode": source_episode,
            "source_video": f"videos/{source_episode}.mp4",
            "source_frame_index": 0,
            "decode_method": "pyav_http_target_clip",
            "conditioning": "first_frame",
        },
    }


def test_setup_cli_derives_and_validates_text_video_manifests(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "external"
    targets = data_root / "video_world" / "targets"
    targets.mkdir(parents=True)
    (targets / "train.mp4").write_bytes(b"train-video")
    (targets / "eval.mp4").write_bytes(b"eval-video")

    sources = tmp_path / "sources"
    sources.mkdir()
    train_source = sources / "train.jsonl"
    eval_source = sources / "eval.jsonl"
    train_source.write_text(
        json.dumps(
            _source_row(
                target_video="video_world/targets/train.mp4",
                source_episode="000001",
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    eval_source.write_text(
        json.dumps(
            _source_row(
                target_video="video_world/targets/eval.mp4",
                source_episode="000002",
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = data_root / "video_world" / "manifests"

    setup.main(
        [
            "derive-text-video-targets",
            "--train-manifest",
            str(train_source),
            "--eval-manifest",
            str(eval_source),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--name",
            "test_targets_t2v",
        ],
    )

    train_output = output_dir / "test_targets_t2v_train.jsonl"
    eval_output = output_dir / "test_targets_t2v_eval.jsonl"
    report = json.loads((output_dir / "test_targets_t2v_report.json").read_text())
    assert json.loads(train_output.read_text())["task_type"] == "text_to_video"
    assert json.loads(eval_output.read_text())["task_type"] == "text_to_video"
    assert report["train_rows"] == report["eval_rows"] == 1
    assert report["validation_summary"]["artifact_count"] == 1


def test_bootstrap_returns_canonical_derived_manifest_command(tmp_path: Path) -> None:
    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_manifest",
            "manifest": ("data/external/video_world/manifests/droid_full_targets_t2v_train.jsonl"),
            "eval_manifest": (
                "data/external/video_world/manifests/droid_full_targets_t2v_eval.jsonl"
            ),
            "source_report": (
                "data/external/video_world/manifests/droid_full_targets_t2v_report.json"
            ),
        },
        repo_root=tmp_path,
    )

    commands = {step["get"] for step in plan["steps"]}
    assert plan["ready"] is False
    assert len(commands) == 1
    command = commands.pop()
    assert command.startswith(
        "python -m vrl.scripts.data.setup derive-text-video-targets ",
    )
    assert "--data-root data/external" in command
    assert "--output-dir data/external/video_world/manifests" in command


def test_bootstrap_runs_shared_derived_manifest_producer_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "data": {
                "loader": "prompt_manifest",
                "manifest": (
                    "data/external/video_world/manifests/droid_full_targets_t2v_train.jsonl"
                ),
                "eval_manifest": (
                    "data/external/video_world/manifests/droid_full_targets_t2v_eval.jsonl"
                ),
                "source_report": (
                    "data/external/video_world/manifests/droid_full_targets_t2v_report.json"
                ),
            },
        },
    )
    commands: list[str] = []
    monkeypatch.setattr("vrl.config.loading.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(bootstrap, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "_run_setup_command", commands.append)

    bootstrap._cmd_for_experiment(
        Namespace(experiment="test/derived", override=[], run=True),
    )

    assert len(commands) == 1
    assert "derive-text-video-targets" in commands[0]
