from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.models import checkpoint_identity
from vrl.scripts.eval import wan_robotics_checkpoint_eval as checkpoint_eval
from vrl.trainers.data import PromptExample


def _example(
    prompt: str,
    *,
    episode: str,
    source_video: str,
    frame: int,
    target: str,
) -> PromptExample:
    return PromptExample(
        prompt=prompt,
        target_video=target,
        metadata={
            "source_repo": "robot/data",
            "source_episode": episode,
            "source_video": source_video,
            "source_frame_index": frame,
        },
    )


def test_strict_selection_excludes_training_prompts_and_source_shards() -> None:
    train = [
        _example(
            "Move the bowl",
            episode="train-1",
            source_video="file-0.mp4",
            frame=0,
            target="train.mp4",
        ),
    ]
    evaluation = [
        _example(
            "  move   the BOWL ",
            episode="eval-1",
            source_video="file-1.mp4",
            frame=1,
            target="eval-1.mp4",
        ),
        _example(
            "Pick up the cup",
            episode="eval-2",
            source_video="file-0.mp4",
            frame=2,
            target="eval-2.mp4",
        ),
        _example(
            "Open the drawer",
            episode="eval-3",
            source_video="file-2.mp4",
            frame=3,
            target="eval-3.mp4",
        ),
    ]

    selected = checkpoint_eval.select_strict_examples(train, evaluation, limit=1)

    assert [row.row_index for row in selected] == [2]


def test_strict_selection_rejects_episode_or_target_leakage() -> None:
    train = [
        _example(
            "Train",
            episode="shared",
            source_video="train.mp4",
            frame=0,
            target="train-target.mp4",
        ),
    ]
    evaluation = [
        _example(
            "Eval",
            episode="shared",
            source_video="eval.mp4",
            frame=1,
            target="eval-target.mp4",
        ),
    ]

    with pytest.raises(ValueError, match="source episodes overlap"):
        checkpoint_eval.select_strict_examples(train, evaluation, limit=1)


def test_seed_grid_depends_on_manifest_row_not_checkpoint() -> None:
    first = checkpoint_eval._seed_for(
        base_seed=100,
        row_index=54,
        sample_index=0,
        seed_stride=1_000,
    )
    second = checkpoint_eval._seed_for(
        base_seed=100,
        row_index=54,
        sample_index=1,
        seed_stride=1_000,
    )

    assert first == 154
    assert second == 1_154


def test_selection_identity_preserves_zero_frame_index() -> None:
    train = [
        _example(
            "Train",
            episode="train-1",
            source_video="file-0.mp4",
            frame=1,
            target="train.mp4",
        ),
    ]
    evaluation = [
        _example(
            "Take the pink plate off the blue bowl",
            episode="001214",
            source_video="file-4.mp4",
            frame=0,
            target="video_world/targets/droid_001214_target.mp4",
        ),
    ]

    selected = checkpoint_eval.select_strict_examples(train, evaluation, limit=1)

    assert selected[0].identity_sha256 == (
        "edcd62f602d2ca0a85d141fb901530c8200a016014d3c26cef8ab1b3b6e0187e"
    )


def test_target_resolution_distinguishes_base_from_complete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    payload = checkpoint / "checkpoint.pt"
    payload.write_bytes(b"checkpoint")
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "family": "wan_2_1",
                "uses_lora": False,
                "completed_epoch": 5,
                "checkpoint_file_bytes": payload.stat().st_size,
            },
        ),
        encoding="utf-8",
    )

    base = checkpoint_eval._resolve_target(tmp_path, "0")
    trained = checkpoint_eval._resolve_target(tmp_path, "5")

    assert base.label == "base"
    assert base.path is None
    assert trained.label == "checkpoint-5"
    assert trained.path == checkpoint.resolve()


def test_summary_reports_paired_deltas_and_reference_separately() -> None:
    rows = []
    for label, values in {
        "base": [0.0, 1.0],
        "checkpoint-5": [1.0, 3.0],
        "reference": [4.0],
    }.items():
        for index, value in enumerate(values):
            row = {
                "checkpoint_label": label,
                "row_index": index,
                "sample_index": 0,
            }
            row.update({f"r_{key}": value for key in checkpoint_eval.SCORE_KEYS})
            rows.append(row)

    summary = checkpoint_eval.summarize_scores(rows)

    blend = summary["paired_delta_from_base"]["checkpoint-5"]["robotics_blend"]
    assert blend["mean"] == pytest.approx(1.5)
    assert blend["median"] == pytest.approx(1.5)
    assert blend["win_rate"] == 1.0
    assert blend["clear_improvement"] is True
    assert summary["best_mean_robotics_blend"] == "checkpoint-5"


def test_scoring_artifact_preserves_target_metadata(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.mp4"
    generated_path.write_bytes(b"generated")
    target_path = tmp_path / "target.mp4"
    target_path.write_bytes(b"target")
    generated = [
        {
            "checkpoint_label": "base",
            "epoch": 0,
            "row_index": 2,
            "sample_index": 0,
            "seed": 7,
            "prompt": "Move the cup",
            "target_video": "target.mp4",
            "source_episode": "episode-2",
            "path": str(generated_path),
            "bytes": generated_path.stat().st_size,
            "sha256": checkpoint_eval.sha256_file(generated_path),
        },
    ]
    protocol = {
        "selected_examples": [
            {
                "row_index": 2,
                "prompt": "Move the cup",
                "target_video": "target.mp4",
                "source_episode": "episode-2",
            },
        ],
    }
    cfg = OmegaConf.create({"data": {"artifact_data_root": str(tmp_path)}})

    artifacts, rows = checkpoint_eval._build_scoring_artifacts(
        generated,
        protocol,
        cfg,
        include_reference_targets=True,
    )

    assert len(artifacts) == 2
    assert artifacts[0].metadata["target_video"] == "target.mp4"
    assert artifacts[1].metadata["target_video"] == "target.mp4"
    assert [row["checkpoint_label"] for row in rows] == ["base", "reference"]


def test_base_generation_never_reads_a_training_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    output_dir = tmp_path / "eval"
    cfg = OmegaConf.create(
        {
            "model": {"family": "wan"},
            "precision": {"float32_precision": "ieee", "training": {"dtype": "fp32"}},
        },
    )
    target = checkpoint_eval.CheckpointTarget.base()
    protocol = {
        "selected_examples": [
            {
                "row_index": 2,
                "identity_sha256": "a" * 64,
                "prompt": "Move the cup",
                "target_video": "target.mp4",
                "source_episode": "episode-2",
                "source_video": "source.mp4",
            },
        ],
        "sampling": {
            "fps": 15,
            "width": 8,
            "height": 8,
            "num_frames": 2,
            "num_steps": 1,
            "max_sequence_length": 8,
            "guidance_scale": 1.0,
            "denoise_mode": "native",
            "noise_level": 0.0,
            "sde_type": "cps",
        },
    }

    class FakeModel:
        def eval(self):
            return self

    bundle = SimpleNamespace(model=FakeModel())
    entry = SimpleNamespace(
        family="wan_2_1",
        resolve_model_build=lambda *args, **kwargs: object(),
        build_rollout=lambda build: bundle,
    )
    monkeypatch.setattr(checkpoint_eval, "_load_run", lambda path: (run_dir, cfg, config_path))
    monkeypatch.setattr(checkpoint_eval, "_resolve_target", lambda *args: target)
    monkeypatch.setattr(checkpoint_eval, "_build_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(checkpoint_eval, "get_model_family_entry", lambda family: entry)
    # run.resolve_model resolves identity off the build; the fake entry returns a
    # bare object(), so stub the resolver at its owning module (the seam-test pattern).
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda actual_build: {"schema": "test"},
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "load_training_checkpoint",
        lambda path: pytest.fail("base generation must not read a checkpoint"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "generate_one_video",
        lambda *args, **kwargs: torch.rand(3, 2, 8, 8),
    )

    def fake_write_mp4(video, path, *, fps):
        del video, fps
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")

    monkeypatch.setattr(checkpoint_eval, "write_mp4", fake_write_mp4)
    args = argparse.Namespace(
        run_dir=run_dir,
        output_dir=output_dir,
        target="base",
        limit=1,
        samples_per_prompt=1,
        base_seed=100,
        seed_stride=1_000,
        device="cpu",
    )

    result = checkpoint_eval.generate_shard(args)

    assert result["target"] == "base"
    assert result["videos"] == 1
    assert (output_dir / "generation/base/generated.jsonl").is_file()
