from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.scripts.eval.fixtures import (
    WAN_TINY_SAMPLING,
    write_prompt_manifest,
    write_tiny_wan_snapshot,
)
from tests.trainers._checkpoint_helpers import _Trainer
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.eval import wan_robotics_checkpoint_eval as checkpoint_eval
from vrl.trainers.checkpointing import save_training_checkpoint
from vrl.trainers.data.prompts import PromptExample


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


def _tiny_wan_run(tmp_path: Path) -> Path:
    """A full-parameter Wan run directory: tiny snapshot, disjoint train/eval manifests,
    and the resolved config ``_load_run`` reads -- everything the generator needs on disk."""

    snapshot = write_tiny_wan_snapshot(tmp_path / "wan-snapshot")
    train = write_prompt_manifest(
        tmp_path / "manifests" / "train.jsonl",
        [
            {
                "prompt": "move the cup",
                "target_video": "targets/train.mp4",
                "metadata": {
                    "source_repo": "robot/data",
                    "source_episode": "train-1",
                    "source_video": "file-0.mp4",
                    "source_frame_index": 0,
                },
            },
        ],
    )
    evaluation = write_prompt_manifest(
        tmp_path / "manifests" / "eval.jsonl",
        [
            {
                "prompt": "pick up the bowl",
                "target_video": "targets/eval.mp4",
                "metadata": {
                    "source_repo": "robot/data",
                    "source_episode": "eval-1",
                    "source_video": "file-1.mp4",
                    "source_frame_index": 0,
                },
            },
        ],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "family": "wan",
                    "path": str(snapshot),
                    "revision": None,
                    "use_lora": False,
                    "torch_compile": {"enable": False},
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                    "rollout": {"dtype": "fp32"},
                },
                "sampling": dict(WAN_TINY_SAMPLING),
                "rollout": {"denoise_mode": "native", "noise_level": 0.0, "sde": {"type": "cps"}},
                "data": {
                    "loader": "prompt_manifest",
                    "manifest": str(train),
                    "eval_manifest": str(evaluation),
                    "artifact_data_root": str(tmp_path),
                    "task_type": "text2video",
                    "preprocessing": {"format": "jsonl"},
                    "sampler": {"type": "random_without_replacement"},
                },
            },
        ),
        run_dir / "resolved_config.yaml",
    )
    return run_dir


def _generate_args(run_dir: Path, output_dir: Path, target: str) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=run_dir,
        output_dir=output_dir,
        target=target,
        limit=1,
        samples_per_prompt=1,
        base_seed=100,
        seed_stride=1_000,
        device="cpu",
    )


def _save_epoch_checkpoint(run_dir: Path, *, epoch: int, fill: float) -> Path:
    """A real full-parameter checkpoint of the run's model with every weight set to ``fill``."""

    _run_dir, cfg, _config_path = checkpoint_eval._load_run(run_dir)
    root = parse_config(cfg)
    entry = get_model_family_entry("wan_2_1")
    build = entry.resolve_model_build(
        root,
        torch.device("cpu"),
        precision=PrecisionPolicy.from_section(root.precision),
        for_rollout=True,
    )
    bundle = entry.build_rollout(build)
    with torch.no_grad():
        for parameter in bundle.trainable_modules["transformer"].parameters():
            parameter.fill_(fill)
    path = run_dir / f"checkpoint-{epoch}"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=bundle,
        family="wan_2_1",
        progress={"completed_epoch": epoch, "next_epoch": epoch},
        rng_state={},
        model_identity=resolve_checkpoint_model_identity(build),
    )
    return path


def test_base_generation_never_reads_a_training_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--target base`` runs the real resolve -> build -> generate -> mp4 chain on the
    snapshot weights, with the checkpoint loader as a red line."""

    run_dir = _tiny_wan_run(tmp_path)
    monkeypatch.setattr(
        checkpoint_eval.TrainingCheckpoint,
        "load",
        classmethod(lambda _cls, path: pytest.fail("base generation must not read a checkpoint")),
    )

    result = checkpoint_eval.generate_shard(_generate_args(run_dir, tmp_path / "eval", "base"))

    shard = tmp_path / "eval" / "generation" / "base"
    assert result == {
        "output_dir": str(shard),
        "protocol_sha256": result["protocol_sha256"],
        "target": "base",
        "videos": 1,
    }
    (row,) = [json.loads(line) for line in (shard / "generated.jsonl").read_text().splitlines()]
    video = Path(row["path"])
    assert video.is_file() and row["sha256"] == checkpoint_eval.sha256_file(video)
    assert row["prompt"] == "pick up the bowl"
    provenance = json.loads((shard / "provenance.json").read_text())
    assert provenance["target"] == {
        "label": "base",
        "epoch": 0,
        "path": None,
        "checkpoint_loaded": False,
        "checkpoint_meta": {},
    }


def test_checkpoint_target_generates_with_the_loaded_weights(tmp_path: Path) -> None:
    """A numbered target loads the real checkpoint into the real bundle: the same
    seed grid renders a different video than base only because the weights changed."""

    run_dir = _tiny_wan_run(tmp_path)
    _save_epoch_checkpoint(run_dir, epoch=1, fill=0.25)

    base = checkpoint_eval.generate_shard(_generate_args(run_dir, tmp_path / "eval", "base"))
    trained = checkpoint_eval.generate_shard(_generate_args(run_dir, tmp_path / "eval", "1"))

    assert trained["target"] == "checkpoint-1"
    assert trained["protocol_sha256"] == base["protocol_sha256"]

    def shard_row(result):
        (row,) = [
            json.loads(line)
            for line in (Path(result["output_dir"]) / "generated.jsonl").read_text().splitlines()
        ]
        return row

    base_row, trained_row = shard_row(base), shard_row(trained)
    assert base_row["seed"] == trained_row["seed"]
    assert base_row["sha256"] != trained_row["sha256"]
    provenance = json.loads((Path(trained["output_dir"]) / "provenance.json").read_text())
    assert provenance["target"]["checkpoint_loaded"] is True
    assert provenance["target"]["epoch"] == 1
