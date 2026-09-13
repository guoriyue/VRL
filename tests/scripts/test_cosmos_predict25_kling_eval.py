from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.scripts.eval.fixtures import cosmos25_eval_config, write_tiny_cosmos25_snapshot
from tests.trainers._checkpoint_helpers import _Trainer
from vrl.config.builders import RewardRuntimeConfig
from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.eval import cosmos_predict25_kling_eval as eval_script
from vrl.trainers.checkpointing import CheckpointTarget, save_training_checkpoint

MODEL_IDENTITY = {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}


def _minimal_eval_config(*, family: str = "cosmos-predict2.5"):
    return OmegaConf.create(
        {
            "model": {"family": family, "path": "org/model"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16"},
            },
            "sampling": {
                "width": 8,
                "height": 8,
                "num_frames": 9,
                "num_steps": 1,
                "fps": 16,
                "max_sequence_length": 8,
                "guidance_scale": 4.5,
            },
            "trainer": {},
        },
    )


def test_parse_checkpoint_accepts_label_and_path(tmp_path) -> None:
    """Checks checkpoint CLI values can carry stable labels."""
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()

    (target,) = CheckpointTarget.from_cli_values(
        [f"baseline={checkpoint}"], require_directory=False
    )

    assert target.label == "baseline"
    assert target.path == checkpoint.resolve()


def test_seed_grid_cell_is_identical_across_checkpoints(tmp_path) -> None:
    """The generator must derive each seed from the (prompt, sample) cell only.

    ``target`` is in scope inside the generation loop, so folding the checkpoint
    label into the seed is one live edit away -- and it would silently turn every
    reward delta into a different latent-noise draw instead of a weight effect.
    That edit is invisible one level down in ``seed_for``, which never sees a
    checkpoint at all, so the claim is driven through the real loop, real tiny
    generation and a real mp4 encode.
    """

    _snapshot, _config_path, root, entry, build, _identity = _tiny_cosmos_run(tmp_path)
    model = entry.build_rollout(build).model.eval()
    sampling = eval_script._resolve_sampling(
        eval_script.build_parser().parse_args(["--checkpoint", "unused"]), root
    )

    def run(label: str) -> list[int]:
        videos = eval_script._generate_checkpoint_videos(
            model,
            eval_script.CheckpointTarget(label, tmp_path / label),
            ["p0", "p1"],
            samples_per_prompt=2,
            base_seed=17,
            output_dir=tmp_path / label,
            sampling=sampling,
        )
        assert all(video.path.is_file() and video.path.stat().st_size > 0 for video in videos)
        return [video.seed for video in videos]

    base, trained = run("base"), run("a-much-longer-label")
    assert base == trained
    assert len(set(base)) == 4  # non-degeneracy: four cells, four distinct seeds


def test_reward_worker_config_adds_reward_model_name_default() -> None:
    """Checks direct Kling scorer gets the model name from reward config."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "reward_name": "KlingTeam/VideoReward@main",
                        "worker_config": {"local_files_only": True},
                    },
                },
            },
        },
    )

    worker_config = RewardRuntimeConfig.from_cfg(cfg).worker_config(
        "kling_video_reward",
        default_reward_model_name="KlingTeam/VideoReward@main",
    )

    assert worker_config["local_files_only"] is True
    assert worker_config["reward_model_name"] == "KlingTeam/VideoReward@main"


def test_score_summary_groups_by_checkpoint() -> None:
    """Checks summary statistics are grouped by checkpoint label."""
    rows = [
        {"checkpoint_label": "base", "selected_score": 1.0},
        {"checkpoint_label": "base", "selected_score": 3.0},
        {"checkpoint_label": "trained", "selected_score": 5.0},
    ]

    summary = eval_script._summarize_scores(rows)

    assert summary["base"]["mean"] == 2.0
    assert summary["base"]["count"] == 2
    assert summary["trained"]["mean"] == 5.0


def _video_root(**sampling: object):
    """A parsed cosmos config declaring every key the eval projection carries."""
    return parse_config(
        OmegaConf.create(
            {
                "model": {"family": "cosmos-predict2.5"},
                "sampling": {
                    "width": 8,
                    "height": 8,
                    "num_frames": 9,
                    "num_steps": 1,
                    "fps": 16,
                    "max_sequence_length": 8,
                    **sampling,
                },
            },
        ),
    )


def test_eval_sampling_inherits_guidance_when_cli_omits_it() -> None:
    """An omitted guidance flag inherits the merged sampling config."""
    root = _video_root(guidance_scale=4.0)
    args = eval_script.build_parser().parse_args(["--checkpoint", "unused"])

    sampling = eval_script._resolve_sampling(args, root)

    assert sampling["guidance_scale"] == 4.0


def test_eval_sampling_preserves_explicit_zero_guidance() -> None:
    """An explicit zero disables CFG instead of falling back to the config."""
    root = _video_root(guidance_scale=4.0)
    args = eval_script.build_parser().parse_args(
        ["--checkpoint", "unused", "--guidance-scale", "0"],
    )

    sampling = eval_script._resolve_sampling(args, root)

    assert sampling["guidance_scale"] == 0.0


def _tiny_cosmos_run(tmp_path: Path):
    """A real tiny Cosmos-2.5 snapshot, its resolved config on disk, and the real entry/build."""

    snapshot = write_tiny_cosmos25_snapshot(tmp_path / "cosmos-snapshot")
    config_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(cosmos25_eval_config(snapshot), config_path)
    root = parse_config(load_config(config_path))
    entry = get_model_family_entry("cosmos-predict2.5")
    build = entry.resolve_model_build(
        root,
        torch.device("cpu"),
        precision=PrecisionPolicy.from_section(root.precision),
        for_rollout=True,
    )
    identity = resolve_checkpoint_model_identity(build)
    return snapshot, config_path, root, entry, build, identity


def _save_real_checkpoint(path: Path, *, entry, build, identity, fill: float) -> Path:
    """Save a strict checkpoint from the real bundle with every LoRA weight set to ``fill``."""

    bundle = entry.build_rollout(build)
    with torch.no_grad():
        for parameter in bundle.trainable_modules["transformer"].parameters():
            if parameter.requires_grad:
                parameter.fill_(fill)
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=bundle,
        family="cosmos-predict2.5",
        progress={"next_epoch": 1},
        rng_state={},
        model_identity=identity,
    )
    return path


def _spy_from_build(monkeypatch, on_built=None):
    """Record every real CosmosPredict25Model.from_build call (a spy, not a fake)."""

    from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model

    real = CosmosPredict25Model.from_build.__func__
    built: list[weakref.ReferenceType] = []

    def from_build(cls, build):
        model = real(cls, build)
        built.append(weakref.ref(model))
        if on_built is not None:
            on_built(model)
        return model

    monkeypatch.setattr(CosmosPredict25Model, "from_build", classmethod(from_build))
    return built


def test_explicit_dtype_path_runs_after_structural_validation(monkeypatch, tmp_path) -> None:
    """``--dtype fp32`` reaches the real build, and generation runs on it for real."""

    _snapshot, config_path, _root, entry, build, identity = _tiny_cosmos_run(tmp_path)
    checkpoint = _save_real_checkpoint(
        tmp_path / "checkpoint-final", entry=entry, build=build, identity=identity, fill=1.0
    )
    seen_dtypes: list[torch.dtype] = []
    real_generate_all = eval_script._generate_all

    def spy_generate_all(build, *args, **kwargs):
        seen_dtypes.append(build.parameter_dtype)
        return real_generate_all(build, *args, **kwargs)

    monkeypatch.setattr(eval_script, "_generate_all", spy_generate_all)
    output_dir = tmp_path / "output"

    eval_script.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "a world-model test",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--samples-per-prompt",
            "1",
            "--generate-only",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert seen_dtypes == [torch.float32]
    videos = sorted(output_dir.rglob("*.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 0
    assert (output_dir / "run_config.json").is_file()


def test_explicit_dtype_rejects_malformed_model_family_before_generation(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    monkeypatch.setattr(
        eval_script,
        "load_config",
        lambda *_args, **_kwargs: _minimal_eval_config(family="not-a-family"),
    )
    monkeypatch.setattr(
        eval_script,
        "_generate_all",
        lambda *_args, **_kwargs: pytest.fail("generation started before config validation"),
    )

    with pytest.raises(ValueError, match="unsupported model family"):
        eval_script.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                "a world-model test",
                "--dtype",
                "fp32",
                "--generate-only",
                "--output-dir",
                str(tmp_path / "output"),
            ],
        )


def test_checkpoint_identity_rejects_before_model_generation(monkeypatch, tmp_path) -> None:
    _snapshot, config_path, *_ = _tiny_cosmos_run(tmp_path)
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "family": "cosmos-predict2.5",
                "model_identity": {"schema": "wrong/v1"},
            },
        ),
    )
    built = _spy_from_build(monkeypatch)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        eval_script.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                "a world-model test",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--generate-only",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert built == []
    assert not output_dir.exists()


def test_generate_all_releases_model_before_rebuilding(monkeypatch, tmp_path) -> None:
    """Without model reuse, each checkpoint gets a fresh real bundle and the old model is gone."""

    _snapshot, _config_path, root, entry, build, identity = _tiny_cosmos_run(tmp_path)
    targets = [
        eval_script.CheckpointTarget(
            label,
            _save_real_checkpoint(
                tmp_path / label, entry=entry, build=build, identity=identity, fill=fill
            ),
        )
        for label, fill in (("base", 1.0), ("trained", 2.0))
    ]
    monkeypatch.setattr(eval_script, "release_cuda_memory", gc.collect)

    def previous_model_is_gone(_model) -> None:
        gc.collect()
        assert all(ref() is None for ref in built[:-1])

    built = _spy_from_build(monkeypatch, on_built=previous_model_is_gone)
    restored: list[tuple[Path, float]] = []
    real_restore = eval_script.restore_model_checkpoint

    def spy_restore(checkpoint, *, bundle, **kwargs):
        real_restore(checkpoint, bundle=bundle, **kwargs)
        weight = next(
            p for p in bundle.trainable_modules["transformer"].parameters() if p.requires_grad
        )
        restored.append((checkpoint.checkpoint_dir, float(weight.flatten()[0])))

    monkeypatch.setattr(eval_script, "restore_model_checkpoint", spy_restore)
    sampling = eval_script._resolve_sampling(
        eval_script.build_parser().parse_args(["--checkpoint", "unused"]), root
    )

    videos = eval_script._generate_all(
        build,
        targets,
        ["prompt"],
        samples_per_prompt=1,
        base_seed=0,
        output_dir=tmp_path / "videos",
        sampling=sampling,
        keep_model_between_checkpoints=False,
        expected_model_identity=identity,
    )

    # Two real builds, each checkpoint strictly restored into its own bundle
    # (the LoRA weights it wrote are the ones the model then generated with).
    assert len(built) == 2
    assert restored == [(tmp_path / "base", 1.0), (tmp_path / "trained", 2.0)]
    assert [video.checkpoint_label for video in videos] == ["base", "trained"]
    assert all(video.path.is_file() and video.path.stat().st_size > 0 for video in videos)
    gc.collect()
    assert [ref() for ref in built] == [None, None]


def test_generate_all_rejects_model_source_drift_before_checkpoint_load(
    monkeypatch,
    tmp_path,
) -> None:
    """A model directory that changes while the bundle is built is caught before any checkpoint."""

    snapshot, _config_path, root, _entry, build, identity = _tiny_cosmos_run(tmp_path)

    def drift(_model) -> None:
        (snapshot / "extra-weights.bin").write_bytes(b"drift")

    built = _spy_from_build(monkeypatch, on_built=drift)
    checkpoint_loads: list[Path] = []
    monkeypatch.setattr(
        eval_script.TrainingCheckpoint,
        "load",
        classmethod(lambda _cls, path: checkpoint_loads.append(Path(path))),
    )
    sampling = eval_script._resolve_sampling(
        eval_script.build_parser().parse_args(["--checkpoint", "unused"]), root
    )

    with pytest.raises(
        RuntimeError, match="Cosmos model source changed during runtime construction"
    ):
        eval_script._generate_all(
            build,
            [eval_script.CheckpointTarget("trained", tmp_path / "checkpoint")],
            ["prompt"],
            samples_per_prompt=1,
            base_seed=0,
            output_dir=tmp_path / "videos",
            sampling=sampling,
            keep_model_between_checkpoints=True,
            expected_model_identity=identity,
        )

    assert len(built) == 1
    assert checkpoint_loads == []
