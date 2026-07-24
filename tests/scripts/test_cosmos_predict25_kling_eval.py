from __future__ import annotations

import argparse
import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import RolePrecision
from vrl.scripts.eval import cosmos_predict25_kling_eval as eval_script

MODEL_IDENTITY = {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}


def _minimal_eval_config(*, family: str = "cosmos-predict2.5"):
    return OmegaConf.create(
        {
            "model": {"family": family, "path": "org/model"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16"},
            },
            "trainer": {},
        },
    )


def test_parse_checkpoint_accepts_label_and_path(tmp_path) -> None:
    """Checks checkpoint CLI values can carry stable labels."""
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()

    target = eval_script._parse_checkpoint_target(f"baseline={checkpoint}")

    assert target.label == "baseline"
    assert target.path == checkpoint.resolve()


def test_seed_grid_is_identical_across_checkpoints() -> None:
    """Eval seeds depend only on the (prompt, sample) cell, never the checkpoint,
    so reward deltas reflect weight changes and not a different latent-noise draw —
    yet distinct cells still get distinct seeds (the grid is not degenerate)."""

    def seed(checkpoint_index: int, prompt_index: int, sample_index: int) -> int:
        del checkpoint_index
        return eval_script._seed_for(
            base_seed=17,
            prompt_index=prompt_index,
            sample_index=sample_index,
            samples_per_prompt=4,
        )

    # Same cell across different checkpoints -> identical seed (the real contract).
    assert seed(0, 2, 1) == seed(3, 2, 1)

    # Non-degeneracy: different sample / prompt cells must not collide, otherwise a
    # constant function would also satisfy the checkpoint-independence assert above.
    assert seed(0, 2, 1) != seed(0, 2, 2)
    assert seed(0, 2, 1) != seed(0, 3, 1)


def test_video_to_cthw_accepts_btchw_layout() -> None:
    """Checks decoded Cosmos video layout is normalized for mp4 writing."""
    video = torch.zeros(1, 5, 3, 8, 8)
    video[:, :, 0] = 1.0

    out = eval_script._video_to_cthw(video)

    assert tuple(out.shape) == (3, 5, 8, 8)
    assert torch.all(out[0] == 1.0)


def test_reward_worker_config_adds_reward_model_name_default() -> None:
    """Checks direct Kling scorer gets the model name from reward config."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "kwargs": {
                    "kling_video_reward": {
                        "reward_name": "KlingTeam/VideoReward@main",
                        "worker_config": {"local_files_only": True},
                    },
                },
            },
        },
    )

    worker_config = eval_script._reward_worker_config(cfg)

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


def test_checkpoint_eval_reuses_model_by_default() -> None:
    """Checks single-GPU checkpoint eval avoids repeated full pipeline loads."""
    args = argparse.Namespace(
        keep_model_between_checkpoints=False,
        rebuild_model_between_checkpoints=False,
    )

    assert eval_script._keep_model_between_checkpoints(args) is True


def test_checkpoint_eval_rejects_conflicting_lifecycle_flags() -> None:
    """Checks checkpoint eval lifecycle flags cannot disagree."""
    args = argparse.Namespace(
        keep_model_between_checkpoints=True,
        rebuild_model_between_checkpoints=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        eval_script._keep_model_between_checkpoints(args)


def test_eval_sampling_inherits_guidance_when_cli_omits_it() -> None:
    """An omitted guidance flag inherits the merged sampling config."""
    cfg = OmegaConf.create({"sampling": {"guidance_scale": 4.0}})
    args = eval_script.build_parser().parse_args(["--checkpoint", "unused"])

    sampling = eval_script._resolve_sampling(args, cfg)

    assert sampling["guidance_scale"] == 4.0


def test_eval_sampling_preserves_explicit_zero_guidance() -> None:
    """An explicit zero disables CFG instead of falling back to the config."""
    cfg = OmegaConf.create({"sampling": {"guidance_scale": 4.0}})
    args = eval_script.build_parser().parse_args(
        ["--checkpoint", "unused", "--guidance-scale", "0"],
    )

    sampling = eval_script._resolve_sampling(args, cfg)

    assert sampling["guidance_scale"] == 0.0


def test_explicit_dtype_path_runs_after_structural_validation(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        eval_script,
        "load_config",
        lambda *_args, **_kwargs: _minimal_eval_config(),
    )
    entry = SimpleNamespace(
        family="cosmos-predict2.5",
        resolve_model_build=lambda _root, _device, **kwargs: SimpleNamespace(
            family="cosmos-predict2.5",
            parameter_dtype=kwargs["parameter_dtype_override"],
        ),
    )
    monkeypatch.setattr(eval_script, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        eval_script,
        "resolve_checkpoint_model_identity",
        lambda _build: MODEL_IDENTITY,
    )

    def fake_generate_all(build, *_args, **_kwargs):
        captured["dtype"] = build.parameter_dtype
        return []

    monkeypatch.setattr(eval_script, "_generate_all", fake_generate_all)

    eval_script.main(
        [
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
            str(tmp_path / "output"),
        ],
    )

    assert captured["dtype"] is torch.float32


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
    monkeypatch.setattr(
        eval_script,
        "load_config",
        lambda *_args, **_kwargs: _minimal_eval_config(),
    )
    build = SimpleNamespace(family="cosmos-predict2.5")
    entry = SimpleNamespace(
        family="cosmos-predict2.5",
        resolve_model_build=lambda *_args, **_kwargs: build,
    )
    monkeypatch.setattr(eval_script, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        eval_script,
        "resolve_checkpoint_model_identity",
        lambda _build: MODEL_IDENTITY,
    )
    monkeypatch.setattr(
        eval_script,
        "_generate_all",
        lambda *_args, **_kwargs: pytest.fail("generation started before identity preflight"),
    )
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        eval_script.main(
            [
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

    assert not output_dir.exists()


def test_generate_all_releases_model_before_rebuilding(monkeypatch, tmp_path) -> None:
    """Checks non-reused checkpoint eval does not keep old models alive."""

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

    class FakeBundle:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.precision = RolePrecision(
                dtype="fp32",
                float32_precision="ieee",
                outer_autocast=False,
            )
            self.model.precision = self.precision

    model_refs: list[weakref.ReferenceType[FakeModel]] = []
    bundle_ids: list[int] = []

    def fake_build_runtime_bundle(_build):
        gc.collect()
        if model_refs:
            assert model_refs[-1]() is None
        bundle = FakeBundle()
        model_refs.append(weakref.ref(bundle.model))
        bundle_ids.append(id(bundle))
        return bundle

    monkeypatch.setattr(eval_script, "_release_cuda", gc.collect)
    entry = SimpleNamespace(
        family="cosmos-predict2.5",
        build_rollout=fake_build_runtime_bundle,
    )
    monkeypatch.setattr(eval_script, "get_model_family_entry", lambda _family: entry)
    checkpoints = {}

    def fake_load_checkpoint(path):
        checkpoint = object()
        checkpoints[path] = checkpoint
        return checkpoint

    monkeypatch.setattr(
        eval_script,
        "load_training_checkpoint",
        fake_load_checkpoint,
    )
    restored = []

    def fake_restore(
        checkpoint,
        *,
        bundle,
        family,
        expected_model_identity,
        strict,
    ):
        restored.append(
            (
                checkpoint,
                id(bundle),
                family,
                expected_model_identity,
                strict,
            ),
        )

    monkeypatch.setattr(eval_script, "restore_model_checkpoint", fake_restore)
    monkeypatch.setattr(
        eval_script,
        "resolve_checkpoint_model_identity",
        lambda _build: MODEL_IDENTITY,
    )
    monkeypatch.setattr(eval_script, "_generate_checkpoint_videos", lambda *args, **kwargs: [])

    build = SimpleNamespace(family="cosmos-predict2.5")
    videos = eval_script._generate_all(
        build,
        [
            eval_script.CheckpointTarget("base", tmp_path / "base"),
            eval_script.CheckpointTarget("trained", tmp_path / "trained"),
        ],
        ["prompt"],
        samples_per_prompt=1,
        base_seed=0,
        output_dir=tmp_path,
        sampling={},
        keep_model_between_checkpoints=False,
        expected_model_identity=MODEL_IDENTITY,
    )

    assert videos == []
    assert list(checkpoints) == [tmp_path / "base", tmp_path / "trained"]
    assert [
        (
            checkpoint,
            family,
            expected_model_identity,
            strict,
        )
        for checkpoint, _bundle_id, family, expected_model_identity, strict in restored
    ] == [
        (checkpoints[tmp_path / "base"], "cosmos-predict2.5", MODEL_IDENTITY, True),
        (checkpoints[tmp_path / "trained"], "cosmos-predict2.5", MODEL_IDENTITY, True),
    ]
    assert [bundle_id for _checkpoint, bundle_id, *_rest in restored] == bundle_ids
    gc.collect()
    assert [ref() for ref in model_refs] == [None, None]


def test_generate_all_rejects_model_source_drift_before_checkpoint_load(
    monkeypatch,
    tmp_path,
) -> None:
    build = SimpleNamespace(family="cosmos-predict2.5")
    built = False
    checkpoint_loaded = False

    def build_rollout(actual_build):
        nonlocal built
        assert actual_build is build
        built = True
        return object()

    entry = SimpleNamespace(
        family="cosmos-predict2.5",
        build_rollout=build_rollout,
    )
    monkeypatch.setattr(eval_script, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        eval_script,
        "resolve_checkpoint_model_identity",
        lambda _build: {"schema": "changed"},
    )

    def fail_if_checkpoint_loaded(_path):
        nonlocal checkpoint_loaded
        checkpoint_loaded = True
        raise AssertionError("checkpoint load must not run after model source drift")

    monkeypatch.setattr(eval_script, "load_training_checkpoint", fail_if_checkpoint_loaded)

    with pytest.raises(
        RuntimeError,
        match="Cosmos model source changed during runtime construction",
    ):
        eval_script._generate_all(
            build,
            [eval_script.CheckpointTarget("trained", tmp_path / "checkpoint")],
            ["prompt"],
            samples_per_prompt=1,
            base_seed=0,
            output_dir=tmp_path,
            sampling={},
            keep_model_between_checkpoints=True,
            expected_model_identity=MODEL_IDENTITY,
        )

    assert built is True
    assert checkpoint_loaded is False
