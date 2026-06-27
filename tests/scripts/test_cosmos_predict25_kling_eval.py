from __future__ import annotations

import argparse
import gc
import weakref

import pytest
import torch
from omegaconf import OmegaConf

from vrl.scripts.eval import cosmos_predict25_kling_eval as eval_script


def test_parse_checkpoint_accepts_label_and_path(tmp_path) -> None:
    """Checks checkpoint CLI values can carry stable labels."""
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()

    spec = eval_script._parse_checkpoint_spec(f"baseline={checkpoint}")

    assert spec.label == "baseline"
    assert spec.path == checkpoint.resolve()


def test_seed_grid_is_identical_across_checkpoints() -> None:
    """Eval seeds depend only on the (prompt, sample) cell, never the checkpoint,
    so reward deltas reflect weight changes and not a different latent-noise draw —
    yet distinct cells still get distinct seeds (the grid is not degenerate)."""

    def seed(checkpoint_index: int, prompt_index: int, sample_index: int) -> int:
        return eval_script._seed_for(
            base_seed=17,
            checkpoint_index=checkpoint_index,
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


def test_generate_all_releases_model_before_rebuilding(monkeypatch, tmp_path) -> None:
    """Checks non-reused checkpoint eval does not keep old models alive."""

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

    class FakeBundle:
        def __init__(self) -> None:
            self.model = FakeModel()

    model_refs: list[weakref.ReferenceType[FakeModel]] = []

    def fake_build_runtime_bundle(_spec):
        gc.collect()
        if model_refs:
            assert model_refs[-1]() is None
        bundle = FakeBundle()
        model_refs.append(weakref.ref(bundle.model))
        return bundle

    monkeypatch.setattr(eval_script, "_release_cuda", gc.collect)
    monkeypatch.setattr(
        eval_script,
        "extract_cosmos_predict25_runtime_spec",
        lambda cfg, device, dtype: object(),
    )
    monkeypatch.setattr(
        eval_script,
        "build_cosmos_predict25_runtime_bundle",
        fake_build_runtime_bundle,
    )
    monkeypatch.setattr(eval_script, "_load_checkpoint_into_bundle", lambda bundle, checkpoint: None)
    monkeypatch.setattr(eval_script, "_generate_checkpoint_videos", lambda *args, **kwargs: [])

    videos = eval_script._generate_all(
        OmegaConf.create({}),
        [
            eval_script.CheckpointSpec("base", tmp_path / "base"),
            eval_script.CheckpointSpec("trained", tmp_path / "trained"),
        ],
        ["prompt"],
        samples_per_prompt=1,
        base_seed=0,
        output_dir=tmp_path,
        sampling={},
        device=torch.device("cpu"),
        dtype=torch.float32,
        keep_model_between_checkpoints=False,
    )

    assert videos == []
    gc.collect()
    assert [ref() for ref in model_refs] == [None, None]
