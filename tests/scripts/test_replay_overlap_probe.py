"""The recipe CLI runs a real update and writes a failed verdict on bad inputs."""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from tests.tiny_real_stack import TinySanaPipeline, tiny_sana_online_config
from vrl.scripts.perf import replay_overlap_probe


def test_cli_compares_real_recipe_updates_without_loading_a_reward(monkeypatch, tmp_path):
    cfg = tiny_sana_online_config(
        tmp_path,
        overrides=(
            "actor.training_microbatch_size=1",
            "actor.ppo_epochs=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    path = tmp_path / "recipe.yaml"
    OmegaConf.save(cfg, path)
    out = tmp_path / "probe"
    replay_overlap_probe.main(
        [
            "--config",
            str(path),
            "--artifact-dir",
            str(out),
            "--modes",
            "serial,deferred_wgrad",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ]
    )
    report = json.loads((out / "result.json").read_text())
    assert report["passed"]
    assert report["arms"]["deferred_wgrad"]["correctness_passed"]
    assert len(report["arms"]["serial"]["seconds"]) == 1
    assert report["reward_source"].startswith("synthetic")


def test_invalid_recipe_writes_a_failed_verdict(tmp_path):
    out = tmp_path / "bad"
    with pytest.raises(FileNotFoundError):
        replay_overlap_probe.main(
            ["--config", str(tmp_path / "missing.yaml"), "--artifact-dir", str(out)]
        )
    report = json.loads((out / "result.json").read_text())
    assert not report["passed"]
    assert "error" in report


@pytest.mark.gpu
def test_cuda_streams_and_checkpointed_dw_preserve_the_real_update(monkeypatch, tmp_path, recwarn):
    cfg = tiny_sana_online_config(
        tmp_path,
        overrides=(
            "actor.training_microbatch_size=1",
            "rollout.n_samples_per_prompt=4",
            "rollout.samples_per_generation_batch=4",
            "actor.ppo_epochs=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    # Only the text encoder stays on CPU; the real tiny DiT and trainer run
    # on CUDA through the same loader and placement code as a full recipe.
    cfg.model.memory = {"cpu_resident": ["text_encoder"]}
    path = tmp_path / "recipe.yaml"
    OmegaConf.save(cfg, path)
    out = tmp_path / "cuda"
    replay_overlap_probe.main(
        [
            "--config",
            str(path),
            "--artifact-dir",
            str(out),
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--correctness-only",
        ]
    )
    report = json.loads((out / "result.json").read_text())
    assert report["passed"]
    assert not report["timing_measured"]
    assert all(not arm["seconds"] for arm in report["arms"].values())
    assert set(report["arms"]) == {"serial", "streams", "deferred_wgrad"}
    assert report["arms"]["streams"]["schedule"]["forward_calls"] >= 2
    assert report["arms"]["streams"]["schedule"]["private_trainable_leaves"] > 0
    assert not any("AccumulateGrad" in str(w.message) for w in recwarn)
    assert report["update_device"].startswith("cuda")


def test_nonzero_lora_survives_rollout_to_replay_and_delayed_dw(monkeypatch, tmp_path):
    cfg = tiny_sana_online_config(
        tmp_path,
        overrides=(
            "model.use_lora=true",
            "actor.ppo_epochs=1",
            "actor.training_microbatch_size=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    path = tmp_path / "lora.yaml"
    OmegaConf.save(cfg, path)
    out = tmp_path / "lora-probe"
    replay_overlap_probe.main(
        [
            "--config",
            str(path),
            "--artifact-dir",
            str(out),
            "--modes",
            "serial,deferred_wgrad",
            "--lora-b-std",
            "0.01",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--trace",
        ]
    )
    report = json.loads((out / "result.json").read_text())
    assert report["passed"]
    deferred = report["arms"]["deferred_wgrad"]["schedule"]["deferred_linears"]
    assert deferred and all("lora_" in name for name in deferred)
    assert (out / "deferred_wgrad.trace.json").is_file()


def test_probe_honors_recipe_rng_policy_before_model_construction(monkeypatch, tmp_path):
    import random

    import numpy as np
    import torch

    cfg = tiny_sana_online_config(
        tmp_path,
        overrides=(
            "trainer.seed=123",
            "trainer.deterministic=true",
            "actor.ppo_epochs=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    path = tmp_path / "deterministic.yaml"
    OmegaConf.save(cfg, path)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def before_build(args, root):
        assert torch.are_deterministic_algorithms_enabled()
        torch.testing.assert_close(
            torch.rand(3), torch.rand(3, generator=torch.Generator().manual_seed(123))
        )
        np.testing.assert_array_equal(np.random.rand(3), np.random.RandomState(123).rand(3))
        assert random.random() == random.Random(123).random()
        raise RuntimeError("stop before model construction")

    monkeypatch.setattr(replay_overlap_probe, "prepare_inputs", before_build)
    try:
        with pytest.raises(RuntimeError, match="stop before model construction"):
            replay_overlap_probe.main(
                ["--config", str(path), "--artifact-dir", str(tmp_path / "probe")]
            )
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
