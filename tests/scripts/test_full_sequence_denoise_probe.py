from __future__ import annotations

import json

from omegaconf import OmegaConf

from tests.tiny_real_stack import TinySanaPipeline, tiny_sana_online_config
from vrl.scripts.generation import full_sequence_probe


def test_probe_replays_a_recipe_rollout_at_the_trainer_batch(monkeypatch, tmp_path) -> None:
    # The SANA online recipe on tiny real modules: fp32, two steps, generation
    # batch 2 replayed as batch 1, so replay runs a different batch shape than
    # the rollout through the production loop, executor hooks and SDE math.
    cfg = tiny_sana_online_config(tmp_path, overrides=("actor.training_microbatch_size=1",))
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    config = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, config)

    full_sequence_probe.main(["--config", str(config), "--artifact-dir", str(tmp_path / "probe")])

    report = json.loads((tmp_path / "probe" / "result.json").read_text())
    assert report["passed"]
    assert (report["rollout_batch"], report["replay_batch"]) == (2, 1)
    assert [step["step"] for step in report["steps"]] == [0, 1]
    assert report["max_logprob_abs"] <= report["max_abs_logprob_diff_gate"]
    assert "first_divergent_module" in report["steps"][0]
    assert (tmp_path / "probe" / "image.png").is_file()
