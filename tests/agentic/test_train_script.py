"""The chain training entry loads a vrl config and hands the chains to the online recipe."""

from __future__ import annotations

import json

from PIL import Image

from agentic.scripts import train_edit_chains


def test_smoke_config_and_chains_reach_the_recipe(tmp_path, monkeypatch) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "page.png")
    manifest = tmp_path / "chains.jsonl"
    manifest.write_text(
        json.dumps({"chain_id": "c", "source": "page.png", "steps": ["a", "b"]}) + "\n"
    )
    calls = []

    async def fake_recipe(cfg, *, prompt_examples):
        calls.append((cfg, prompt_examples))

    monkeypatch.setattr(train_edit_chains, "run_online_recipe", fake_recipe)
    train_edit_chains.main(
        [
            "--config",
            "experiment/qwen_image_21/edit_chains_smoke_single_gpu",
            "--chains",
            str(manifest),
            f"trainer.output_dir={tmp_path / 'run'}",
        ]
    )
    ((cfg, chains),) = calls
    assert cfg.rollout.n_samples_per_prompt == 2
    assert [chain.chain_id for chain in chains] == ["c"]
    assert chains[0].media_dir == tmp_path / "run" / "edit_chains"
    assert [step.prompt for step in chains[0].steps] == ["a", "b"]
