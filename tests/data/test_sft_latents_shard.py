"""The clean-latents shard contract (GRPO diffusion-loss regularizer data)."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.data.artifacts import load_sft_latents, save_sft_latents


def test_sft_latents_round_trip(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    latents = {"a red fox": torch.randn(4, 2, 8, 8), "a blue car": torch.randn(4, 2, 8, 8)}
    save_sft_latents(shard, family="cosmos-predict2", model_path="nvidia/x", latents_by_prompt=latents)

    loaded = load_sft_latents(shard, family="cosmos-predict2")
    assert set(loaded) == set(latents)
    for prompt, value in latents.items():
        torch.testing.assert_close(loaded[prompt], value)
        assert loaded[prompt].device.type == "cpu"


def test_sft_latents_rejects_family_mismatch(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    save_sft_latents(shard, family="cosmos-predict2", model_path="x", latents_by_prompt={"p": torch.zeros(1)})
    with pytest.raises(ValueError, match="not interchangeable"):
        load_sft_latents(shard, family="wan_2_1")


def test_sft_latents_missing_file_names_the_producer(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="encode_targets"):
        load_sft_latents(tmp_path / "nope.pt")


def test_sft_latents_refuses_empty_shard(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_sft_latents(tmp_path / "s.pt", family="f", model_path="m", latents_by_prompt={})


def test_sft_latents_rejects_foreign_payload(tmp_path) -> None:
    shard = tmp_path / "junk.pt"
    torch.save({"weights": torch.zeros(1)}, shard)
    with pytest.raises(ValueError, match="not an sft-latents shard"):
        load_sft_latents(shard)
