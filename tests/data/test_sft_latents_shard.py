"""The clean-latents shard contract (GRPO diffusion-loss regularizer data)."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.data.artifacts import load_sft_latents, save_sft_latents


def test_sft_latents_round_trip(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    latents = {
        "targets/fox.mp4": torch.randn(4, 2, 8, 8),
        "targets/car.mp4": torch.randn(4, 2, 8, 8),
    }
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="nvidia/x",
        model_revision="main",
        latents_by_target=latents,
    )

    loaded = load_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="nvidia/x",
        model_revision="main",
    )
    assert set(loaded) == set(latents)
    for target, value in latents.items():
        torch.testing.assert_close(loaded[target], value)
        assert loaded[target].device.type == "cpu"


def test_sft_latents_rejects_family_mismatch(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="x",
        model_revision="main",
        latents_by_target={"target.mp4": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match="not interchangeable"):
        load_sft_latents(shard, family="wan_2_1")


def test_sft_latents_rejects_model_mismatch(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="nvidia/source",
        model_revision="main",
        latents_by_target={"target.mp4": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match=r"model\.path"):
        load_sft_latents(
            shard,
            family="cosmos-predict2",
            model_path="nvidia/training",
            model_revision="main",
        )


def test_sft_latents_rejects_model_revision_mismatch(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    save_sft_latents(
        shard,
        family="cosmos-predict2.5",
        model_path="nvidia/model",
        model_revision="base",
        latents_by_target={"target.mp4": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match=r"model\.revision"):
        load_sft_latents(
            shard,
            family="cosmos-predict2.5",
            model_path="nvidia/model",
            model_revision="post-trained",
        )


def test_sft_latents_missing_file_names_the_producer(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="encode_targets"):
        load_sft_latents(tmp_path / "nope.pt")


def test_sft_latents_refuses_empty_shard(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_sft_latents(
            tmp_path / "s.pt",
            family="f",
            model_path="m",
            model_revision="",
            latents_by_target={},
        )


def test_sft_latents_rejects_foreign_payload(tmp_path) -> None:
    shard = tmp_path / "junk.pt"
    torch.save({"weights": torch.zeros(1)}, shard)
    with pytest.raises(ValueError, match="not an sft-latents shard"):
        load_sft_latents(shard)
