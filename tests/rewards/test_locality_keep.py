"""Outside-keep measures only the pixels away from the declared boxes."""

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.locality_keep import LocalityKeepRewardModel
from vrl.rewards.types import RewardSample


@pytest.mark.asyncio
async def test_changes_inside_the_boxes_are_free_and_outside_changes_are_charged(tmp_path):
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(source).save(path)
    spec = {"source": str(path), "boxes": [[0, 0, 48, 32]]}
    inside = source.copy()
    inside[:32, :48] = 0
    outside = source.copy()
    outside[40:, 60:] = (outside[40:, 60:] // 2).astype(np.uint8)
    noisy = np.clip(source.astype(int) + rng.integers(-40, 41, source.shape), 0, 255).astype(
        np.uint8
    )
    model = LocalityKeepRewardModel({})
    same = model(
        RewardInferenceArtifact(
            "a", "a", "", media=Image.fromarray(inside), metadata={"locality_keep": spec}
        )
    )
    assert (
        same["locality_psnr"] == 60
        and same["locality_keep"] == 1
        and same["locality_hf_ratio"] == 1
    )
    Image.fromarray(outside).save(tmp_path / "x.png")
    changed = model(
        RewardInferenceArtifact(
            "b", "b", str(tmp_path / "x.png"), metadata={"locality_keep": spec}
        )
    )
    assert changed["locality_psnr"] < 30 and changed["locality_keep"] < 1
    textured = model(
        RewardInferenceArtifact(
            "c", "c", "", media=Image.fromarray(noisy), metadata={"locality_keep": spec}
        )
    )
    assert textured["locality_hf_ratio"] > 1 and textured["locality_keep"] < 1
    # Blurring a drifted candidate raises its PSNR; the key must still fall.
    # A smooth photo-like source: the blur removes the drift noise, as on real pages.
    import cv2

    yy, xx = np.mgrid[0:64, 0:96]
    smooth = np.stack([yy * 3, xx * 2, yy + xx], axis=-1).astype(np.uint8)
    smooth[20:44, 30:70] = 200
    smooth_path = tmp_path / "smooth.png"
    Image.fromarray(smooth).save(smooth_path)
    smooth_spec = {"source": str(smooth_path), "boxes": [[0, 0, 48, 32]]}
    drifted = np.clip(smooth.astype(int) + rng.integers(-20, 21, smooth.shape), 0, 255).astype(
        np.uint8
    )
    blurred = cv2.GaussianBlur(drifted, (0, 0), 1.0)
    keep_drifted = model(
        RewardInferenceArtifact(
            "f", "f", "", media=Image.fromarray(drifted), metadata={"locality_keep": smooth_spec}
        )
    )
    keep_blurred = model(
        RewardInferenceArtifact(
            "g", "g", "", media=Image.fromarray(blurred), metadata={"locality_keep": smooth_spec}
        )
    )
    assert keep_blurred["locality_psnr"] > keep_drifted["locality_psnr"]
    assert keep_blurred["locality_keep"] < keep_drifted["locality_keep"]
    with pytest.raises(ValueError, match="source path"):
        model(RewardInferenceArtifact("d", "d", "", media=Image.fromarray(source)))
    with pytest.raises(ValueError, match="does not match"):
        model(
            RewardInferenceArtifact(
                "e", "e", "", media=Image.new("RGB", (8, 8)), metadata={"locality_keep": spec}
            )
        )

    reward = MultiReward.from_dict(
        {"locality_keep": 1.0},
        device="cuda",
        inference_configs={"locality_keep": RewardInferenceConfig(kind="in_process")},
    )
    try:
        output = await reward.score_batch(
            [
                RewardSample(
                    "keep",
                    torch.from_numpy(pixels).permute(2, 0, 1).float() / 255,
                    str(i),
                    {"locality_keep": spec},
                )
                for i, pixels in enumerate((inside, outside))
            ]
        )
    finally:
        await reward.shutdown()
    assert output.scores[0] == 1 > output.scores[1]
    assert output.components["locality_keep/locality_hf_ratio"][0] == 1


@pytest.mark.asyncio
async def test_detail_penalty_selects_configured_key_and_keeps_diagnostics(tmp_path):
    rng = np.random.default_rng(19)
    source = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(source).save(tmp_path / "source.png")
    spec = {"source": str(tmp_path / "source.png"), "boxes": [[0, 0, 32, 32]]}
    inside = source.copy()
    inside[:32, :32] = 0
    damaged = source.copy()
    damaged[32:, 32:] //= 2
    reward = MultiReward.from_dict(
        {"locality_keep": 1.0},
        device="cpu",
        reward_kwargs={
            "locality_keep": {
                "score_key": "locality_detail_keep",
                "worker_config": {
                    "detail_penalty_weights": {
                        "normalized_mse": 1.0,
                        "relative_laplacian_mae": 0.2,
                        "absolute_log_hf_ratio": 0.1,
                    }
                },
            }
        },
        inference_configs={"locality_keep": RewardInferenceConfig(kind="in_process")},
    )
    try:
        output = await reward.score_batch(
            [
                RewardSample(
                    "edit",
                    torch.from_numpy(a).permute(2, 0, 1).float() / 255,
                    str(index),
                    {"locality_keep": spec},
                )
                for index, a in enumerate((inside, damaged))
            ]
        )
    finally:
        await reward.shutdown()
    assert output.scores[0] == 1 > output.scores[1] > 0
    assert (
        output.components["locality_keep"]
        == output.components["locality_keep/locality_detail_keep"]
    )
    assert output.components["locality_keep/locality/normalized_mse"][0] == 0
    assert output.components["locality_keep/locality/relative_laplacian_mae"][1] > 0


def test_local_texture_catches_damage_hidden_by_balanced_global_energy(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    pattern = np.where((xx // 4 + yy // 4) % 2, 1, -1)
    source = np.repeat((128 + 16 * pattern)[..., None], 3, axis=-1).astype(np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(source).save(path)
    candidate = np.full_like(source, 128)
    candidate[:, 64:] = np.repeat((128 + 23 * pattern[:, 64:])[..., None], 3, axis=-1)
    artifact = RewardInferenceArtifact(
        "a",
        "a",
        "",
        media=Image.fromarray(candidate),
        metadata={"locality_keep": {"source": str(path), "boxes": []}},
    )
    scores = LocalityKeepRewardModel(
        {
            "detail_penalty_weights": {
                "normalized_mse": 0.0,
                "relative_laplacian_mae": 0.0,
                "absolute_log_hf_ratio": 0.1,
                "mean_local_log_hf_ratio": 0.1,
            },
            "local_texture": {
                "tile_size": 32,
                "variance_floor": 1.0,
                "minimum_protected_fraction": 0.5,
            },
        }
    )(artifact)
    # One half loses its pattern while the other adds energy. A pooled ratio
    # stays near one, but spatial discrepancies do not cancel each other.
    assert scores["locality/absolute_log_hf_ratio"] < 0.15
    assert scores["locality/mean_local_log_hf_ratio"] > 2
    assert scores["locality_detail_keep"] < 0.8


def test_local_texture_ignores_legal_edits_and_charges_grain_on_flat_partial_tiles(tmp_path):
    source = np.full((53, 77, 3), 128, dtype=np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(source).save(path)
    spec = {"source": str(path), "boxes": [[0, 0, 32, 32]]}
    inside = source.copy()
    inside[:32, :32] = 0
    config = {
        "detail_penalty_weights": {
            "normalized_mse": 1.0,
            "relative_laplacian_mae": 0.2,
            "absolute_log_hf_ratio": 0.1,
            "mean_local_log_hf_ratio": 0.1,
        },
        "local_texture": {
            "tile_size": 32,
            "variance_floor": 1.0,
            "minimum_protected_fraction": 0.5,
        },
    }
    model = LocalityKeepRewardModel(config)
    allowed = model(
        RewardInferenceArtifact(
            "a",
            "a",
            "",
            media=Image.fromarray(inside),
            metadata={"locality_keep": spec},
        )
    )
    assert allowed["locality_detail_keep"] == 1
    assert allowed["locality/mean_local_log_hf_ratio"] == 0
    damaged = inside.copy()
    damaged[32:] = np.clip(
        damaged[32:].astype(int) + np.random.default_rng(7).integers(-12, 13, damaged[32:].shape),
        0,
        255,
    )
    measured = model(
        RewardInferenceArtifact(
            "b",
            "b",
            "",
            media=Image.fromarray(damaged),
            metadata={"locality_keep": spec},
        )
    )
    assert measured["locality/mean_local_log_hf_ratio"] > 0
    assert measured["locality_detail_keep"] < allowed["locality_detail_keep"]
    with pytest.raises(ValueError, match="local_texture must declare"):
        LocalityKeepRewardModel({"detail_penalty_weights": config["detail_penalty_weights"]})
    with pytest.raises(ValueError, match="variance_floor must"):
        LocalityKeepRewardModel(
            {**config, "local_texture": {**config["local_texture"], "variance_floor": 0}}
        )
