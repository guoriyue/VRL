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
