"""The tonal anchor is silent inside its tolerance and penalizes brightening or saturating away from the reference."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.tonal_anchor import TonalAnchorRewardModel


def _artifact(image, reference_path):
    media = torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1)
    return RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="p",
        media=media,
        metadata={"judge_reference": str(reference_path)},
    )


def test_identity_is_zero_and_drift_is_penalized_in_proportion(tmp_path):
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(40, 200, (64, 64, 3), dtype=np.uint8))
    ref = tmp_path / "ref.png"
    base.save(ref)
    model = TonalAnchorRewardModel({"tolerance": 0.02})

    same = model(_artifact(base, ref))
    assert same["tonal_anchor"] == 0.0
    assert same["tonal_anchor/luma_delta"] == pytest.approx(0.0, abs=1e-6)

    bright = model(_artifact(ImageEnhance.Brightness(base).enhance(1.25), ref))
    brighter = model(_artifact(ImageEnhance.Brightness(base).enhance(1.5), ref))
    assert bright["tonal_anchor/luma_delta"] > 0.02
    assert brighter["tonal_anchor"] < bright["tonal_anchor"] < 0.0

    saturated = model(_artifact(ImageEnhance.Color(base).enhance(1.6), ref))
    assert saturated["tonal_anchor/saturation_delta"] > 0.02
    assert saturated["tonal_anchor"] < 0.0

    # a darker image is penalized the same way as a brighter one: the anchor has no direction
    dark = model(_artifact(ImageEnhance.Brightness(base).enhance(0.75), ref))
    assert dark["tonal_anchor"] < 0.0


def test_missing_reference_is_rejected(tmp_path):
    model = TonalAnchorRewardModel({})
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="p",
        media=torch.zeros(3, 8, 8),
        metadata={},
    )
    with pytest.raises(ValueError, match="judge_reference"):
        model(artifact)
    with pytest.raises(ValueError, match="finite"):
        TonalAnchorRewardModel({"tolerance": float("nan")})
