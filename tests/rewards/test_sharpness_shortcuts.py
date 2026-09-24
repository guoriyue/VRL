"""High-frequency shortcuts expose why sharpness is not semantic quality."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.image_sharpness import ImageSharpnessRewardModel


def test_noise_saturates_sharpness_while_blur_reduces_real_edges(tmp_path):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[32:96, 32:96] = 255
    edge = Image.fromarray(image)
    noise = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, image.shape, dtype=np.uint8)
    )
    model = ImageSharpnessRewardModel({})
    scores = {}
    for name, candidate in (
        ("edge", edge),
        ("blur", edge.filter(ImageFilter.GaussianBlur(3))),
        ("noise", noise),
    ):
        path = tmp_path / f"{name}.png"
        candidate.save(path)
        artifact = RewardInferenceArtifact(name, name, str(path))
        scores[name] = model(artifact)["image_sharpness"]
    assert scores["edge"] > scores["blur"]
    assert scores["noise"] == 1.0
    assert scores["noise"] >= scores["edge"]
    # NaN used to fall through the comparison and min(1, NaN) returned a perfect 1.
    with pytest.raises(ValueError, match="finite"):
        ImageSharpnessRewardModel({"scale": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        ImageSharpnessRewardModel({"scale": float("inf")})
