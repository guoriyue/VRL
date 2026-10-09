"""Known-layer decomposition axes: complete truth scores one, hidden erasure and merging lose."""

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.layered_reference import LayeredReferenceRewardModel
from vrl.rewards.types import RewardSample


def _truth() -> np.ndarray:
    """Background, a box whose lower half sits behind a bar, and the bar in front."""
    h = w = 32
    layers = np.zeros((3, h, w, 4), dtype=np.float32)
    # Colors are exact in 8 bits so the PNG truth equals the float candidate.
    layers[0, ..., :3] = (230 / 255, 230 / 255, 204 / 255)
    layers[0, ..., 3] = 1.0
    layers[1, 8:24, 8:24, :3] = (51 / 255, 77 / 255, 204 / 255)
    layers[1, 8:24, 8:24, 3] = 1.0
    layers[2, 16:28, 4:28, :3] = (204 / 255, 51 / 255, 51 / 255)
    layers[2, 16:28, 4:28, 3] = 1.0
    return layers


def _write(directory, layers) -> None:
    directory.mkdir()
    for index, layer in enumerate(layers):
        Image.fromarray((layer * 255).round().astype(np.uint8), mode="RGBA").save(
            directory / f"layer_{index}.png"
        )


def _artifact(stack: np.ndarray, directory) -> RewardInferenceArtifact:
    media = torch.from_numpy(stack).permute(0, 3, 1, 2)  # [L, 4, H, W]
    return RewardInferenceArtifact(
        "a", "a", "", media=media, metadata={"layered_reference": {"layers": str(directory)}}
    )


def test_complete_truth_scores_one_and_hidden_erasure_or_merging_lose(tmp_path) -> None:
    truth = _truth()
    _write(tmp_path / "truth", truth)
    model = LayeredReferenceRewardModel({})

    exact = model(_artifact(truth, tmp_path / "truth"))
    assert exact["layered_reference"] == pytest.approx(1.0)
    assert exact["layered_hidden_completion"] == pytest.approx(1.0)
    assert exact["layered_object_match"] == pytest.approx(1.0)

    erased = truth.copy()
    erased[1, 16:24, 8:24] = 0.0  # the box's hidden half is missing
    partial = model(_artifact(erased, tmp_path / "truth"))
    assert partial["layered_hidden_completion"] < 0.8  # mean over objects; background keeps 1
    assert partial["layered_visible_retention"] == pytest.approx(1.0)
    assert partial["layered_edit_holes"] < 1.0

    merged = truth.copy()
    merged[1] = np.where(
        truth[2, ..., 3:4] > 0, truth[2], truth[1]
    )  # bar painted into the box layer
    merged[2] = 0.0
    fused = model(_artifact(merged, tmp_path / "truth"))
    assert fused["layered_object_match"] < partial["layered_object_match"]
    assert fused["layered_object_match"] < 0.5


def test_candidate_geometry_and_metadata_are_validated(tmp_path) -> None:
    truth = _truth()
    _write(tmp_path / "truth", truth)
    model = LayeredReferenceRewardModel({})

    with pytest.raises(ValueError, match="training geometry"):
        model(_artifact(truth[:, ::2, ::2], tmp_path / "truth"))
    with pytest.raises(ValueError, match="layered_reference"):
        model(RewardInferenceArtifact("b", "b", "", media=torch.zeros(3, 4, 32, 32), metadata={}))
    with pytest.raises(TypeError, match="RGBA stack"):
        model(
            RewardInferenceArtifact(
                "c",
                "c",
                "",
                media=torch.zeros(3, 32, 32),
                metadata={"layered_reference": {"layers": str(tmp_path / "truth")}},
            )
        )


@pytest.mark.asyncio
async def test_components_are_separately_weighted_rewards(tmp_path) -> None:
    truth = _truth()
    _write(tmp_path / "truth", truth)
    erased = truth.copy()
    erased[1, 16:24, 8:24] = 0.0
    reward = MultiReward.from_dict(
        {"layered_hidden_completion": 1.0, "layered_visible_retention": 0.5},
        inference_configs={
            "layered_hidden_completion": RewardInferenceConfig(kind="in_process"),
            "layered_visible_retention": RewardInferenceConfig(kind="in_process"),
        },
        device="cpu",
    )
    await reward.activate()
    try:
        output = await reward.score_batch(
            [
                RewardSample(
                    prompt="decompose",
                    output=torch.from_numpy(stack).permute(0, 3, 1, 2),
                    sample_id=f"s{index}",
                    metadata={"layered_reference": {"layers": str(tmp_path / "truth")}},
                )
                for index, stack in enumerate((truth, erased))
            ]
        )
    finally:
        await reward.shutdown()
    assert output.components["layered_hidden_completion"][0] == pytest.approx(1.0)
    assert output.components["layered_hidden_completion"][1] < 0.8
    assert output.components["layered_visible_retention"] == pytest.approx((1.0, 1.0))
    assert output.scores[0] == pytest.approx(1.5)
    assert output.scores[1] < output.scores[0]
