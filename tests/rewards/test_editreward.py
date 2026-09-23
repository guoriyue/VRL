"""EditReward keeps the original instruction/reference across media conversion."""

import pytest
import torch
from PIL import Image

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.editreward import EditRewardModel


def test_reference_and_rgba_candidate_reach_judge_with_original_instruction(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 12), (255, 0, 0)).save(source)
    calls = []

    class Judge:
        def reward(self, *, prompts, image_src, image_paths):
            calls.append((prompts, image_src, image_paths))
            return torch.tensor([[0.75, -2.0]])

    model = EditRewardModel({"data_root": str(tmp_path), "device": "cpu"})
    model._module = Judge()
    # A transparent blue image must be judged as white, matching the baseline.
    image = torch.zeros(4, 6, 4, dtype=torch.uint8)
    image[2] = 255
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="Change only the seat to blue.",
        media=image,
        metadata={"reference_images": ["source.png"]},
    )
    assert model(artifact) == {"editreward": 0.75, "editreward_log_sigma": -2.0}
    prompts, references, candidates = calls[0]
    assert prompts == ["Change only the seat to blue."]
    assert references[0].size == candidates[0].size == (4, 6)
    assert references[0].getpixel((0, 0)) == (255, 0, 0)
    assert candidates[0].getpixel((0, 0)) == (255, 255, 255)


def test_multiple_references_are_rejected_instead_of_silently_dropping_one(tmp_path):
    model = EditRewardModel({"data_root": str(tmp_path), "device": "cpu"})
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="Combine both chairs.",
        media=torch.zeros(3, 4, 4),
        metadata={"reference_images": ["a.png", "b.png"]},
    )
    with pytest.raises(ValueError, match="one reference"):
        model(artifact)
