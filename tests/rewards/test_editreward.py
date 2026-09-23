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


def test_locality_adapter_validates_task_identity_and_emits_separate_scores(tmp_path):
    import json
    from dataclasses import replace

    from vrl.utils.artifacts import sha256_file

    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "grey").save(source)
    prompt = "Change only the left half blue."
    config = {
        "color_hue_degrees": {"blue": 240},
        "hue_sigma_degrees": 25,
        "saturation_start": 0.15,
        "saturation_full": 0.5,
        "quality_weight": 0.2,
        "tasks": {
            "test": {
                "prompt": prompt,
                "source_sha256": sha256_file(source),
                "color": "blue",
                "target": [[0, 0, 0.5, 1]],
                "protected": {"right": [0.5, 0, 1, 1]},
            }
        },
    }
    path = tmp_path / "locality.json"
    path.write_text(json.dumps(config))

    class Judge:
        def reward(self, **kwargs):
            return torch.tensor([[0.5, -2.0]])

    model = EditRewardModel(
        {
            "data_root": str(tmp_path),
            "device": "cpu",
            "locality_config": str(path),
            "audit_dir": str(tmp_path / "audit"),
        }
    )
    model._module = Judge()
    image = torch.full((3, 10, 20), 128, dtype=torch.uint8)
    image[:, :, :10] = torch.tensor([0, 0, 255], dtype=torch.uint8)[:, None, None]
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt=prompt,
        media=image,
        metadata={"task_id": "test", "reference_images": ["source.png"]},
    )
    scores = model(artifact)
    assert scores["editreward"] == 0.5
    assert scores["completion"] > 0.99
    assert scores["preservation_right"] == 1
    assert scores["editreward_locality"] > 1
    record = json.loads(next((tmp_path / "audit").glob("*.json")).read_text())
    assert record["sample_id"] == artifact.sample_id
    assert record["scores"] == scores
    with Image.open(tmp_path / "audit" / record["image"]) as saved:
        assert saved.getpixel((0, 0)) == (0, 0, 255)
        assert saved.getpixel((19, 0)) == (128, 128, 128)
    with pytest.raises(ValueError, match="Uncalibrated"):
        model(replace(artifact, prompt="Change everything blue."))
    Image.new("RGB", (20, 10), "red").save(source)
    with pytest.raises(ValueError, match="source image changed"):
        model(artifact)
