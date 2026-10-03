"""The VLM preference reward is the weighted mean of per-criterion answer scores, with the prompt substituted."""

import pytest
import torch
from PIL import Image

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.vlm_preference import DEFAULT_CRITERIA, VLMPreferenceRewardModel


def _model(answers, **config):
    model = VLMPreferenceRewardModel({"device": "cpu", **config})
    model.asked = []

    def answer_scores(image, questions, modes):
        model.asked.append((image.size, list(questions), list(modes)))
        return list(answers)

    model.answer_scores = answer_scores
    return model


def test_weighted_mean_and_per_criterion_keys():
    criteria = [
        {"name": "a", "question": "A?", "weight": 3.0, "answer": "rating"},
        {"name": "b", "question": "B?", "weight": 1.0},
    ]
    scores = _model([1.0, 0.0], criteria=criteria).score(Image.new("RGB", (8, 8)), prompt="x")
    assert scores == {
        "vlm_preference": pytest.approx(0.75),
        "vlm_preference/a": 1.0,
        "vlm_preference/b": 0.0,
    }


def test_prompt_is_substituted_and_all_criteria_batch_once_with_their_answer_modes():
    model = _model([0.5] * len(DEFAULT_CRITERIA))
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="a red fox on mossy stones",
        media=torch.zeros(3, 6, 6, dtype=torch.uint8),
        metadata={},
    )
    scores = model(artifact)
    assert scores["vlm_preference"] == pytest.approx(0.5)
    assert len(model.asked) == 1  # one batched call covers every criterion
    _, questions, modes = model.asked[0]
    assert len(questions) == len(DEFAULT_CRITERIA)
    assert sum(q.count('"a red fox on mossy stones"') for q in questions) == 2  # clean + alignment
    assert not any("{prompt}" in q for q in questions)
    assert modes == [c["answer"] for c in DEFAULT_CRITERIA]
    assert "rating" in modes and "yesno" in modes


def test_bad_criteria_are_rejected():
    with pytest.raises(ValueError, match="name and a question"):
        VLMPreferenceRewardModel({"device": "cpu", "criteria": [{"name": "x"}]})
    with pytest.raises(ValueError, match="unique"):
        VLMPreferenceRewardModel(
            {
                "device": "cpu",
                "criteria": [{"name": "x", "question": "q"}, {"name": "x", "question": "r"}],
            }
        )
    with pytest.raises(ValueError, match="answer must be one of"):
        VLMPreferenceRewardModel(
            {"device": "cpu", "criteria": [{"name": "x", "question": "q", "answer": "score"}]}
        )


def test_pairwise_mode_reads_the_reference_and_averages_both_orders(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(ref)
    model = VLMPreferenceRewardModel(
        {"device": "cpu", "reference_key": "judge_reference", "data_root": str(tmp_path)}
    )
    calls = []

    def answer_scores(image, questions, modes, reference=None):
        calls.append((image.getpixel((0, 0)), reference.getpixel((0, 0)), list(questions)))
        # candidate-first: 0.8 for every criterion; reference-first: 0.4 -> averaged 0.7
        return [0.8 if image.getpixel((0, 0)) != (255, 0, 0) else 0.4] * len(questions)

    model.answer_scores = answer_scores
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="a red fox",
        media=torch.zeros(3, 8, 8, dtype=torch.uint8),
        metadata={"judge_reference": "ref.png"},
    )
    scores = model(artifact)
    assert scores["vlm_preference"] == pytest.approx(0.7)
    assert len(calls) == 2 and calls[0][1] == (255, 0, 0) and calls[1][0] == (255, 0, 0)
    assert all('"a red fox"' in q and "Image 1" in q for q in calls[0][2])


def test_pairwise_mode_without_a_reference_in_metadata_is_rejected():
    model = VLMPreferenceRewardModel({"device": "cpu", "reference_key": "judge_reference"})
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="x",
        media=torch.zeros(3, 4, 4),
        metadata={},
    )
    with pytest.raises(ValueError, match="judge_reference"):
        model(artifact)


def test_crop_criteria_are_judged_per_tile_and_averaged(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(ref)
    criteria = [
        {"name": "whole", "question": "W?", "weight": 1.0},
        {"name": "detail", "question": "D?", "weight": 1.0, "crops": 2},
    ]
    model = VLMPreferenceRewardModel(
        {
            "device": "cpu",
            "reference_key": "judge_reference",
            "data_root": str(tmp_path),
            "both_orders": False,
            "criteria": criteria,
        }
    )
    calls = []

    def answer_scores(image, questions, modes, reference=None):
        calls.append((image.size, list(questions)))
        return [0.25 if "D?" in questions[0] else 1.0] * len(questions)

    model.answer_scores = answer_scores
    artifact = RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="a red fox",
        media=torch.zeros(3, 8, 8, dtype=torch.uint8),
        metadata={"judge_reference": "ref.png"},
    )
    scores = model(artifact)
    # one whole-image call plus four 4x4 tiles for the crop criterion
    assert [size for size, _ in calls] == [(8, 8), (4, 4), (4, 4), (4, 4), (4, 4)]
    assert scores["vlm_preference/whole"] == 1.0
    assert scores["vlm_preference/detail"] == pytest.approx(0.25)
    assert scores["vlm_preference"] == pytest.approx(0.625)
