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
