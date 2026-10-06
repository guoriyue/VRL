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


def test_compare_answer_mode_is_accepted_and_batched_with_its_suffix():
    criteria = [{"name": "fidelity", "question": "F?", "answer": "compare"}]
    model = _model([0.25], criteria=criteria)
    scores = model.score(Image.new("RGB", (8, 8)), prompt="x")
    assert scores["vlm_preference/fidelity"] == 0.25
    assert model.asked[0][2] == ["compare"]
    with pytest.raises(ValueError, match="answer must be one of"):
        VLMPreferenceRewardModel(
            {"device": "cpu", "criteria": [{"name": "x", "question": "q", "answer": "grade"}]}
        )


def test_intent_metadata_fills_the_intent_placeholder_and_falls_back_to_the_prompt():
    criteria = [{"name": "fid", "question": 'Does it show "{intent}" for "{prompt}"?'}]
    model = _model([1.0], criteria=criteria, intent_key="judge_intent")

    def artifact(metadata):
        return RewardInferenceArtifact(
            artifact_id="a",
            sample_id="s",
            path="",
            prompt="gold jaguar print hoodie",
            media=torch.zeros(3, 6, 6, dtype=torch.uint8),
            metadata=metadata,
        )

    model(artifact({"judge_intent": "a hoodie with a gold leopard-spot pattern"}))
    model(artifact({}))
    expanded, plain = (q[0] for _, q, _ in model.asked)
    assert expanded == (
        'Does it show "a hoodie with a gold leopard-spot pattern" for "gold jaguar print hoodie"?'
    )
    assert plain == 'Does it show "gold jaguar print hoodie" for "gold jaguar print hoodie"?'


def test_questions_per_forward_splits_one_image_into_bounded_forwards():
    criteria = [{"name": f"c{i}", "question": f"Q{i}?"} for i in range(5)]
    model = VLMPreferenceRewardModel(
        {"device": "cpu", "criteria": criteria, "questions_per_forward": 2}
    )
    forwards = []

    def leaf_forward(image, questions, answers, reference=None):
        forwards.append(list(questions))
        return [0.1 * (int(q[1]) + 1) for q in questions]

    # The instance attribute replaces the per-chunk leaf; the class method on top
    # of it is the chunking under test.
    model.answer_scores = leaf_forward
    scores = VLMPreferenceRewardModel.answer_scores(
        model, Image.new("RGB", (4, 4)), [c["question"] for c in criteria], ["yesno"] * 5
    )
    assert forwards == [["Q0?", "Q1?"], ["Q2?", "Q3?"], ["Q4?"]]
    assert scores == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_checklist_criterion_asks_each_item_about_the_candidate_alone(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(ref)
    criteria = [
        {"name": "overall", "question": "Is Image 1 better?", "weight": 1.0},
        {
            "name": "elements",
            "question": "Does Image 1 clearly show {item}?",
            "weight": 1.0,
            "checklist": True,
        },
    ]
    model = VLMPreferenceRewardModel(
        {
            "device": "cpu",
            "criteria": criteria,
            "reference_key": "judge_reference",
            "checklist_key": "judge_checklist",
            "data_root": str(tmp_path),
        }
    )
    calls = []

    def answer_scores(image, questions, modes, reference=None):
        calls.append((list(questions), reference is not None))
        if reference is None:  # checklist: the motorcycle is missing
            return [0.0 if "motorcycle" in q else 1.0 for q in questions]
        return [0.6] * len(questions)

    model.answer_scores = answer_scores

    def artifact(items):
        return RewardInferenceArtifact(
            artifact_id="a",
            sample_id="s",
            path="",
            prompt="a rabbit riding a suzuki",
            media=torch.zeros(3, 8, 8, dtype=torch.uint8),
            metadata={"judge_reference": "ref.png", "judge_checklist": items},
        )

    scores = model(artifact(["a black rabbit", "a Suzuki motorcycle being ridden"]))
    checklist_calls = [q for q, paired in calls if not paired]
    assert checklist_calls == [
        [
            "Does Image 1 clearly show a black rabbit?",
            "Does Image 1 clearly show a Suzuki motorcycle being ridden?",
        ]
    ]
    assert scores["vlm_preference/elements"] == pytest.approx(0.5)
    assert all("{item}" not in q for q, _ in calls)

    calls.clear()
    assert model(artifact([]))["vlm_preference/elements"] == 0.5
    assert all(paired for _, paired in calls)  # no checklist forward without items


def test_checklist_criterion_needs_an_item_placeholder():
    with pytest.raises(ValueError, match="item"):
        VLMPreferenceRewardModel(
            {"device": "cpu", "criteria": [{"name": "c", "question": "Q?", "checklist": True}]}
        )


def test_absolute_inverted_defect_questions_see_the_candidate_alone(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(ref)
    criteria = [
        {"name": "overall", "question": "Is Image 1 better?", "weight": 1.0},
        {
            "name": "anatomy",
            "question": "Does Image 1 show extra or merged limbs?",
            "weight": 1.0,
            "absolute": True,
            "invert": True,
        },
    ]
    model = VLMPreferenceRewardModel(
        {
            "device": "cpu",
            "criteria": criteria,
            "reference_key": "judge_reference",
            "data_root": str(tmp_path),
        }
    )
    calls = []

    def answer_scores(image, questions, modes, reference=None):
        calls.append((list(questions), reference is not None))
        return [0.8] * len(questions) if reference is None else [0.5] * len(questions)

    model.answer_scores = answer_scores
    scores = model(
        RewardInferenceArtifact(
            artifact_id="a",
            sample_id="s",
            path="",
            prompt="a knight",
            media=torch.zeros(3, 8, 8, dtype=torch.uint8),
            metadata={"judge_reference": "ref.png"},
        )
    )
    assert scores["vlm_preference/anatomy"] == pytest.approx(0.2)  # P(defect)=0.8 -> 0.2
    alone = [q for q, paired in calls if not paired]
    assert alone == [["Does Image 1 show extra or merged limbs?"]]
    assert all("extra or merged" not in q for qs, paired in calls if paired for q in qs)


def test_invert_requires_an_absolute_yesno_criterion():
    with pytest.raises(ValueError, match="invert"):
        VLMPreferenceRewardModel(
            {"device": "cpu", "criteria": [{"name": "c", "question": "Q?", "invert": True}]}
        )
