"""Label agreement reports one AUC per contrast and never scores what the run did not."""

import json

import pytest

from reward.labels import Contrast, OutcomeLabel, agreement, rank_auc
from vrl.rewards.evaluation import Evaluation


def _evaluation(scores: dict[str, float | None]) -> Evaluation:
    records = {}
    for sample_id, score in scores.items():
        if score is None:
            records[sample_id] = {
                "input": {"sample_id": sample_id},
                "status": "error",
                "error": {},
            }
        else:
            records[sample_id] = {
                "input": {"sample_id": sample_id, "prompt_id": "p"},
                "status": "success",
                "result": {"scores": {"judge": score}},
            }
    return Evaluation("run", {}, records)


def _labels(rows: list[tuple[str, str, bool]]) -> list[OutcomeLabel]:
    return [
        OutcomeLabel(sample_id=s, labels={"outcome": outcome, "collateral": collateral})
        for s, outcome, collateral in rows
    ]


CONTRASTS = [
    Contrast(
        name="done_vs_not_done", positive={"outcome": ["done"]}, negative={"outcome": ["not_done"]}
    ),
    Contrast(
        name="clean_vs_collateral",
        positive={"outcome": ["done"], "collateral": [False]},
        negative={"outcome": ["done"], "collateral": [True]},
    ),
]


def test_rank_auc_counts_ties_as_half():
    assert rank_auc([2, 3], [1, 1]) == 1.0
    assert rank_auc([1], [1]) == 0.5
    assert rank_auc([1, 2], [2, 3]) == 0.125


def test_agreement_reports_each_contrast_and_lists_unscored_samples():
    evaluation = _evaluation({"a": 2.0, "b": 1.5, "c": -1.0, "d": -0.5, "e": 1.9, "f": None})
    labels = _labels(
        [
            ("a", "done", False),
            ("b", "done", False),
            ("e", "done", True),
            ("c", "not_done", False),
            ("d", "not_done", False),
            ("f", "done", False),
        ]
    )
    report = agreement(evaluation, labels, CONTRASTS, axis="judge", gate=0.85, resamples=50)
    done = report["contrasts"]["done_vs_not_done"]
    assert done["auc"] == 1.0 and done["passed"] is True
    assert (done["positive_count"], done["negative_count"]) == (3, 2)
    clean = report["contrasts"]["clean_vs_collateral"]
    # e (collateral) sits between a and b: one of two clean samples ranks above it.
    assert clean["auc"] == 0.5 and clean["passed"] is False
    assert report["unscored_sample_ids"] == ["f"]
    low, high = done["auc_bootstrap_95ci"]
    assert 0.0 <= low <= done["auc"] <= high <= 1.0


def test_agreement_direction_flips_and_empty_side_is_not_a_verdict():
    evaluation = _evaluation({"a": -2.0, "c": 1.0})
    labels = _labels([("a", "done", False), ("c", "not_done", False)])
    lower_is_better = agreement(
        evaluation, labels, CONTRASTS[:1], axis="judge", direction=-1, resamples=10
    )
    assert lower_is_better["contrasts"]["done_vs_not_done"]["auc"] == 1.0
    empty = agreement(evaluation, labels, CONTRASTS[1:], axis="judge", resamples=10)
    assert empty["contrasts"]["clean_vs_collateral"]["passed"] is None


def test_label_and_contrast_files_reject_duplicates_and_ambiguous_sides(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"sample_id": "a", "labels": {"outcome": "done"}})
        + "\n"
        + json.dumps({"sample_id": "a", "labels": {"outcome": "done"}})
        + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        OutcomeLabel.load_jsonl(labels)
    contrasts = tmp_path / "contrasts.json"
    contrasts.write_text(
        json.dumps(
            [{"name": "x", "positive": {"outcome": ["done"]}, "negative": {"outcome": ["done"]}}]
        )
    )
    (loaded,) = Contrast.load_json(contrasts)
    with pytest.raises(ValueError, match="both sides"):
        loaded.side(OutcomeLabel(sample_id="a", labels={"outcome": "done"}))
