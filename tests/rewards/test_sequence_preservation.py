"""Sequence audits separate measured regressions from unobserved intervals."""

import copy

import pytest

from vrl.rewards.evaluation import Evaluation
from vrl.rewards.sequences import EditSequenceSpec


def test_staged_requirements_expose_regression_even_when_final_state_is_repaired():
    spec = EditSequenceSpec.model_validate(
        {
            "sequence_id": "dialogue-edit",
            "samples": ["initial", "added", "damaged", "repaired"],
            "requirements": [
                {"name": "first-bubble", "axis": "text", "threshold": 1.0},
                {
                    "name": "protect-artwork",
                    "axis": "damage",
                    "direction": -1,
                    "threshold": 0.1,
                    "active_from": 1,
                },
            ],
        }
    )
    evaluation = Evaluation("test", {"revision": "fixed"}, {})
    for sample, text, damage in zip(spec.samples, (0, 1, 0, 1), (0.8, 0.0, 0.2, 0.0), strict=True):
        evaluation.records[sample] = {
            "input": {
                "sample_id": sample,
                "sha256": sample,
                "path": sample + ".png",
                "prompt": "Keep dialogue",
                "metadata": {"target": "Hello"},
            },
            "status": "success",
            "result": {"scores": {"text": text, "damage": damage}},
        }
    report = spec.report(evaluation)
    assert report["final_requirements_met"] and report["coverage_complete"]
    assert report["requirements"]["first-bubble"]["first_satisfied_step"] == 1
    assert report["requirements"]["first-bubble"]["regression_steps"] == [2]
    assert report["requirements"]["first-bubble"]["improvement_steps"] == [1, 3]
    assert report["states"][0]["requirements"]["protect-artwork"]["status"] == "not_active"
    assert report["requirements"]["protect-artwork"]["regression_steps"] == [2]
    changed = copy.deepcopy(evaluation)
    changed.records["damaged"]["input"]["metadata"]["target"] = "Goodbye"
    with pytest.raises(ValueError, match="specification changed"):
        spec.report(changed)


def test_missing_axis_and_failed_scoring_are_unknown_not_regressions_or_zero_rewards():
    spec = EditSequenceSpec.model_validate(
        {
            "sequence_id": "partial",
            "samples": ["a", "b", "c", "d"],
            "requirements": [{"name": "text", "axis": "text", "threshold": 1.0}],
        }
    )
    records = {
        "a": {"status": "success", "result": {"scores": {"text": 1.0}}},
        "b": {"status": "error", "error": {"type": "TimeoutError", "message": "unmeasured"}},
        "c": {"status": "success", "result": {"scores": {"text": 0.0}}},
        "d": {"status": "success", "result": {"scores": {"unrelated": 10}}},
    }
    for sample, row in records.items():
        row["input"] = {"sample_id": sample, "sha256": sample, "prompt": "fixed"}
    evaluation = Evaluation("partial", {}, records)
    report = spec.report(evaluation)
    assert report["final_requirements_met"] is None and not report["coverage_complete"]
    assert report["requirements"]["text"]["unknown_steps"] == [1, 3]
    assert report["requirements"]["text"]["regression_steps"] == []
    assert report["states"][1]["error"]["type"] == "TimeoutError"
    assert report["states"][1]["requirements"]["text"]["value"] is None
    records["d"]["result"]["scores"]["text"] = float("nan")
    with pytest.raises(ValueError, match="finite numbers"):
        spec.report(evaluation)


def test_invalid_sequence_cannot_reuse_one_score_as_multiple_steps():
    with pytest.raises(ValueError, match="unique"):
        EditSequenceSpec.model_validate(
            {
                "sequence_id": "copy",
                "samples": ["same", "same"],
                "requirements": [{"name": "text", "axis": "text", "threshold": 1.0}],
            }
        )
    with pytest.raises(ValueError, match="outside the sequence"):
        EditSequenceSpec.model_validate(
            {
                "sequence_id": "never",
                "samples": ["a", "b"],
                "requirements": [
                    {"name": "text", "axis": "text", "threshold": 1.0, "active_from": 2}
                ],
            }
        )
