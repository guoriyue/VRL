"""Frozen reward application preserves raw evidence and cannot refit or hide failures."""

import copy
import json

import pytest

from reward_lab.calibration import PreferencePair, apply_combination, fit_combination
from vrl.utils.json_files import canonical_json_sha256


def test_application_uses_frozen_signed_coefficients_and_preserves_failed_measurements():
    records, pairs = {}, []
    for group in range(2):
        for side, quality, damage in (("left", 3.0, 0.0), ("right", 1.0, 2.0)):
            key = f"{group}-{side}"
            records[key] = {
                "status": "success",
                "input": {"prompt_id": f"p{group}", "sha256": key},
                "result": {"scores": {"quality": quality, "damage": damage}},
            }
        pairs.append(
            PreferencePair(
                pair_id=f"pair-{group}",
                left=f"{group}-left",
                right=f"{group}-right",
                source_group=f"group-{group}",
                split="calibration",
                preference="left",
            )
        )
    calibration = {"run_id": "fit", "config": {"revision": "frozen"}, "records": records}
    combination = fit_combination(calibration, pairs, axes=["quality", "damage"])
    frozen_copy = copy.deepcopy(combination)
    assert combination["weights"][0] > 0 > combination["weights"][1]
    evaluation = {
        "run_id": "new",
        "config": calibration["config"],
        "records": {
            "new": {
                "status": "success",
                "input": {"prompt_id": "new", "sha256": "new"},
                "result": {
                    "scores": {"quality": 10.0, "damage": 4.0},
                    "diagnostics": {"why": "retained evidence"},
                },
            },
            "failed": {
                "status": "error",
                "input": {"sha256": "bad"},
                "error": {"type": "TimeoutError", "message": "no observation"},
            },
            "absent": {"status": "missing", "input": {"sha256": "absent"}},
        },
    }
    applied = apply_combination(evaluation, combination)
    row = applied["records"]["new"]
    expected = sum(
        (evaluation["records"]["new"]["result"]["scores"][axis] - mean) / scale * weight
        for axis, mean, scale, weight in zip(
            combination["axes"],
            combination["means"],
            combination["scales"],
            combination["weights"],
            strict=True,
        )
    )
    assert row["score"] == pytest.approx(expected)
    assert sum(row["contributions"].values()) == pytest.approx(expected)
    assert row["source_result"]["diagnostics"]["why"] == "retained evidence"
    assert "score" not in applied["records"]["failed"]
    assert "score" not in applied["records"]["absent"]
    assert applied["records"]["failed"]["error"] == evaluation["records"]["failed"]["error"]
    assert combination == frozen_copy
    assert apply_combination(evaluation, json.loads(json.dumps(combination))) == applied
    changed = copy.deepcopy(evaluation)
    changed["records"]["new"]["result"]["scores"]["quality"] += 1
    assert apply_combination(changed, combination)["application_id"] != applied["application_id"]
    changed["config"]["revision"] = "other"
    with pytest.raises(ValueError, match="recipe differs"):
        apply_combination(changed, combination)
    changed = copy.deepcopy(combination)
    changed["weights"][0] *= 2
    with pytest.raises(ValueError, match="digest"):
        apply_combination(evaluation, changed)
    changed = copy.deepcopy(evaluation)
    del changed["records"]["new"]["result"]["scores"]["damage"]
    with pytest.raises(ValueError, match="axis missing"):
        apply_combination(changed, combination)
    # A correctly hashed but numerically invalid artifact is still rejected.
    changed = copy.deepcopy(combination)
    changed["scales"][0] = 0.0
    changed["combination_id"] = canonical_json_sha256(
        {k: v for k, v in changed.items() if k != "combination_id"}, allow_nan=False
    )
    with pytest.raises(ValueError, match="vectors"):
        apply_combination(evaluation, changed)
    changed["scales"][0] = 1e-320
    changed["combination_id"] = canonical_json_sha256(
        {k: v for k, v in changed.items() if k != "combination_id"}, allow_nan=False
    )
    with pytest.raises(ValueError, match="arithmetic is nonfinite"):
        apply_combination(evaluation, changed)
