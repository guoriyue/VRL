"""Frozen preference fitting never uses holdout labels or refits holdout scales."""

import copy

import pytest

from reward_lab.calibration import PreferencePair, evaluate_combination, fit_combination


def test_frozen_fit_excludes_holdout_labels_and_retains_ties_unsure_and_leakage_checks():
    # Synthetic judgments test mechanics; they are not human evaluation evidence.
    records, pairs = {}, []
    for index in range(6):
        for side, value in (("left", 2.0), ("right", 0.0)):
            sample_id = f"{index}-{side}"
            records[sample_id] = {
                "status": "success",
                "input": {
                    "sample_id": sample_id,
                    "prompt_id": f"p{index}",
                    "sha256": f"hash-{sample_id}",
                    "asset_sha256": {"reference_image": f"source-{index}"},
                },
                "result": {"scores": {"quality": 1.0 if index == 4 else value, "constant": 7.0}},
            }
        pairs.append(
            PreferencePair(
                pair_id=f"pair-{index}",
                left=f"{index}-left",
                right=f"{index}-right",
                source_group=f"source-{index}",
                split="calibration" if index < 2 else "holdout",
                preference="tie" if index == 4 else "unsure" if index == 5 else "left",
                tags=["synthetic"],
            )
        )
    evaluation = {"run_id": "test", "config": {"revision": "v1"}, "records": records}
    fitted = fit_combination(evaluation, pairs, axes=["quality", "constant"])
    assert fitted["weights"][0] > 0 and fitted["weights"][1] == 0
    assert fitted["means"] == [1.0, 7.0]
    assert fitted["fit_pairs"] == 2
    changed_pairs = [
        p.model_copy(update={"preference": "right"}) if p.split == "holdout" else p for p in pairs
    ]
    # Even contradictory holdout judgments cannot change the frozen coefficients/ID.
    assert fit_combination(evaluation, changed_pairs, axes=["quality", "constant"]) == fitted
    report = evaluate_combination(evaluation, pairs, fitted)
    assert report["source_balanced_agreement"] == 1.0
    assert report["judged_source_groups"] == 3
    assert report["unsure_count"] == 1
    assert report["outcomes"][2]["prediction"] == "tie"
    assert report["outcomes"][3]["correct"] is None
    assert (
        evaluate_combination(evaluation, changed_pairs, fitted)["source_balanced_agreement"] == 0.0
    )

    changed = copy.deepcopy(evaluation)
    changed["records"]["2-left"]["input"]["asset_sha256"]["reference_image"] = "source-0"
    with pytest.raises(ValueError, match="leakage"):
        fit_combination(changed, pairs, axes=["quality"])
    with pytest.raises(ValueError, match="used for calibration"):
        evaluate_combination(changed, pairs[2:], fitted)
    changed = copy.deepcopy(evaluation)
    changed["config"]["revision"] = "different-model"
    with pytest.raises(ValueError, match="recipe differs"):
        evaluate_combination(changed, pairs, fitted)
    changed_fit = copy.deepcopy(fitted)
    changed_fit["weights"][0] *= -1
    with pytest.raises(ValueError, match="digest"):
        evaluate_combination(evaluation, pairs, changed_fit)
    with pytest.raises(RuntimeError, match="converge"):
        fit_combination(evaluation, pairs, axes=["quality"], max_iterations=1)


def test_calibration_rejects_cross_prompt_pairs_and_missing_measurements():
    evaluation = {
        "records": {
            "a": {"status": "success", "input": {"prompt_id": "one", "sha256": "a"}},
            "b": {"status": "success", "input": {"prompt_id": "two", "sha256": "b"}},
        }
    }
    pairs = [
        PreferencePair(
            pair_id="one",
            left="a",
            right="b",
            source_group="group",
            split="calibration",
            preference="left",
        )
    ]
    with pytest.raises(ValueError, match="same prompt"):
        fit_combination(evaluation, pairs, axes=["quality"])
    evaluation["records"]["b"]["status"] = "error"
    with pytest.raises(ValueError, match="unscored"):
        fit_combination(evaluation, pairs, axes=["quality"])
