"""Repeat-scoring noise retains missing observations and balances source groups."""

import copy
import math

import pytest

from reward_lab.diagnostics import repeatability_report


def test_repeated_scores_distinguish_jitter_missingness_and_source_weights():
    runs = []
    for repeat in range(3):
        rows = {}
        for sample, group, scores in (
            ("a", "shared", [1, 2, 3]),
            ("b", "shared", [5, 5, 5]),
            ("c", "other", [0, 4, None]),
            ("d", "failed-source", [None, None, None]),
        ):
            value = scores[repeat]
            rows[sample] = {
                "input": {
                    "sample_id": sample,
                    "prompt_id": sample,
                    "metadata": {"source_group": group},
                },
                "status": "success" if value is not None else "error",
            }
            if value is not None:
                rows[sample]["result"] = {"scores": {"quality": value}}
            else:
                rows[sample]["error"] = {"type": "Unavailable"}
        # Equal run IDs are valid for repeated identical inputs and recipe.
        runs.append(
            {"run_id": "same-recipe-and-inputs", "config": {"revision": "v1"}, "records": rows}
        )
    report = repeatability_report(runs)
    quality = report["axes"]["quality"]
    assert quality["source_balanced_mean_score_range"] == 2.5
    assert quality["source_balanced_mean_sample_std"] == pytest.approx((0.5 + math.sqrt(8)) / 2)
    assert quality["samples_with_all_scores"] == 2
    assert quality["samples_with_two_scores"] == 3
    assert quality["observation_status_counts"] == {"success": 8, "error": 4}
    assert quality["excluded_source_groups"] == ["failed-source"]
    assert quality["samples"][2]["scores"] == [0, 4, None]
    assert quality["samples"][3]["sample_std"] is None
    changed = copy.deepcopy(runs)
    changed[1]["records"]["a"]["input"]["sha256"] = "different-media"
    with pytest.raises(ValueError, match="inputs differ"):
        repeatability_report(changed)
    changed = copy.deepcopy(runs)
    changed[1]["config"]["revision"] = "v2"
    with pytest.raises(ValueError, match="configurations differ"):
        repeatability_report(changed)
    with pytest.raises(ValueError, match="at least two"):
        repeatability_report(runs[:1])
    runs[1]["records"]["a"]["result"]["scores"] = {"auxiliary": 0.5}
    report = repeatability_report(runs)
    assert report["axes"]["quality"]["observation_status_counts"]["missing_axis"] == 1
    assert report["axes"]["quality"]["samples"][0]["scores"] == [1, None, 3]
    assert report["axes"]["auxiliary"]["source_balanced_mean_score_range"] is None
