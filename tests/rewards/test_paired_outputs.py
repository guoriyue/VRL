"""Paired generator comparisons weight sources equally and retain missing evidence."""

import copy

import pytest

from vrl.rewards.diagnostics import compare_paired_outputs


def test_repeated_draws_are_grouped_and_failed_pairs_never_become_zero_scores():
    baselines, candidates = [], []
    for repeat in range(3):
        baseline = {"run_id": f"base-{repeat}", "config": {"revision": "fixed"}, "records": {}}
        candidate = {
            "run_id": f"candidate-{repeat}",
            "config": {"revision": "fixed"},
            "records": {},
        }
        for sample in ("a", "b", "failed") if repeat == 0 else ("a",):
            inputs = {
                "sample_id": sample,
                "prompt_id": sample,
                "prompt": "task",
                "assets": {"reference_image": sample},
                "asset_sha256": {"reference_image": sample},
                "metadata": {"seed": repeat, "opacity": 64 if sample == "a" else 255},
                "sha256": f"base-{repeat}-{sample}",
            }
            baseline["records"][sample] = {
                "input": inputs,
                "status": "success",
                "result": {"scores": {"quality": 0.0}},
            }
            candidate["records"][sample] = {
                "input": {**copy.deepcopy(inputs), "sha256": f"candidate-{repeat}-{sample}"},
                "status": "success",
                "result": {"scores": {"quality": 1.0 if sample == "a" else -1.0}},
            }
        if repeat == 0:
            candidate["records"]["failed"] = {
                "input": candidate["records"]["failed"]["input"],
                "status": "error",
                "error": {"type": "TimeoutError", "message": "unmeasured"},
            }
        baselines.append(baseline)
        candidates.append(candidate)
    report = compare_paired_outputs(baselines, candidates, axis="quality")
    assert not report["complete"] and report["scored_pairs"] == 4
    assert report["expected_pairs"] == 5 and report["expected_source_groups"] == 3
    assert report["scored_source_groups"] == 2
    # Three improving draws from one source must not outweigh one declining source.
    assert report["directed_delta"]["mean"] == 0.0
    assert report["per_source"]["reference:a"]["scored_pairs"] == 3
    failed = next(row for row in report["observations"] if row["sample_id"] == "failed")
    assert failed["baseline"] == 0.0 and failed["candidate"] is None
    assert failed["directed_delta"] is None and failed["candidate_error"]["type"] == "TimeoutError"
    assert report == compare_paired_outputs(baselines, candidates, axis="quality")
    stratified = compare_paired_outputs(
        baselines, candidates, axis="quality", stratify_by="opacity"
    )
    assert stratified["directed_delta"] == report["directed_delta"]
    strata = {row["value"]: row["comparison"] for row in stratified["stratification"]["strata"]}
    assert strata[64]["scored_pairs"] == 3 and strata[64]["scored_source_groups"] == 1
    assert strata[64]["directed_delta"]["mean"] == 1.0
    assert strata[255]["directed_delta"]["mean"] == -1.0
    assert not strata[255]["complete"] and strata[255]["expected_pairs"] == 2
    assert strata[255]["scored_pairs"] == 1
    assert strata[255]["observations"][-1]["candidate_error"]["type"] == "TimeoutError"
    assert stratified["comparison_id"] != report["comparison_id"]
    with pytest.raises(ValueError, match="stratification metadata missing"):
        compare_paired_outputs(baselines, candidates, axis="quality", stratify_by="absent")
    inverted = compare_paired_outputs(
        baselines, candidates, axis="quality", direction=-1, stratify_by="opacity"
    )
    assert inverted["per_source"]["reference:a"]["directed_delta"] == -1.0
    assert (
        next(
            row["comparison"]["directed_delta"]["mean"]
            for row in inverted["stratification"]["strata"]
            if row["value"] == 64
        )
        == -1.0
    )
    assert inverted["comparison_id"] != report["comparison_id"]
    changed = copy.deepcopy(candidates)
    changed[0]["records"]["a"]["input"]["metadata"]["seed"] = 999
    with pytest.raises(ValueError, match="task inputs differ"):
        compare_paired_outputs(baselines, changed, axis="quality")
    changed = copy.deepcopy(candidates)
    changed[0]["config"]["revision"] = "other"
    with pytest.raises(ValueError, match="recipes differ"):
        compare_paired_outputs(baselines, changed, axis="quality")
    with pytest.raises(ValueError, match="axis missing"):
        compare_paired_outputs(baselines, candidates, axis="absent")
    changed = copy.deepcopy(candidates)
    for run in changed:
        for row in run["records"].values():
            row["status"] = "missing"
            row.pop("result", None)
    missing = compare_paired_outputs(baselines, changed, axis="quality", stratify_by="opacity")
    assert missing["scored_pairs"] == 0 and missing["directed_delta"] is None
    assert missing["source_bootstrap_95ci"] is None
    assert all(
        row["comparison"]["scored_pairs"] == 0 and row["comparison"]["directed_delta"] is None
        for row in missing["stratification"]["strata"]
    )


def test_strata_preserve_json_category_types_and_reject_mismatched_task_metadata():
    baseline = {"run_id": "base", "config": {}, "records": {}}
    for sample, category in (("boolean", True), ("integer", 1), ("text", "1")):
        baseline["records"][sample] = {
            "input": {
                "sample_id": sample,
                "prompt_id": sample,
                "prompt": "task",
                "sha256": sample,
                "metadata": {"category": category},
            },
            "status": "success",
            "result": {"scores": {"quality": 0.0}},
        }
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "candidate"
    candidate["records"]["integer"]["result"]["scores"]["quality"] = -1.0
    report = compare_paired_outputs(
        [baseline], [candidate], axis="quality", stratify_by="category"
    )
    strata = report["stratification"]["strata"]
    assert len(strata) == 3
    integer = next(row for row in strata if type(row["value"]) is int)
    assert integer["comparison"]["directed_delta"]["mean"] == -1.0
    candidate["records"]["integer"]["input"]["metadata"]["category"] = True
    with pytest.raises(ValueError, match="task inputs differ"):
        compare_paired_outputs([baseline], [candidate], axis="quality", stratify_by="category")
