"""Persisted reward failures remain visible and ranking comparisons stay paired."""

import copy
import json

import pytest
from PIL import Image

from reward_lab.diagnostics import compare_rankings, health_report
from vrl.rewards.evaluation import ScoringConfig, read_evaluation, rescore_media
from vrl.utils.json_files import canonical_json_sha256


@pytest.mark.asyncio
async def test_health_reads_real_scores_and_exposes_missing_and_tampered_records(tmp_path):
    Image.new("RGB", (12, 12), "white").save(tmp_path / "image.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": name,
                    "prompt_id": "white",
                    "prompt": "white square",
                    "path": "image.png",
                }
            )
            for name in ("a", "b")
        )
    )
    config = ScoringConfig(
        name="sharpness",
        revision="v1",
        preprocessing_revision="native",
        rubric_revision="laplacian",
        worker_config={
            "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
            "device": "cpu",
        },
    )
    output = tmp_path / "scores"
    await rescore_media(manifest, config, output)
    evaluation = read_evaluation(output)
    report = health_report(evaluation)
    assert report["status_counts"] == {"success": 2}
    assert report["axes"]["image_sharpness"]["zero_spread_prompt_ids"] == ["white"]
    path = output / "samples" / f"{canonical_json_sha256('b', allow_nan=False)}.json"
    original = path.read_text()
    path.unlink()
    assert health_report(read_evaluation(output))["status_counts"] == {"success": 1, "missing": 1}
    with pytest.raises(ValueError, match="complete scoring"):
        compare_rankings(
            evaluation,
            read_evaluation(output),
            first_axis="image_sharpness",
            second_axis="image_sharpness",
        )
    record = json.loads(original)
    record["input"]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incompatible sample"):
        read_evaluation(output)
    path.write_text(original)
    provenance = output / "provenance.json"
    changed = json.loads(provenance.read_text())
    changed["config"]["revision"] = "unrecorded-revision"
    provenance.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest mismatch"):
        read_evaluation(output)


def test_candidate_ranking_uses_prompt_units_and_retains_tie_disagreements():
    records = {}
    for sample_id, prompt_id, score in [
        ("a", "p", 0),
        ("b", "p", 1),
        ("c", "q", 0),
        ("d", "q", 1),
        ("e", "q", 2),
    ]:
        records[sample_id] = {
            "input": {"sample_id": sample_id, "prompt_id": prompt_id},
            "status": "success",
            "result": {"scores": {"quality": score}},
        }
    first = {"run_id": "first", "records": records}
    second = copy.deepcopy(first)
    second["run_id"] = "second"
    for row in second["records"].values():
        if row["input"]["prompt_id"] == "q":
            row["result"]["scores"]["quality"] *= -1
    report = compare_rankings(first, second, first_axis="quality", second_axis="quality")
    # One agreeing prompt, one reversed prompt; not 1 agreeing pair out of 4.
    assert report["prompt_balanced_agreement"] == 0.5
    assert len(report["disagreements"]) == 3
    assert report["per_prompt"]["q"]["reversed"] == 3
    assert report == compare_rankings(first, second, first_axis="quality", second_axis="quality")
    second["records"]["b"]["result"]["scores"]["quality"] = 0.01
    report = compare_rankings(
        first, second, first_axis="quality", second_axis="quality", second_tie_epsilon=0.02
    )
    assert report["per_prompt"]["p"]["tie_disagreement"] == 1
    second["records"]["a"]["input"]["prompt_id"] = "different-source"
    with pytest.raises(ValueError, match="inputs differ"):
        compare_rankings(first, second, first_axis="quality", second_axis="quality")


def test_health_keeps_errors_and_variable_axis_coverage_separate_from_zero():
    evaluation = {
        "run_id": "test",
        "records": {
            "a": {
                "status": "success",
                "input": {"prompt_id": "p"},
                "result": {"scores": {"quality": 0.0, "alignment": 0.7}},
            },
            "b": {
                "status": "success",
                "input": {"prompt_id": "p"},
                "result": {"scores": {"quality": 0.0}},
            },
            "c": {
                "status": "error",
                "input": {"prompt_id": "p"},
                "error": {"type": "TimeoutError", "message": "timeout"},
            },
        },
    }
    report = health_report(evaluation)
    assert report["error_types"] == {"TimeoutError": 1}
    assert report["axes"]["alignment"]["missing_from_successful"] == 1
    assert report["axes"]["alignment"]["zero_spread_fraction"] is None
    assert report["axes"]["quality"]["distribution"]["count"] == 2
