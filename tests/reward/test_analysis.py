"""Reports over scoring runs keep failures visible and weight sources equally."""

import copy
import json
import math

import numpy as np
import pytest
from PIL import Image

from reward.analysis import Analysis
from reward.calibration import Calibration, PreferencePair
from reward.stress import build_stress_manifest
from vrl.rewards.evaluation import Evaluation, ScoringConfig, load_media_manifest
from vrl.utils.artifacts import sha256_file


def _sharpness(**overrides) -> ScoringConfig:
    return ScoringConfig(
        name="sharpness",
        revision="v1",
        preprocessing_revision="native",
        rubric_revision="laplacian",
        worker_config={
            "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
            "device": "cpu",
            **overrides,
        },
    )


@pytest.mark.asyncio
async def test_health_reads_real_scores_and_keeps_missing_records(tmp_path):
    Image.new("RGB", (12, 12), "white").save(tmp_path / "image.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": n,
                    "prompt_id": "white",
                    "prompt": "white square",
                    "path": "image.png",
                }
            )
            for n in ("a", "b")
        )
    )
    output = tmp_path / "scores"
    evaluation = await Evaluation.score(manifest, _sharpness(), output)
    report = Analysis(evaluation).health()
    assert report["status_counts"] == {"success": 2}
    assert report["axes"]["image_sharpness"]["zero_spread_prompt_ids"] == ["white"]
    next(p for p in (output / "samples").glob("*.json")).unlink()
    partial = Evaluation.load(output)
    assert Analysis(partial).health()["status_counts"] == {"success": 1, "missing": 1}
    with pytest.raises(ValueError, match="complete scoring"):
        Analysis(evaluation).compare_rankings(
            partial, first_axis="image_sharpness", second_axis="image_sharpness"
        )


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
    first = Evaluation("first", {}, records)
    second = Evaluation("second", {}, copy.deepcopy(records))
    for row in second.records.values():
        if row["input"]["prompt_id"] == "q":
            row["result"]["scores"]["quality"] *= -1
    report = Analysis(first).compare_rankings(second, first_axis="quality", second_axis="quality")
    # One agreeing prompt, one reversed prompt; not 1 agreeing pair out of 4.
    assert report["prompt_balanced_agreement"] == 0.5
    assert len(report["disagreements"]) == 3
    assert report["per_prompt"]["q"]["reversed"] == 3
    second.records["b"]["result"]["scores"]["quality"] = 0.01
    report = Analysis(first).compare_rankings(
        second, first_axis="quality", second_axis="quality", second_tie_epsilon=0.02
    )
    assert report["per_prompt"]["p"]["tie_disagreement"] == 1
    second.records["a"]["input"]["prompt_id"] = "different-source"
    with pytest.raises(ValueError, match="inputs differ"):
        Analysis(first).compare_rankings(second, first_axis="quality", second_axis="quality")


def test_health_keeps_errors_and_variable_axis_coverage_separate_from_zero():
    evaluation = Evaluation(
        "test",
        {},
        {
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
    )
    report = Analysis(evaluation).health()
    assert report["error_types"] == {"TimeoutError": 1}
    assert report["axes"]["alignment"]["missing_from_successful"] == 1
    assert report["axes"]["alignment"]["zero_spread_fraction"] is None
    assert report["axes"]["quality"]["distribution"]["count"] == 2


def test_repeated_draws_are_grouped_and_failed_pairs_never_become_zero_scores():
    baselines, candidates = [], []
    for repeat in range(3):
        baseline = Evaluation(f"base-{repeat}", {"revision": "fixed"}, {})
        candidate = Evaluation(f"candidate-{repeat}", {"revision": "fixed"}, {})
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
            baseline.records[sample] = {
                "input": inputs,
                "status": "success",
                "result": {"scores": {"quality": 0.0}},
            }
            candidate.records[sample] = {
                "input": {**copy.deepcopy(inputs), "sha256": f"candidate-{repeat}-{sample}"},
                "status": "success",
                "result": {"scores": {"quality": 1.0 if sample == "a" else -1.0}},
            }
        if repeat == 0:
            candidate.records["failed"] = {
                "input": candidate.records["failed"]["input"],
                "status": "error",
                "error": {"type": "TimeoutError", "message": "unmeasured"},
            }
        baselines.append(baseline)
        candidates.append(candidate)
    report = Analysis.paired(baselines, candidates, axis="quality")
    assert not report["complete"] and report["scored_pairs"] == 4
    assert report["expected_pairs"] == 5 and report["expected_source_groups"] == 3
    assert report["scored_source_groups"] == 2
    # Three improving draws from one source must not outweigh one declining source.
    assert report["directed_delta"]["mean"] == 0.0
    assert report["per_source"]["reference:a"]["scored_pairs"] == 3
    failed = next(row for row in report["observations"] if row["sample_id"] == "failed")
    assert failed["baseline"] == 0.0 and failed["candidate"] is None
    assert failed["directed_delta"] is None and failed["candidate_error"]["type"] == "TimeoutError"
    stratified = Analysis.paired(baselines, candidates, axis="quality", stratify_by="opacity")
    assert stratified["directed_delta"] == report["directed_delta"]
    strata = {row["value"]: row["comparison"] for row in stratified["stratification"]["strata"]}
    assert strata[64]["scored_pairs"] == 3 and strata[64]["directed_delta"]["mean"] == 1.0
    assert strata[255]["directed_delta"]["mean"] == -1.0
    assert not strata[255]["complete"] and strata[255]["scored_pairs"] == 1
    with pytest.raises(ValueError, match="stratification metadata missing"):
        Analysis.paired(baselines, candidates, axis="quality", stratify_by="absent")
    inverted = Analysis.paired(baselines, candidates, axis="quality", direction=-1)
    assert inverted["per_source"]["reference:a"]["directed_delta"] == -1.0
    changed = copy.deepcopy(candidates)
    changed[0].records["a"]["input"]["metadata"]["seed"] = 999
    with pytest.raises(ValueError, match="task inputs differ"):
        Analysis.paired(baselines, changed, axis="quality")
    changed = copy.deepcopy(candidates)
    changed[0].config = {"revision": "other"}
    with pytest.raises(ValueError, match="recipes differ"):
        Analysis.paired(baselines, changed, axis="quality")
    with pytest.raises(ValueError, match="axis missing"):
        Analysis.paired(baselines, candidates, axis="absent")


def test_strata_preserve_json_category_types():
    baseline = Evaluation("base", {}, {})
    for sample, category in (("boolean", True), ("integer", 1), ("text", "1")):
        baseline.records[sample] = {
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
    candidate.run_id = "candidate"
    candidate.records["integer"]["result"]["scores"]["quality"] = -1.0
    report = Analysis.paired([baseline], [candidate], axis="quality", stratify_by="category")
    strata = report["stratification"]["strata"]
    assert len(strata) == 3
    integer = next(row for row in strata if type(row["value"]) is int)
    assert integer["comparison"]["directed_delta"]["mean"] == -1.0


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
        runs.append(Evaluation("same-recipe-and-inputs", {"revision": "v1"}, rows))
    report = Analysis.repeatability(runs)
    quality = report["axes"]["quality"]
    assert quality["source_balanced_mean_score_range"] == 2.5
    assert quality["source_balanced_mean_sample_std"] == pytest.approx((0.5 + math.sqrt(8)) / 2)
    assert quality["samples_with_all_scores"] == 2
    assert quality["observation_status_counts"] == {"success": 8, "error": 4}
    assert quality["excluded_source_groups"] == ["failed-source"]
    changed = copy.deepcopy(runs)
    changed[1].config = {"revision": "v2"}
    with pytest.raises(ValueError, match="configurations differ"):
        Analysis.repeatability(changed)
    with pytest.raises(ValueError, match="at least two"):
        Analysis.repeatability(runs[:1])
    runs[1].records["a"]["result"]["scores"] = {"auxiliary": 0.5}
    report = Analysis.repeatability(runs)
    assert report["axes"]["quality"]["observation_status_counts"]["missing_axis"] == 1


def test_join_prefixes_axes_and_supports_source_disjoint_holdout():
    evaluations = {}
    for name in ("semantic", "locality"):
        records = {}
        for index in range(3):
            for side, value in (("left", 2), ("right", 0)):
                sample_id = f"{index}-{side}"
                records[sample_id] = {
                    "status": "success",
                    "input": {
                        "sample_id": sample_id,
                        "prompt_id": f"prompt-{index}",
                        "sha256": f"media-{sample_id}",
                        "asset_sha256": {"reference_image": f"source-{index}"},
                    },
                    "result": {
                        "artifact_id": sample_id,
                        "scores": {"score": value if name == "semantic" else 1},
                        "reward_model_version": name + "-v1",
                        "timing_ms": {"inference": 2},
                    },
                }
        evaluations[name] = Evaluation(name, {"revision": name + "-v1"}, records)
    joined = Analysis.join(evaluations)
    assert joined == Analysis.join(dict(reversed(list(evaluations.items()))))
    row = joined.records["0-left"]
    assert row["result"]["scores"] == {"semantic/score": 2, "locality/score": 1}
    assert row["source_results"]["semantic"]["reward_model_version"] == "semantic-v1"
    pairs = [
        PreferencePair(
            pair_id=f"pair-{i}",
            left=f"{i}-left",
            right=f"{i}-right",
            source_group=f"source-{i}",
            split="calibration" if i < 2 else "holdout",
            preference="left",
        )
        for i in range(3)
    ]
    calibration, holdout = copy.deepcopy(evaluations), copy.deepcopy(evaluations)
    for name in evaluations:
        calibration[name].records = {
            k: v for k, v in calibration[name].records.items() if not k.startswith("2-")
        }
        holdout[name].records = {
            k: v for k, v in holdout[name].records.items() if k.startswith("2-")
        }
    fitted = Calibration(Analysis.join(calibration)).fit(
        pairs[:2], axes=["semantic/score", "locality/score"]
    )
    holdout_report = Calibration(Analysis.join(holdout)).evaluate(pairs[2:], fitted)
    assert holdout_report["source_balanced_agreement"] == 1
    holdout["locality"].config = {"revision": "changed-verifier"}
    with pytest.raises(ValueError, match="recipe differs"):
        Calibration(Analysis.join(holdout)).evaluate(pairs[2:], fitted)
    damaged = copy.deepcopy(evaluations)
    damaged["locality"].records.pop("0-left")
    with pytest.raises(ValueError, match="sample grids differ"):
        Analysis.join(damaged)
    with pytest.raises(ValueError, match="aliases"):
        Analysis.join({"a/b": evaluations["semantic"]})


@pytest.mark.asyncio
async def test_stress_preserves_alpha_except_explicit_probes_and_records_score_changes(tmp_path):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = [255, 0, 0, 255]
    Image.fromarray(rgba).save(tmp_path / "target.png")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "original",
                "prompt_id": "extract",
                "prompt": "Extract red square",
                "path": "target.png",
                "assets": {"target_image": "target.png"},
            }
        )
        + "\n"
    )
    manifest = build_stress_manifest(source, tmp_path / "stress")
    repeated = build_stress_manifest(source, tmp_path / "repeat")
    rows, copies = load_media_manifest(manifest), load_media_manifest(repeated)
    assert [sha256_file(row.path) for row in rows] == [sha256_file(row.path) for row in copies]
    for row in rows:
        variant = row.metadata["reward_stress"]["transform"]
        with Image.open(row.path) as image:
            alpha = np.asarray(image)[..., 3]
        if variant == "opaque_alpha":
            assert (alpha == 255).all()
        elif variant == "empty_alpha":
            assert not alpha.any()
        else:
            np.testing.assert_array_equal(alpha, rgba[..., 3])
    with pytest.raises(FileExistsError):
        build_stress_manifest(source, tmp_path / "stress")
    with pytest.raises(ValueError, match="nested"):
        build_stress_manifest(manifest, tmp_path / "nested")
    # Unsaturated: the default scale clips this hard-edged square at 1.
    evaluation = await Evaluation.score(manifest, _sharpness(scale=10.0), tmp_path / "scores")
    report = Analysis(evaluation).stress()
    # An empty layer composites to a blank canvas; RGB noise is the shortcut the
    # sharpness docstring warns about and must show up as an increase.
    assert report["transforms"]["empty_alpha"]["image_sharpness"]["decreases"] == 1
    assert report["transforms"]["noise_rgb"]["image_sharpness"]["increases"] == 1
    broken = copy.deepcopy(evaluation)
    changed = next(
        row
        for row in broken.records.values()
        if row["input"]["metadata"]["reward_stress"]["transform"] == "noise_rgb"
    )
    changed["status"] = "error"
    changed.pop("result")
    changed["error"] = {"type": "RuntimeError"}
    failed_report = Analysis(broken).stress()
    failed = next(row for row in failed_report["observations"] if row["transform"] == "noise_rgb")
    assert failed["deltas"] == {} and failed["status"] == "error"
    assert "noise_rgb" not in failed_report["transforms"]


def test_spread_reports_success_band_and_within_prompt_signal():
    records = {}
    for sample_id, prompt_id, score in [
        ("a", "p", 0.9),
        ("b", "p", 0.1),
        ("c", "q", 0.1),
        ("d", "q", 0.2),
        ("e", "r", 0.9),
        ("f", "r", 0.9),
    ]:
        records[sample_id] = {
            "input": {"sample_id": sample_id, "prompt_id": prompt_id},
            "status": "success",
            "result": {"scores": {"judge": score}},
        }
    report = Analysis(Evaluation("run", {}, records)).spread(axis="judge", threshold=0.5)
    assert report["success_rate"] == pytest.approx(0.5) and report["in_band"] is True
    assert report["mixed_prompt_share"] == pytest.approx(1 / 3)
    assert report["zero_spread_prompt_share"] == pytest.approx(1 / 3)
    assert report["prompts"]["p"]["mixed"] is True and report["prompts"]["r"]["mixed"] is False
    with pytest.raises(ValueError, match="band"):
        Analysis(Evaluation("run", {}, records)).spread(
            axis="judge", threshold=0.5, band=(0.6, 0.2)
        )
