"""Independent scorers align exactly and keep held-out recipes separate from data."""

import copy

import pytest

from reward_lab.calibration import PreferencePair, evaluate_combination, fit_combination
from reward_lab.diagnostics import join_evaluations


def test_join_binds_observations_and_supports_source_disjoint_holdout():
    # Synthetic values exercise fitting contracts; these are not human labels.
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
        evaluations[name] = {
            "config": {"revision": name + "-v1"},
            "run_id": name,
            "records": records,
        }
    joined = join_evaluations(evaluations)
    assert joined == join_evaluations(dict(reversed(list(evaluations.items()))))
    row = joined["records"]["0-left"]
    assert row["result"]["scores"] == {"semantic/score": 2, "locality/score": 1}
    assert row["source_results"]["semantic"]["reward_model_version"] == "semantic-v1"
    assert row["result"]["timing_ms"] == {"semantic/inference": 2, "locality/inference": 2}
    changed = copy.deepcopy(evaluations)
    changed["semantic"]["records"]["0-left"]["result"]["scores"]["score"] = 3
    assert join_evaluations(changed)["run_id"] != joined["run_id"]
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
        calibration[name]["records"] = {
            k: v for k, v in calibration[name]["records"].items() if not k.startswith("2-")
        }
        holdout[name]["records"] = {
            k: v for k, v in holdout[name]["records"].items() if k.startswith("2-")
        }
        holdout[name]["run_id"] += "-holdout"
    fitted = fit_combination(
        join_evaluations(calibration), pairs[:2], axes=["semantic/score", "locality/score"]
    )
    assert (
        evaluate_combination(join_evaluations(holdout), pairs[2:], fitted)[
            "source_balanced_agreement"
        ]
        == 1
    )
    holdout["locality"]["config"]["revision"] = "changed-verifier"
    with pytest.raises(ValueError, match="recipe differs"):
        evaluate_combination(join_evaluations(holdout), pairs[2:], fitted)

    damaged = copy.deepcopy(evaluations)
    damaged["locality"]["records"].pop("0-left")
    with pytest.raises(ValueError, match="sample grids differ"):
        join_evaluations(damaged)
    damaged = copy.deepcopy(evaluations)
    damaged["locality"]["records"]["0-left"]["input"]["asset_sha256"]["reference_image"] = (
        "different-source"
    )
    with pytest.raises(ValueError, match="inputs differ"):
        join_evaluations(damaged)
    damaged = copy.deepcopy(evaluations)
    damaged["locality"]["records"]["0-left"]["status"] = "missing"
    with pytest.raises(ValueError, match="complete scoring"):
        join_evaluations(damaged)
    with pytest.raises(ValueError, match="aliases"):
        join_evaluations({"a/b": evaluations["semantic"]})
