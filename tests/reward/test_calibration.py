"""Frozen fitting never uses holdout labels; application and review keep evidence intact."""

import copy
import json

import pytest
from PIL import Image

from reward.__main__ import main
from reward.calibration import Calibration, PreferencePair
from vrl.rewards.evaluation import Evaluation
from vrl.utils.artifacts import sha256_file


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
    evaluation = Evaluation("test", {"revision": "v1"}, records)
    calibration = Calibration(evaluation)
    fitted = calibration.fit(pairs, axes=["quality", "constant"])
    assert fitted["weights"][0] > 0 and fitted["weights"][1] == 0
    assert fitted["means"] == [1.0, 7.0]
    assert fitted["fit_pairs"] == 2
    changed_pairs = [
        p.model_copy(update={"preference": "right"}) if p.split == "holdout" else p for p in pairs
    ]
    # Even contradictory holdout judgments cannot change the frozen coefficients.
    assert calibration.fit(changed_pairs, axes=["quality", "constant"]) == fitted
    report = calibration.evaluate(pairs, fitted)
    assert report["source_balanced_agreement"] == 1.0
    assert report["judged_source_groups"] == 3
    assert report["unsure_count"] == 1
    assert report["outcomes"][2]["prediction"] == "tie"
    assert report["outcomes"][3]["correct"] is None
    assert calibration.evaluate(changed_pairs, fitted)["source_balanced_agreement"] == 0.0

    changed = copy.deepcopy(evaluation)
    changed.records["2-left"]["input"]["asset_sha256"]["reference_image"] = "source-0"
    with pytest.raises(ValueError, match="leakage"):
        Calibration(changed).fit(pairs, axes=["quality"])
    changed = copy.deepcopy(evaluation)
    for sample_id in ("2-left", "2-right"):
        changed.records[sample_id]["input"]["prompt_id"] = "p0"
    with pytest.raises(ValueError, match="used for calibration"):
        Calibration(changed).evaluate(pairs[2:], fitted)
    changed = copy.deepcopy(evaluation)
    changed.config["revision"] = "different-model"
    with pytest.raises(ValueError, match="recipe differs"):
        Calibration(changed).evaluate(pairs, fitted)


def test_calibration_rejects_cross_prompt_pairs_and_missing_measurements():
    evaluation = Evaluation(
        "test",
        {},
        {
            "a": {"status": "success", "input": {"prompt_id": "one", "sha256": "a"}},
            "b": {"status": "success", "input": {"prompt_id": "two", "sha256": "b"}},
        },
    )
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
        Calibration(evaluation).fit(pairs, axes=["quality"])
    evaluation.records["b"]["status"] = "error"
    with pytest.raises(ValueError, match="unscored"):
        Calibration(evaluation).fit(pairs, axes=["quality"])


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
    fitting = Evaluation("fit", {"revision": "frozen"}, records)
    combination = Calibration(fitting).fit(pairs, axes=["quality", "damage"])
    frozen_copy = copy.deepcopy(combination)
    assert combination["weights"][0] > 0 > combination["weights"][1]
    evaluation = Evaluation(
        "new",
        fitting.config,
        {
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
    )
    applied = Calibration(evaluation).apply(combination)
    row = applied["records"]["new"]
    expected = sum(
        (evaluation.records["new"]["result"]["scores"][axis] - mean) / scale * weight
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
    assert applied["records"]["failed"]["error"] == evaluation.records["failed"]["error"]
    assert applied["status_counts"] == {"success": 1, "error": 1, "missing": 1}
    assert combination == frozen_copy
    assert Calibration(evaluation).apply(json.loads(json.dumps(combination))) == applied
    changed = copy.deepcopy(evaluation)
    changed.config["revision"] = "other"
    with pytest.raises(ValueError, match="recipe differs"):
        Calibration(changed).apply(combination)
    changed = copy.deepcopy(evaluation)
    del changed.records["new"]["result"]["scores"]["damage"]
    with pytest.raises(ValueError, match="axis missing"):
        Calibration(changed).apply(combination)
    changed = copy.deepcopy(combination)
    changed["scales"][0] = 0.0
    with pytest.raises(ValueError, match="vectors"):
        Calibration(evaluation).apply(changed)
    changed["scales"][0] = 1e-320
    with pytest.raises(ValueError, match="arithmetic is nonfinite"):
        Calibration(evaluation).apply(changed)


def test_review_randomizes_without_scores_and_round_trips_only_answered_pairs(tmp_path):
    records, pairs = {}, []
    for index in range(4):
        reference = tmp_path / f"source-{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(reference)
        for side, color in (("left", (20 + index, 0, 0)), ("right", (40 + index, 0, 0))):
            sample_id = f"private-model-{index}-{side}"
            path = tmp_path / f"{sample_id}.png"
            Image.new("RGBA", (4, 4), (*color, 127)).save(path)
            records[sample_id] = {
                "status": "success",
                "input": {
                    "sample_id": sample_id,
                    "prompt_id": f"prompt-{index}",
                    "prompt": "literal </script><script>alert(1)</script> text",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "assets": {"reference_image": str(reference)},
                    "asset_sha256": {"reference_image": sha256_file(reference)},
                },
                "result": {"scores": {"private_score_axis": 12345.678}},
            }
        pairs.append(
            {
                "pair_id": f"annotation-{index}",
                "left": f"private-model-{index}-left",
                "right": f"private-model-{index}-right",
                "source_group": f"source-{index}",
                "split": "calibration" if index < 2 else "holdout",
            }
        )
    calibration = Calibration(Evaluation("scoring-snapshot", {}, records))
    output = tmp_path / "review"
    result = calibration.review_packet(pairs, output, seed=42)
    audit = json.loads((output / "audit.json").read_text())
    html = (output / "index.html").read_text()
    assert "private-model" not in html and "private_score_axis" not in html
    assert "12345.678" not in html and "</script><script>alert" not in html
    assert {entry["reversed"] for entry in audit["mapping"].values()} == {False, True}
    assert "preference" not in next(iter(audit["mapping"].values()))["pair"]
    for entry in audit["display"]:
        mapping = audit["mapping"][entry["key"]]
        expected = mapping["right_sha256"] if mapping["reversed"] else mapping["left_sha256"]
        assert sha256_file(output / entry["a"]["path"]) == expected
        assert Image.open(output / entry["a"]["path"]).mode == "RGBA"
    repeated = calibration.review_packet(pairs, tmp_path / "repeat", seed=42)
    assert repeated["review_id"] == result["review_id"]

    keys = list(audit["mapping"])
    # Leave the final pair unanswered: it must not be invented as a tie/unsure.
    answers = {
        "review_id": result["review_id"],
        "answers": dict(zip(keys[:3], ["a", "b", "tie"], strict=True)),
    }
    restored = PreferencePair.from_review(audit, answers)
    assert len(restored) == 3
    for key, pair in zip(keys, restored, strict=False):
        shown_answer = answers["answers"][key]
        if shown_answer == "tie":
            assert pair.preference == "tie"
        else:
            expected = (
                "left" if (shown_answer == "a") != audit["mapping"][key]["reversed"] else "right"
            )
            assert pair.preference == expected
    answer_file = tmp_path / "answers.json"
    answer_file.write_text(json.dumps(answers))
    labels = tmp_path / "preferences.jsonl"
    argv = [
        "review-import",
        "--review",
        str(output / "audit.json"),
        "--answers",
        str(answer_file),
        "--output",
        str(labels),
    ]
    main(argv)
    assert PreferencePair.load_jsonl(labels) == restored
    with pytest.raises(FileExistsError):
        main(argv)

    with pytest.raises(ValueError, match="no explicit"):
        PreferencePair.from_review(audit, {"review_id": result["review_id"], "answers": {}})
    with pytest.raises(ValueError, match="different"):
        PreferencePair.from_review(audit, {**answers, "review_id": "another-review"})
    with pytest.raises(ValueError, match="unknown"):
        PreferencePair.from_review(
            audit, {"review_id": result["review_id"], "answers": {keys[0]: "auto"}}
        )
    with pytest.raises(ValueError, match="must not contain"):
        calibration.review_packet(
            [{**pairs[0], "preference": "left"}], tmp_path / "prelabeled", seed=42
        )
    assert not (tmp_path / "prelabeled").exists()
    assert not list(tmp_path.glob(".prelabeled-*"))
