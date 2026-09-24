"""Blinded review preserves media identity and maps only explicit human answers."""

import copy
import json

import pytest
from PIL import Image

from vrl.rewards.annotation import export_preference_review, import_preference_review
from vrl.rewards.calibration import load_preferences
from vrl.scripts.rewards.calibrate_scores import main
from vrl.utils.artifacts import sha256_file


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
    evaluation = {"run_id": "scoring-snapshot", "records": records}
    output = tmp_path / "review"
    result = export_preference_review(evaluation, pairs, output, seed=42)
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
    repeated = export_preference_review(evaluation, pairs, tmp_path / "repeat", seed=42)
    assert repeated["review_id"] == result["review_id"]

    keys = list(audit["mapping"])
    # Leave the final pair unanswered: it must not be invented as a tie/unsure.
    answers = {
        "review_id": result["review_id"],
        "answers": dict(zip(keys[:3], ["a", "b", "tie"], strict=True)),
    }
    restored = import_preference_review(audit, answers)
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
    main(
        [
            "import-review",
            "--review",
            str(output / "audit.json"),
            "--answers",
            str(answer_file),
            "--output",
            str(labels),
        ]
    )
    assert load_preferences(labels) == restored
    with pytest.raises(FileExistsError):
        main(
            [
                "import-review",
                "--review",
                str(output / "audit.json"),
                "--answers",
                str(answer_file),
                "--output",
                str(labels),
            ]
        )

    tampered = copy.deepcopy(audit)
    tampered["mapping"][keys[0]]["reversed"] = not tampered["mapping"][keys[0]]["reversed"]
    with pytest.raises(ValueError, match="digest"):
        import_preference_review(tampered, answers)
    with pytest.raises(ValueError, match="no explicit"):
        import_preference_review(audit, {"review_id": result["review_id"], "answers": {}})
    with pytest.raises(ValueError, match="different"):
        import_preference_review(audit, {**answers, "review_id": "another-review"})
    with pytest.raises(ValueError, match="unknown"):
        import_preference_review(
            audit, {"review_id": result["review_id"], "answers": {keys[0]: "auto"}}
        )
    with pytest.raises(ValueError, match="must not contain"):
        export_preference_review(
            evaluation, [{**pairs[0], "preference": "left"}], tmp_path / "prelabeled", seed=42
        )
    records[pairs[0]["left"]]["input"]["path"] = str(reference)
    with pytest.raises(ValueError, match="changed since scoring"):
        export_preference_review(evaluation, pairs, tmp_path / "changed", seed=42)
    assert not (tmp_path / "changed").exists()
    assert not list(tmp_path.glob(".changed-*"))
