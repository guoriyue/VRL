from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from vrl.scripts.eval.score_report import main, write_curve_report


def _rows() -> list[dict]:
    return [
        {
            "checkpoint_label": label,
            "epoch": epoch,
            "prompt_index": prompt_index,
            "sample_index": sample_index,
            "seed": 10 + prompt_index * 2 + sample_index,
            "prompt": f"prompt {prompt_index}",
            "r_quality": value + delta,
            "r_alignment": 1 - value / 10,
        }
        for label, epoch, delta in (("base", 0, 0), ("checkpoint-4", 4, 1))
        for prompt_index, values in enumerate(((0, 2), (5,)))
        for sample_index, value in enumerate(values)
    ]


def test_curve_weights_prompts_not_samples_and_publishes_outputs(tmp_path: Path) -> None:
    summary = write_curve_report(_rows(), tmp_path, bootstrap_resamples=100)

    base = summary["arms"]["base"]
    trained = summary["arms"]["checkpoint-4"]
    assert summary["statistical_unit"] == "prompt"
    assert base["sample_count"] == 3
    assert base["prompt_count"] == 2
    assert base["scores"]["quality"]["absolute"]["mean"] == 3
    paired = trained["scores"]["quality"]["paired_delta_from_base"]
    assert paired["mean"] == 1
    assert paired["count"] == 2
    assert paired["bootstrap_95ci"] == [1, 1]
    assert paired["clear_increase"] is True
    assert paired["clear_decrease"] is False
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with (tmp_path / "curve.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    with Image.open(tmp_path / "curve.png") as image:
        assert image.size == (1120, 560)
        assert image.getextrema() != ((255, 255),) * 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "duplicate score cell"),
        ("grid", "paired score grid differs"),
        ("prompt", "prompt mismatch"),
        ("seed", "seed mismatch"),
        ("epoch", "conflicting epochs"),
        ("nonfinite", "non-finite"),
        ("missing_score", "missing or invalid"),
    ],
)
def test_curve_rejects_invalid_comparisons(tmp_path: Path, change: str, message: str) -> None:
    rows = _rows()
    if change == "duplicate":
        rows.append(dict(rows[-1]))
    elif change == "grid":
        rows.pop()
    elif change in {"seed", "epoch"}:
        rows[-1][change] += 1
    elif change == "prompt":
        rows[-1]["prompt"] = "different prompt"
    elif change == "nonfinite":
        rows[-1]["r_quality"] = float("nan")
    elif change == "missing_score":
        del rows[-1]["r_quality"]

    with pytest.raises(ValueError, match=message):
        write_curve_report(rows, tmp_path, bootstrap_resamples=10)
    assert not (tmp_path / "summary.json").exists()


def test_report_cli_scores_existing_rows_without_models(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps(row) for row in _rows()))
    output = tmp_path / "report"

    main(
        [
            "--scores",
            str(scores),
            "--output-dir",
            str(output),
            "--score-key",
            "quality",
            "--bootstrap-resamples",
            "50",
            "--seed",
            "7",
        ],
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["bootstrap"] == {"resamples": 50, "seed": 7}
    assert set(summary["arms"]["base"]["scores"]) == {"quality"}
    repeated = write_curve_report(
        _rows(),
        output,
        score_keys=["quality"],
        bootstrap_resamples=50,
        seed=7,
    )
    assert summary == repeated


def test_joint_score_is_the_fraction_passing_every_threshold(tmp_path: Path, capsys) -> None:
    """``--joint-min`` scores each sample 1 only when ALL listed components pass.

    Base never clears alignment >= 0.9 together with quality >= 1; the trained arm
    does on one of prompt 0's two samples, so its passing fraction is 0.5 on
    prompt 0, 0 on prompt 1, and 0.25 over prompts. Alignment is not a reported
    score key, yet it gates the joint score.
    """
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps(row) for row in _rows()))

    main(
        [
            "--scores",
            str(scores),
            "--output-dir",
            str(tmp_path / "report"),
            "--score-key",
            "quality",
            "--joint-min",
            "quality=1",
            "--joint-min",
            "alignment=0.9",
            "--bootstrap-resamples",
            "50",
        ],
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["joint_min"] == {"quality": 1.0, "alignment": 0.9}
    assert set(summary["arms"]["base"]["scores"]) == {"quality", "joint"}
    assert summary["arms"]["base"]["scores"]["joint"]["absolute"]["mean"] == 0
    joint = summary["arms"]["checkpoint-4"]["scores"]["joint"]
    lower, upper = joint["absolute"]["bootstrap_95ci"]
    assert joint["absolute"]["mean"] == 0.25
    assert lower <= 0.25 <= upper
    assert joint["paired_delta_from_base"]["mean"] == 0.25
    assert joint["paired_delta_from_base"]["win_rate"] == 0.5

    with pytest.raises(ValueError, match="missing or invalid r_typo"):
        write_curve_report(_rows(), tmp_path / "typo", joint_min={"typo": 0.0})
