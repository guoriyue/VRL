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
