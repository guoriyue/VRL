"""Tests for the pure aggregation pieces of the Anima GenEval eval."""

from __future__ import annotations

import pytest

from vrl.scripts.eval import anima_geneval_eval as eval_script


def _score(index: int, tag: str, strict: float, why: str = "ok", partial: float | None = None):
    graded = strict if partial is None else partial
    return eval_script.RowScore(
        index=index,
        tag=tag,
        strict=strict,
        partial=graded,
        dense=graded,
        why=why,
        sharpness=10.0 + index,
    )


def _report(label: str, stricts: dict[int, float]) -> dict:
    report = eval_script.summarize(
        [_score(i, "position" if i % 2 else "counting", s) for i, s in stricts.items()]
    )
    report["label"] = label
    return report


def test_summarize_reports_per_task_and_headline_task_mean() -> None:
    report = eval_script.summarize(
        [
            _score(0, "counting", 1.0),
            _score(1, "counting", 0.0, "missing:bus", partial=0.5),
            _score(2, "position", 0.0, "missing:kite;position:above", partial=1 / 3),
        ]
    )
    assert report["per_task"]["counting"] == {
        "n": 2,
        "strict": 0.5,
        "partial": 0.75,
        "dense": 0.75,
    }
    assert report["strict_overall"] == pytest.approx(1 / 3)
    assert report["strict_task_mean"] == pytest.approx(0.25)
    assert report["failure_reasons"] == {"missing": 2, "position": 1}
    assert report["sharpness_x1e-3"]["median"] == 11.0


def test_paired_delta_counts_wins_ties_losses_per_task() -> None:
    base = _report("base", {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0})
    other = _report("ck", {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0})

    overall = eval_script.paired_delta(base, other)
    counting = eval_script.paired_delta(base, other, tag="counting")

    assert (overall["wins"], overall["ties"], overall["losses"]) == (1, 2, 1)
    assert overall["delta"] == 0.0
    assert counting["n"] == 2 and counting["delta"] == pytest.approx(0.5)
    assert len(overall["bootstrap_95ci"]) == 2


def test_paired_delta_rejects_unpaired_rows() -> None:
    base = _report("base", {0: 0.0})
    other = _report("ck", {0: 1.0, 1: 1.0})
    with pytest.raises(ValueError, match="not paired"):
        eval_script.paired_delta(base, other)


def test_format_summary_and_paired_render() -> None:
    base = _report("base", {0: 0.0, 1: 1.0})
    other = _report("ck", {0: 1.0, 1: 1.0})
    for report in (base, other):
        report.update({"row_count": 2, "lora_path": ""})
    text = eval_script.format_summary(other)
    assert "counting" in text and "task mean" in text
    assert "ALL" in eval_script.format_paired(base, other)
