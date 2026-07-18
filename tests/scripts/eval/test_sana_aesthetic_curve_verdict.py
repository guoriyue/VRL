from __future__ import annotations

import csv

import pytest

from vrl.scripts.eval import sana_aesthetic_curve_verdict as verdict
from vrl.scripts.eval.sana_aesthetic_curve_verdict import evaluate


def _train_rows(count: int = 300) -> list[dict[str, float]]:
    return [
        {
            "epoch": float(index),
            "reward_std": 0.5,
            "grad_norm": 0.1,
            "trained_prompt_num": 8.0,
            "active_clip_fraction": 0.0,
            "logprob_abs_diff_max": 0.0,
            "pre_update_logprob_abs_diff_max": 0.0,
            "pre_update_clip_fraction": 0.0,
            "pre_update_active_clip_fraction": 0.0,
        }
        for index in range(count)
    ]


def _eval_rows(*, pickscore_end: float = 0.80) -> list[dict[str, float]]:
    rows = [
        {
            "epoch": -1.0,
            "eval_reward_stderr": 0.02,
            "r_aesthetic": 5.0,
            "r_pickscore": 0.80,
        }
    ]
    for index in range(12):
        rows.append(
            {
                "epoch": float(24 + 25 * index),
                "eval_reward_stderr": 0.02,
                "r_aesthetic": 5.02 + 0.02 * index,
                "r_pickscore": pickscore_end,
            }
        )
    return rows


def test_passes_only_with_complete_curve_and_qualitative_gate() -> None:
    result = evaluate(_eval_rows(), _train_rows(), qualitative_audit="pass")
    assert result["verdict"] == "PASS"
    assert result["failures"] == []


def test_fails_reward_hacking_even_when_aesthetic_rises() -> None:
    result = evaluate(
        _eval_rows(pickscore_end=0.70),
        _train_rows(),
        qualitative_audit="pass",
    )
    assert result["verdict"] == "FAIL"
    assert any("PickScore regressed" in failure for failure in result["failures"])


def test_fails_incomplete_run_and_pending_visual_audit() -> None:
    result = evaluate(_eval_rows(), _train_rows(299), qualitative_audit="pending")
    assert result["verdict"] == "FAIL"
    assert any("incomplete" in failure for failure in result["failures"])
    assert any("qualitative" in failure for failure in result["failures"])


def test_fails_when_only_endpoint_window_exists_without_full_curve() -> None:
    result = evaluate(_eval_rows()[:-1], _train_rows(), qualitative_audit="pass")

    assert result["verdict"] == "FAIL"
    assert result["criteria"]["min_post_eval_points"] == 12
    assert any("at least 12 post-training points" in failure for failure in result["failures"])


def test_fails_pre_update_logprob_parity_error() -> None:
    train_rows = _train_rows()
    train_rows[10]["pre_update_logprob_abs_diff_max"] = 0.02

    result = evaluate(_eval_rows(), train_rows, qualitative_audit="pass")

    assert result["verdict"] == "FAIL"
    assert any("pre-update logprob parity" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    "field",
    [
        "pre_update_clip_fraction",
        "pre_update_active_clip_fraction",
        "active_clip_fraction",
    ],
)
def test_single_pass_fullparam_rejects_any_policy_clip(field: str) -> None:
    train_rows = _train_rows()
    train_rows[10][field] = 0.01

    result = evaluate(_eval_rows(), train_rows, qualitative_audit="pass")

    assert result["verdict"] == "FAIL"
    assert any("clip" in failure for failure in result["failures"])


def test_eval_reader_keeps_historical_inline_csv_compatibility(tmp_path) -> None:
    path = tmp_path / "eval_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(_eval_rows()[0]))
        writer.writeheader()
        writer.writerows(_eval_rows())
    (tmp_path / "resolved_config.yaml").write_text(
        "trainer:\n  eval:\n    enabled: true\n",
        encoding="utf-8",
    )

    assert verdict._read_eval_rows(tmp_path) == _eval_rows()


def test_eval_reader_rejects_unmarked_csv_from_new_run(tmp_path) -> None:
    (tmp_path / "eval_metrics.csv").write_text("epoch,r_aesthetic\n-1,5.0\n")
    (tmp_path / "resolved_config.yaml").write_text("trainer: {}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="sana_aesthetic_checkpoint_eval"):
        verdict._read_eval_rows(tmp_path)


def test_eval_reader_rejects_missing_standalone_report(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="sana_aesthetic_checkpoint_eval"):
        verdict._read_eval_rows(tmp_path)


def test_main_reports_missing_standalone_report(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit, match="standalone SANA evaluation report"):
        verdict.main(["--run-dir", str(tmp_path)])
    assert capsys.readouterr().out == ""


def test_verdict_parser_does_not_accept_arbitrary_eval_csv(tmp_path) -> None:
    with pytest.raises(SystemExit) as raised:
        verdict.main(
            ["--run-dir", str(tmp_path), "--eval-csv", str(tmp_path / "foreign.csv")],
        )
    assert raised.value.code == 2
