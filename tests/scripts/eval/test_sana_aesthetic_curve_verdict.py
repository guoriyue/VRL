from __future__ import annotations

from vrl.scripts.eval.sana_aesthetic_curve_verdict import evaluate


def _train_rows(count: int = 300) -> list[dict[str, float]]:
    return [
        {
            "epoch": float(index),
            "reward_std": 0.5,
            "grad_norm": 0.1,
            "trained_prompt_num": 8.0,
            "clip_fraction": 0.1,
            "logprob_abs_diff_max": 0.0,
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
