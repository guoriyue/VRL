"""The reward card turns the gate reports into one verdict and keeps unrun gates visible."""

import json

import pytest

from reward.card import build_card, render_markdown, write_card


def _write(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _reports(tmp_path, *, auc=0.9, not_below=0.0, success=0.4, mixed=0.5, rng=0.0):
    return {
        "repeat": _write(
            tmp_path,
            "repeat",
            {"repeat_count": 2, "sample_count": 4, "axes": {"judge": {"max_score_range": rng}}},
        ),
        "agreement": _write(
            tmp_path,
            "agreement",
            {
                "axis": "judge",
                "gate": 0.85,
                "unscored_sample_ids": [],
                "contrasts": {
                    "done_vs_not_done": {
                        "passed": auc >= 0.85,
                        "auc": auc,
                        "auc_bootstrap_95ci": [auc - 0.05, min(auc + 0.05, 1.0)],
                        "positive_count": 10,
                        "negative_count": 8,
                    }
                },
            },
        ),
        "stress": _write(
            tmp_path,
            "stress",
            {
                "transforms": {
                    "unchanged_source": {
                        "judge": {
                            "increases": round(10 * not_below),
                            "decreases": 10 - round(10 * not_below),
                            "ties": 0,
                            "paired_delta": {"mean": -1.0},
                        }
                    }
                }
            },
        ),
        "spread": _write(
            tmp_path,
            "spread",
            {
                "axis": "judge",
                "success_rate": success,
                "in_band": 0.2 <= success <= 0.6,
                "band": [0.2, 0.6],
                "mixed_prompt_share": mixed,
                "mean_within_prompt_std": 0.2,
                "zero_spread_prompt_share": 0.0,
            },
        ),
    }


def test_card_passes_only_when_every_gate_passes(tmp_path):
    good = build_card(name="judge", kind="learned", axis="judge", **_reports(tmp_path))
    assert good["ready_for_training_key"] is True and good["gates_not_run"] == []
    assert all(gate["passed"] is True for gate in good["gates"].values())
    gamed = build_card(
        name="judge", kind="learned", axis="judge", **_reports(tmp_path, not_below=0.3)
    )
    assert gamed["gates"]["shortcuts"]["passed"] is False
    assert gamed["ready_for_training_key"] is False
    flat = build_card(name="judge", kind="learned", axis="judge", **_reports(tmp_path, mixed=0.1))
    assert flat["gates"]["spread"]["passed"] is False
    blind = build_card(name="judge", kind="learned", axis="judge", **_reports(tmp_path, auc=0.7))
    assert blind["gates"]["agreement"]["passed"] is False


def test_card_keeps_unrun_gates_visible_and_renders(tmp_path):
    reports = _reports(tmp_path)
    card = build_card(
        name="judge",
        kind="verifiable",
        axis="judge",
        repeat=reports["repeat"],
        agreement=reports["agreement"],
        blind_spots=["cannot see collateral"],
    )
    assert card["ready_for_training_key"] is False
    assert card["gates_not_run"] == ["shortcuts", "spread"]
    text = render_markdown(card)
    assert "not run" in text and "cannot see collateral" in text and "done_vs_not_done" in text
    write_card(card, tmp_path / "card.json")
    assert (tmp_path / "card.md").read_text().startswith("# Reward card: judge")


def test_card_refuses_reports_for_another_axis(tmp_path):
    reports = _reports(tmp_path)
    with pytest.raises(ValueError, match="axis"):
        build_card(name="judge", kind="learned", axis="other", agreement=reports["agreement"])
