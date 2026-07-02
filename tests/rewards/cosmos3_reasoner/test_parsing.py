"""Unit tests for the Cosmos3 reasoner judge: score parsing + registration.

These exercise the pure helpers and the registry entry WITHOUT loading the 16B
Qwen3-VL reasoner, so the risky parts — the fixed-format regex and the public
score contract — are covered deterministically on CPU.
"""

from __future__ import annotations

from vrl.rewards.models.cosmos3_reasoner import (
    _normalize_scores,
    _parse_integer_scores,
)


def test_parse_integer_scores_reads_four_axes() -> None:
    text = (
        "The gripper closes on the block and places it in the bin; motion is stable.\n"
        "task success: 4; contact realism: 3, temporal consistency: 5, "
        "physical plausibility: 2"
    )
    assert _parse_integer_scores(text) == (4, 3, 5, 2)


def test_parse_integer_scores_rejects_unparseable() -> None:
    assert _parse_integer_scores("no scores here") is None


def test_parse_integer_scores_rejects_partial() -> None:
    # Only three of the four axes present -> no full match.
    text = "task success: 4; contact realism: 3, temporal consistency: 5"
    assert _parse_integer_scores(text) is None


def test_parse_integer_scores_rejects_out_of_range() -> None:
    text = (
        "task success: 6; contact realism: 3, temporal consistency: 5, "
        "physical plausibility: 2"
    )
    assert _parse_integer_scores(text) is None


def test_normalize_scores_public_keys_and_mean() -> None:
    scores = _normalize_scores(4.0, 2.0, 3.0, 5.0)
    assert set(scores) == {
        "task_success",
        "contact_realism",
        "temporal_consistency",
        "physical_plausibility",
        "overall",
    }
    assert scores["overall"] == (4.0 + 2.0 + 3.0 + 5.0) / 4.0


def test_registered_reward() -> None:
    """The reward must be registered under its config key."""
    from vrl.rewards.functions.registry import _register_builtins, get_reward

    _register_builtins()
    reward_cls = get_reward("cosmos3_reasoner")
    assert reward_cls is not None
