"""Unit tests for VideoScore2 score parsing and soft expected-value scoring.

These exercise the pure helpers without loading the 7B judge, so the risky
parts — the upstream regex and the digit-token alignment used for continuous
scores — are covered deterministically.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vrl.rewards.models.videoscore2 import (
    _merge_soft_with_hard,
    _normalize_scores,
    _parse_integer_scores,
    _soft_scores_from_generation,
)


def test_parse_integer_scores_reads_three_axes() -> None:
    text = (
        "The character stays consistent and the motion is smooth.\n"
        "visual quality: 4; text-to-video alignment: 3, "
        "physical/common-sense consistency: 5"
    )
    assert _parse_integer_scores(text) == (4, 3, 5)


def test_parse_integer_scores_rejects_unparseable() -> None:
    assert _parse_integer_scores("no scores here") is None


def test_parse_integer_scores_rejects_out_of_range() -> None:
    text = (
        "visual quality: 7; text-to-video alignment: 3, "
        "physical/common-sense consistency: 5"
    )
    assert _parse_integer_scores(text) is None


def test_normalize_scores_public_keys_and_mean() -> None:
    scores = _normalize_scores(4.0, 2.0, 3.0)
    assert set(scores) == {
        "visual_quality",
        "text_alignment",
        "physical_common_sense",
        "overall",
    }
    assert scores["overall"] == (4.0 + 2.0 + 3.0) / 3.0


class _FakeTokenizer:
    """Minimal token vocabulary for the soft-score alignment test.

    digits 1..5 -> ids 1..5; marker words -> dedicated ids; everything else is a
    filler id. Marker tokenization is space-insensitive here, matching how the
    real helper searches both " quality" and "quality".
    """

    _vocab: ClassVar[dict[str, int]] = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "quality": 10, "alignment": 20, "consistency": 30,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        key = text.strip()
        return [self._vocab[key]] if key in self._vocab else [99]


def _logits_for(realized_id: int, vocab: int = 100) -> torch.Tensor:
    # vocab spans all fake ids (markers/filler up to 99); only digit-slot logits
    # are read by the soft path, but every step needs an indexable row.
    logits = torch.zeros(vocab)
    logits[realized_id] = 10.0
    return logits


def test_soft_scores_align_to_digit_after_each_marker() -> None:
    tokenizer = _FakeTokenizer()
    # "... quality : 4 ; ... alignment : 2 , ... consistency : 5"
    generated_ids = [10, 99, 4, 99, 20, 99, 2, 99, 30, 99, 5]
    step_logits = [_logits_for(tid) for tid in generated_ids]
    digit_token_ids = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    soft = _soft_scores_from_generation(
        generated_ids,
        step_logits,
        digit_token_ids,
        tokenizer=tokenizer,
    )

    assert soft["visual_quality"] == _approx(4.0)
    assert soft["text_alignment"] == _approx(2.0)
    assert soft["physical_common_sense"] == _approx(5.0)


def test_soft_scores_anchor_last_marker_not_cot_mention() -> None:
    """Reproduces the live misalignment: CoT mentions the marker digit-free,
    then the numbered answer list "(1) visual quality: 3 ..." follows. The
    list numeral "1" is the first digit after the CoT mention and must NOT be
    read as the score (it produced soft=1.0 vs hard=3 on real weights)."""

    tokenizer = _FakeTokenizer()
    # CoT "quality ..." (no digits), then "( 1 ) quality : 3 ( 2 ) alignment : 4
    # ( 3 ) consistency : 4"
    generated_ids = [10, 99, 99, 99, 1, 10, 3, 99, 2, 20, 4, 99, 3, 30, 4]
    step_logits = [_logits_for(tid) for tid in generated_ids]
    digit_token_ids = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    soft = _soft_scores_from_generation(
        generated_ids,
        step_logits,
        digit_token_ids,
        tokenizer=tokenizer,
    )

    assert soft["visual_quality"] == _approx(3.0)
    assert soft["text_alignment"] == _approx(4.0)
    assert soft["physical_common_sense"] == _approx(4.0)


def test_soft_scores_return_none_when_marker_missing() -> None:
    tokenizer = _FakeTokenizer()
    # No "alignment"/"consistency" markers -> those axes cannot be located.
    generated_ids = [10, 99, 4]
    step_logits = [_logits_for(tid) for tid in generated_ids]
    digit_token_ids = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    soft = _soft_scores_from_generation(
        generated_ids,
        step_logits,
        digit_token_ids,
        tokenizer=tokenizer,
    )

    assert soft["visual_quality"] == _approx(4.0)
    assert soft["text_alignment"] is None
    assert soft["physical_common_sense"] is None


def test_merge_rejects_soft_far_from_hard_keeps_near() -> None:
    """Soft may only refine its hard integer: a soft value more than the
    tolerance away from the emitted digit is a misanchored slot (the live
    failure read a list numeral "1" where the score line said 3) and must
    fall back to hard; a nearby soft value is genuine spread and is kept."""

    soft = {
        "visual_quality": 1.0,  # misanchor vs hard 3 -> rejected
        "text_alignment": 3.6,  # within tolerance of hard 4 -> kept
        "physical_common_sense": None,  # unlocated -> hard fallback
    }
    merged = _merge_soft_with_hard(soft, (3, 4, 5))
    assert merged == (3.0, 3.6, 5.0)


def _approx(value: float, tol: float = 1e-2) -> object:
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - value) <= tol

        def __repr__(self) -> str:
            return f"~{value}"

    return _Approx()
