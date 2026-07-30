"""Unit tests for VideoScore2 score parsing and soft expected-value scoring.

These exercise the pure helpers without loading the 7B judge, so the risky
parts — the upstream regex and the digit-token alignment used for continuous
scores — are covered deterministically.
"""

from __future__ import annotations

import re
from typing import ClassVar

import pytest
import torch

from vrl.rewards.models.videoscore2 import (
    _DIMENSION_MARKERS,
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _merge_soft_with_hard,
    _normalize_scores,
    _parse_integer_scores,
    _soft_scores_from_generation,
)

_FORMAT_HEADER = "Please output in this format:\n"


def test_parser_reads_the_exact_format_the_prompt_asks_for() -> None:
    """The prompt's own format line is the only spec the judge ever sees.

    Every other test here feeds the parser a hand-written sample, so the prompt,
    the regex, and those samples are three copies of one contract that never
    check each other. Filling the prompt's ``<...>`` slots and parsing the result
    makes the prompt the source of truth: reword an axis and this fails until the
    regex follows.
    """

    declared = _USER_TEMPLATE.format(prompt="a cat").split(_FORMAT_HEADER, 1)[1]
    digits = iter("345")
    filled = re.sub(r"<[^>]+>", lambda _: next(digits), declared)

    assert _parse_integer_scores(filled) == (3, 4, 5)


def test_system_and_user_prompts_declare_one_output_format() -> None:
    """Both prompts spell the format out; they must spell the same one."""

    declared = _USER_TEMPLATE.format(prompt="a cat").split(_FORMAT_HEADER, 1)[1]

    assert declared in _SYSTEM_PROMPT


def test_soft_score_markers_appear_in_the_prompt() -> None:
    """Soft scoring locates each digit by these phrases in the generated text.

    The model only emits them because the prompt does; a marker absent from the
    prompt would silently misanchor the expected-value path.
    """

    for key, marker in _DIMENSION_MARKERS:
        assert marker in _SYSTEM_PROMPT, key


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
    text = "visual quality: 7; text-to-video alignment: 3, physical/common-sense consistency: 5"
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
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "quality": 10,
        "alignment": 20,
        "consistency": 30,
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

    assert soft["visual_quality"] == pytest.approx(4.0, abs=1e-2)
    assert soft["text_alignment"] == pytest.approx(2.0, abs=1e-2)
    assert soft["physical_common_sense"] == pytest.approx(5.0, abs=1e-2)


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

    assert soft["visual_quality"] == pytest.approx(3.0, abs=1e-2)
    assert soft["text_alignment"] == pytest.approx(4.0, abs=1e-2)
    assert soft["physical_common_sense"] == pytest.approx(4.0, abs=1e-2)


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

    assert soft["visual_quality"] == pytest.approx(4.0, abs=1e-2)
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
