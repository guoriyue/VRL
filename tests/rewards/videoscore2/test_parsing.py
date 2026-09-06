"""Unit tests for VideoScore2 score parsing and soft expected-value scoring.

These exercise the pure helpers without loading the 7B judge, so the risky
parts — the upstream regex and the digit-token alignment used for continuous
scores — are covered deterministically. The soft path runs on a tiny real
byte-level BPE (``tests/rewards/videoscore2/fixtures.py``) whose marker
tokenizations have the same shape as the real Qwen2 tokenizer's; the
``optional`` lane checks that shape against the real tokenizer.
"""

from __future__ import annotations

import re

import pytest
import torch

from tests.rewards.videoscore2.fixtures import build_tiny_marker_tokenizer
from vrl.rewards.assets.video_judge_prompts import (
    VIDEOSCORE2_SYSTEM_PROMPT,
    VIDEOSCORE2_USER_TEMPLATE,
)
from vrl.rewards.models.videoscore2 import (
    _DIMENSION_MARKERS,
    _marker_token_ids,
    _merge_soft_with_hard,
    _normalize_scores,
    _parse_integer_scores,
    _resolve_digit_token_ids,
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

    declared = VIDEOSCORE2_USER_TEMPLATE.format(prompt="a cat").split(_FORMAT_HEADER, 1)[1]
    digits = iter("345")
    filled = re.sub(r"<[^>]+>", lambda _: next(digits), declared)

    assert _parse_integer_scores(filled) == (3, 4, 5)


def test_system_and_user_prompts_declare_one_output_format() -> None:
    """Both prompts spell the format out; they must spell the same one."""

    declared = VIDEOSCORE2_USER_TEMPLATE.format(prompt="a cat").split(_FORMAT_HEADER, 1)[1]

    assert declared in VIDEOSCORE2_SYSTEM_PROMPT


def test_soft_score_markers_appear_in_the_prompt() -> None:
    """Soft scoring locates each digit by these phrases in the generated text.

    The model only emits them because the prompt does; a marker absent from the
    prompt would silently misanchor the expected-value path.
    """

    for key, marker in _DIMENSION_MARKERS:
        assert marker in VIDEOSCORE2_SYSTEM_PROMPT, key


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


@pytest.fixture(scope="module")
def tokenizer():
    return build_tiny_marker_tokenizer()


def _soft_scores(tokenizer, text: str) -> dict[str, float | None]:
    """Tokenize the judge's output with the real BPE and read the soft scores back.

    The step logits realize exactly the generated token at every position, so the
    expected value at a located digit slot is that digit.
    """

    generated_ids = tokenizer.encode(text, add_special_tokens=False)
    vocab = len(tokenizer)
    step_logits = []
    for token_id in generated_ids:
        logits = torch.zeros(vocab)
        logits[token_id] = 10.0
        step_logits.append(logits)
    return _soft_scores_from_generation(
        generated_ids,
        step_logits,
        _resolve_digit_token_ids(tokenizer),
        tokenizer=tokenizer,
    )


def test_marker_search_covers_both_spacings_and_multi_token_markers(tokenizer) -> None:
    """Both " marker" and "marker" are searched, and a marker may span tokens.

    ``consistency`` without a leading space tokenizes to two pieces here, as it
    does in the real Qwen2 vocabulary, so the multi-token needle path is live.
    """

    variants = {marker: _marker_token_ids(tokenizer, marker) for _, marker in _DIMENSION_MARKERS}

    assert all(len(ids) == 2 for ids in variants.values())
    assert any(len(needle) > 1 for needle in variants["consistency"])
    assert set(_resolve_digit_token_ids(tokenizer)) == {1, 2, 3, 4, 5}


def test_soft_scores_align_to_digit_after_each_marker(tokenizer) -> None:
    soft = _soft_scores(tokenizer, "quality: 4; alignment: 2, consistency: 5")

    assert soft["visual_quality"] == pytest.approx(4.0, abs=1e-2)
    assert soft["text_alignment"] == pytest.approx(2.0, abs=1e-2)
    assert soft["physical_common_sense"] == pytest.approx(5.0, abs=1e-2)


def test_soft_scores_anchor_last_marker_not_cot_mention(tokenizer) -> None:
    """Reproduces the live misalignment: CoT mentions the marker digit-free,
    then the numbered answer list "(1) visual quality: 3 ..." follows. The
    list numeral "1" is the first digit after the CoT mention and must NOT be
    read as the score (it produced soft=1.0 vs hard=3 on real weights)."""

    soft = _soft_scores(
        tokenizer,
        "quality looks fine overall (1) quality: 3 (2) alignment: 4 (3) consistency: 4",
    )

    assert soft["visual_quality"] == pytest.approx(3.0, abs=1e-2)
    assert soft["text_alignment"] == pytest.approx(4.0, abs=1e-2)
    assert soft["physical_common_sense"] == pytest.approx(4.0, abs=1e-2)


def test_unprefixed_multi_token_marker_still_anchors(tokenizer) -> None:
    """ "(3)consistency" has no leading space: only the two-token needle can match."""

    soft = _soft_scores(tokenizer, "(1) quality: 3, (2) alignment: 4, (3)consistency: 5")

    assert soft["physical_common_sense"] == pytest.approx(5.0, abs=1e-2)


def test_soft_scores_return_none_when_marker_missing(tokenizer) -> None:
    # No "alignment"/"consistency" markers -> those axes cannot be located.
    soft = _soft_scores(tokenizer, "quality: 4")

    assert soft["visual_quality"] == pytest.approx(4.0, abs=1e-2)
    assert soft["text_alignment"] is None
    assert soft["physical_common_sense"] is None


@pytest.mark.optional
def test_real_videoscore2_tokenizer_has_the_marker_shape_the_fixture_assumes() -> None:
    """The tiny BPE encodes our belief about Qwen2's vocabulary; only the real
    tokenizer can tell us that belief has expired. Digits must be single tokens
    or the soft path silently disables itself."""

    from transformers import AutoTokenizer

    try:
        real = AutoTokenizer.from_pretrained("TIGER-Lab/VideoScore2", local_files_only=True)
    except Exception as exc:  # pragma: no cover - offline machines without the cache
        pytest.skip(f"VideoScore2 tokenizer is not cached: {exc}")

    variants = {marker: _marker_token_ids(real, marker) for _, marker in _DIMENSION_MARKERS}
    assert all(len(ids) == 2 for ids in variants.values())
    assert [len(needle) for needle in variants["quality"]] == [1, 1]
    assert [len(needle) for needle in variants["alignment"]] == [1, 1]
    assert [len(needle) for needle in variants["consistency"]] == [2, 1]
    assert set(_resolve_digit_token_ids(real)) == {1, 2, 3, 4, 5}


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
