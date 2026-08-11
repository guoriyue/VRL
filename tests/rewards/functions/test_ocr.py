"""Tests for vrl.rewards.functions.ocr (OCRReward)."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.ocr import OCRReward


def test_ocr_text_normalization_matches_flow_grpo_contract() -> None:
    """Pin the real ``_normalize_ocr_text`` contract: lowercase and strip ALL
    spaces, while deliberately PRESERVING punctuation.

    The substring/Levenshtein scoring in ``OCRRewardModel`` depends on this exact
    behavior, so the test imports the production function rather than re-deriving
    a (different) normalizer inside the test.
    """
    from vrl.rewards.models.ocr import _normalize_ocr_text

    assert _normalize_ocr_text("Hello World") == "helloworld"
    assert _normalize_ocr_text("  EXIT 42  ") == "exit42"
    # Punctuation is kept (a regex word-normalizer would have dropped it).
    assert _normalize_ocr_text("Cafe, Free!") == "cafe,free!"
    assert _normalize_ocr_text("") == ""


# Real end-to-end scoring (substring full-credit + video Levenshtein) is exercised
# by the live/fake-engine tests below, so no shadow edit-distance reimplementation
# is needed here.


class _FakePaddleOCR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def ocr(self, frame, cls=False):
        del frame, cls
        text = self.texts.pop(0)
        return [[(None, (text, 1.0))]]


def _make_ocr_sample(
    target_text: str,
    video_tensor=None,
    *,
    sample_id: str = "sample-0",
):
    """Build a minimal RewardSample with target_text metadata and a video tensor."""
    import torch

    from vrl.rewards.types import RewardSample

    if video_tensor is None:
        # Black frames — no OCR text expected
        video_tensor = torch.zeros(3, 8, 64, 64)

    return RewardSample(
        prompt="test",
        output=video_tensor,
        sample_id=sample_id,
        metadata={"target_text": target_text},
    )


@pytest.mark.asyncio
async def test_ocr_reward_paddleocr_core_scoring_behaviors() -> None:
    """Checks OCR reward paddleocr core scoring behaviors."""
    # The real engine is built lazily inside score(); gate on the dependency the
    # production runtime actually imports (vrl.rewards.models.ocr::_build_paddle_ocr).
    pytest.importorskip("paddleocr")

    reward = OCRReward(device="cpu")

    assert await reward.score(_make_ocr_sample("")) == pytest.approx(0.0)
    assert await reward.score(_make_ocr_sample("HELLO")) <= 0.5

    output = await reward.score_batch(
        [_make_ocr_sample("A"), _make_ocr_sample("B", sample_id="sample-1")],
    )
    assert len(output.scores) == 2


@pytest.mark.asyncio
async def test_image_ocr_substring_match_gets_full_credit() -> None:
    """Checks image OCR substring match gets full credit."""
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Cafe Free WiFi Open"])
    sample = _make_ocr_sample(
        "Free WiFi",
        video_tensor=torch.zeros(3, 64, 64),
    )

    score = await reward.score(sample)

    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_video_ocr_keeps_flow_grpo_video_edit_distance_behavior() -> None:
    """Checks video OCR keeps flow GRPO video edit distance behavior."""
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Free WiFiX", "Free WiFiX"])
    sample = _make_ocr_sample(
        "Free WiFi",
        video_tensor=torch.zeros(3, 8, 64, 64),
    )

    score = await reward.score(sample)

    assert 0.0 < score < 1.0
