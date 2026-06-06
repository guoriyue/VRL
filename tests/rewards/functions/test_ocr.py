"""Tests for vrl.rewards.functions.ocr (OCRReward)."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.ocr import (
    OCRReward,
    _normalize_text,
    _normalized_edit_distance,
)


def test_text_normalization_core_cases() -> None:
    assert _normalize_text("  Hello, World!  ") == "hello world"
    assert _normalize_text("a   b\tc") == "a b c"
    assert _normalize_text("EXIT-42.") == "exit42"
    assert _normalize_text("") == ""


def test_normalized_edit_distance_core_cases() -> None:
    assert _normalized_edit_distance("hello", "hello") == pytest.approx(0.0)
    assert _normalized_edit_distance("", "") == pytest.approx(0.0)
    assert _normalized_edit_distance("abc", "xyz") > 0.5

    partial = _normalized_edit_distance("hello", "helo")
    assert 0.0 < partial < 1.0

    d1 = _normalized_edit_distance("abc", "abcd")
    d2 = _normalized_edit_distance("abcd", "abc")
    assert d1 == pytest.approx(d2, abs=0.01)


class _FakePaddleOCR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def ocr(self, frame, cls=False):
        del frame, cls
        text = self.texts.pop(0)
        return [[(None, (text, 1.0))]]


def _has_rapidocr() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_rapidocr = pytest.mark.skipif(
    not _has_rapidocr(), reason="rapidocr_onnxruntime not installed"
)


def _make_ocr_rollout(target_text: str, video_tensor=None):
    """Build a minimal RewardRollout with target_text metadata and a video tensor."""
    import torch

    from vrl.rewards.types import RewardRollout, RewardTrajectory

    if video_tensor is None:
        # Black frames — no OCR text expected
        video_tensor = torch.zeros(3, 8, 64, 64)

    traj = RewardTrajectory(prompt="test", output=video_tensor)
    return RewardRollout(request=None, trajectory=traj, metadata={"target_text": target_text})


@_skip_no_rapidocr
@pytest.mark.asyncio
async def test_ocr_reward_rapidocr_core_scoring_behaviors() -> None:
    reward = OCRReward(device="cpu")

    assert await reward.score(_make_ocr_rollout("")) == pytest.approx(0.0)
    assert await reward.score(_make_ocr_rollout("HELLO")) <= 0.5

    scores = await reward.score_batch([_make_ocr_rollout("A"), _make_ocr_rollout("B")])
    assert len(scores) == 2


@pytest.mark.asyncio
async def test_image_ocr_substring_match_gets_full_credit() -> None:
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Cafe Free WiFi Open"])
    rollout = _make_ocr_rollout(
        "Free WiFi",
        video_tensor=torch.zeros(3, 64, 64),
    )

    score = await reward.score(rollout)

    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_video_ocr_keeps_flow_grpo_video_edit_distance_behavior() -> None:
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Free WiFiX", "Free WiFiX"])
    rollout = _make_ocr_rollout(
        "Free WiFi",
        video_tensor=torch.zeros(3, 8, 64, 64),
    )

    score = await reward.score(rollout)

    assert 0.0 < score < 1.0
