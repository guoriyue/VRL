"""Tests for vrl.rewards.functions.ocr (OCRReward)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.rewards.functions.ocr import OCRReward
from vrl.rewards.models.ocr import _extract_ocr_lines
from vrl.rewards.ocr_text import OcrEngineProfile, normalize_ocr_text


def test_ocr_text_normalization_matches_flow_grpo_contract() -> None:
    """Pin the shared OCR contract: lowercase and strip ASCII spaces while
    deliberately PRESERVING punctuation.

    The substring/Levenshtein scoring in ``OCRRewardModel`` depends on this exact
    behavior. Dataset selection imports the same leaf function rather than
    maintaining a second normalizer.
    """

    assert normalize_ocr_text("Hello World") == "helloworld"
    assert normalize_ocr_text("  EXIT 42  ") == "exit42"
    # Punctuation is kept (a regex word-normalizer would have dropped it).
    assert normalize_ocr_text("Cafe, Free!") == "cafe,free!"
    assert normalize_ocr_text("") == ""


# Real end-to-end scoring (substring full-credit + video Levenshtein) is exercised
# by the live/fake-engine tests below, so no shadow edit-distance reimplementation
# is needed here.


class _FakePaddleOCR:
    def __init__(
        self,
        texts: list[str | list[str | tuple[str, float]]],
    ) -> None:
        self.texts = list(texts)

    def ocr(self, frame, cls=False):
        del frame, cls
        value = self.texts.pop(0)
        texts = [value] if isinstance(value, str) else value
        return [[(None, item if isinstance(item, tuple) else (item, 1.0)) for item in texts]]


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
async def test_image_ocr_can_require_the_complete_recognized_string() -> None:
    """Exact-text curricula must not reward a target buried in junk text."""
    import torch

    reward = OCRReward(device="cpu", substring_full_credit=False)
    reward._engine = _FakePaddleOCR(["D007"])
    sample = _make_ocr_sample(
        "007",
        video_tensor=torch.zeros(3, 64, 64),
    )

    score = await reward.score(sample)

    assert score == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_image_ocr_can_select_the_best_complete_line_without_substring_credit() -> None:
    """Incidental scene text must not erase a complete target line."""
    import torch

    legacy = OCRReward(device="cpu", substring_full_credit=False)
    legacy._engine = _FakePaddleOCR([["Adopt Me", "OPEN DAILY", "123"]])
    sample = _make_ocr_sample(
        "Adopt Me",
        video_tensor=torch.zeros(3, 64, 64),
    )
    assert await legacy.score(sample) < 1.0

    reward = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
    )
    reward._engine = _FakePaddleOCR([["Adopt Me", "OPEN DAILY", "123"]])
    assert await reward.score(sample) == pytest.approx(1.0)

    reward = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
    )
    reward._engine = _FakePaddleOCR([["D007", "incidental"]])
    sample = _make_ocr_sample("007", video_tensor=torch.zeros(3, 64, 64))

    assert await reward.score(sample) == pytest.approx(2 / 3)

    exclusive = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
        exclusive_alphanumeric_lines=True,
        extra_line_min_confidence=0.5,
    )
    sample = _make_ocr_sample("Adopt Me", video_tensor=torch.zeros(3, 64, 64))
    exclusive._engine = _FakePaddleOCR([["Adopt Me", "OPEN DAILY"]])
    assert await exclusive.score(sample) == pytest.approx(0.0)

    exclusive = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
        exclusive_alphanumeric_lines=True,
        extra_line_min_confidence=0.5,
    )
    exclusive._engine = _FakePaddleOCR([["Adopt Me", "-"]])
    assert await exclusive.score(sample) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_image_ocr_can_join_only_adjacent_detected_lines() -> None:
    """A wrapped target may span lines, but unrelated intervening text stays significant."""
    import torch

    sample = _make_ocr_sample(
        "WHIZBANG",
        video_tensor=torch.zeros(3, 64, 64),
    )
    reward = OCRReward(
        device="cpu",
        text_selection="best_contiguous_lines",
        substring_full_credit=False,
    )
    reward._engine = _FakePaddleOCR([["WHIZ", "BANG"]])
    assert await reward.score(sample) == pytest.approx(1.0)

    reward = OCRReward(
        device="cpu",
        text_selection="best_contiguous_lines",
        substring_full_credit=False,
    )
    reward._engine = _FakePaddleOCR([["WHIZ", "OPEN", "BANG"]])
    assert await reward.score(sample) < 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["nalitabari", "OPEN DAILY"], 1.0),
        (["nalitabari", "nalitbari"], 0.5),
        (["nalitabari", "nalitbari", "alitarari"], 1 / 3),
        (["nalitabari", ("nalitbari", 0.49)], 1.0),
    ],
)
async def test_image_ocr_penalizes_only_confident_near_target_duplicates(
    lines: list[str | tuple[str, float]],
    expected: float,
) -> None:
    import torch

    reward = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
        near_duplicate_min_similarity=0.4,
        extra_line_min_confidence=0.5,
    )
    reward._engine = _FakePaddleOCR([lines])
    sample = _make_ocr_sample("nalitabari", video_tensor=torch.zeros(3, 64, 64))

    assert await reward.score(sample) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_ocr_duplicate_policy_excludes_every_line_in_selected_span() -> None:
    import torch

    reward = OCRReward(
        device="cpu",
        text_selection="best_contiguous_lines",
        substring_full_credit=False,
        near_duplicate_min_similarity=0.4,
    )
    reward._engine = _FakePaddleOCR([["WHIZ", "BANG", "OPEN"]])
    sample = _make_ocr_sample("WHIZBANG", video_tensor=torch.zeros(3, 64, 64))

    assert await reward.score(sample) == pytest.approx(1.0)


def test_ocr_policy_and_paddle3_columns_fail_closed() -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        OCRReward(substring_full_credit="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires substring_full_credit=false"):
        OCRReward(text_selection="best_complete_line")
    with pytest.raises(ValueError, match="requires a line-preserving text_selection"):
        OCRReward(exclusive_alphanumeric_lines=True)
    with pytest.raises(ValueError, match="requires a line-preserving text_selection"):
        OCRReward(near_duplicate_min_similarity=0.4)
    with pytest.raises(ValueError, match="null or finite"):
        OCRReward(
            text_selection="best_complete_line",
            substring_full_credit=False,
            near_duplicate_min_similarity=1.1,
        )
    with pytest.raises(ValueError, match="must be one of"):
        OCRReward(engine_profile="latest")
    with pytest.raises(ValueError, match="lengths differ"):
        _extract_ocr_lines({"rec_texts": ["A", "B"], "rec_scores": [1.0]})


@pytest.mark.asyncio
async def test_image_ocr_debug_dump_includes_zero_score_sample(tmp_path: Path) -> None:
    """A zero reward must retain the image and OCR output for audit."""
    import torch

    debug_dir = tmp_path / "ocr_debug"
    reward = OCRReward(device="cpu", debug_dir=str(debug_dir))
    reward._engine = _FakePaddleOCR(["JUNK"])
    sample = _make_ocr_sample(
        "ABC",
        video_tensor=torch.zeros(3, 64, 64),
        sample_id="request:prompt:0:sample:7",
    )

    score = await reward.score(sample)

    assert score == pytest.approx(0.0)
    image_paths = list(debug_dir.glob("*_score0.000.png"))
    metadata_paths = list(debug_dir.glob("*_score0.000.json"))
    assert len(image_paths) == 1
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["sample_id"] == "request:prompt:0:sample:7"
    assert metadata["target_text"] == "ABC"
    assert metadata["engine_profile"] == OcrEngineProfile.FLOW_GRPO_COMPAT.value
    assert metadata["schema"] == "vrl.ocr-debug/v5"
    assert metadata["recognized_lines"] == [{"confidence": 1.0, "text": "JUNK"}]
    assert metadata["selected_line_indices"] is None
    assert metadata["near_duplicate_line_indices"] == []
    assert metadata["aggregate_score"] == 0.0
    assert "sample-request_prompt_0_sample" in image_paths[0].name


@pytest.mark.asyncio
async def test_image_ocr_debug_dump_continues_existing_index(tmp_path: Path) -> None:
    """A restarted worker must not overwrite rollout evidence from a prior run."""
    import torch

    debug_dir = tmp_path / "ocr_debug"
    debug_dir.mkdir()
    (debug_dir / "000000_ABC_score0.000.png").touch()
    (debug_dir / "000000_ABC_score0.000.txt").touch()
    reward = OCRReward(device="cpu", debug_dir=str(debug_dir))
    reward._engine = _FakePaddleOCR(["JUNK"])

    await reward.score(
        _make_ocr_sample("ABC", video_tensor=torch.zeros(3, 64, 64)),
    )

    assert len(list(debug_dir.glob("000001_*_score0.000.png"))) == 1
    assert len(list(debug_dir.glob("000001_*_score0.000.json"))) == 1


@pytest.mark.asyncio
async def test_image_ocr_debug_dump_warns_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Debug I/O failures stay non-fatal but must be visible to operators."""
    import logging

    import torch
    from PIL import Image

    def fail_save(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("debug disk unavailable")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    reward = OCRReward(device="cpu", debug_dir=str(tmp_path / "ocr_debug"))
    reward._engine = _FakePaddleOCR(["ABC"])

    with caplog.at_level(logging.WARNING, logger="vrl.rewards.models.ocr"):
        score = await reward.score(
            _make_ocr_sample("ABC", video_tensor=torch.zeros(3, 64, 64)),
        )

    assert score == pytest.approx(1.0)
    assert "Failed to write OCR debug sample 000000" in caplog.text
    assert "debug disk unavailable" in caplog.text


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
