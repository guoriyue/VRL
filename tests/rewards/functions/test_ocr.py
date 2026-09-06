"""Tests for vrl.rewards.functions.ocr (OCRReward).

The engine doubles below hand-write PaddleOCR's return layouts: ``_PaddleOCR2x``
is the 2.x ``ocr()`` nesting ``[[(box, (text, score))]]`` and ``_PaddleOCR3x``
is the 3.x ``predict()`` column dict ``{"rec_texts", "rec_scores"}`` that the
``[ocr]`` extra (``paddleocr>=3.5.0``) actually ships. Both shapes are literals
only the opt-in real-engine test (``WM_RUN_REAL_MODEL_TESTS=1``) validates; the
scoring tests are parametrized over both so a protocol drift is at least
visible on both branches of ``_run_paddle_ocr``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests import ci_envs
from vrl.rewards.functions.ocr import OCRReward
from vrl.rewards.models.ocr import (
    _build_paddle_ocr,
    _extract_ocr_lines,
    _run_paddle_ocr,
)
from vrl.rewards.ocr_text import OcrEngineProfile, normalize_ocr_text

_TRUETYPE_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


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


class _PaddleOCR2x:
    """PaddleOCR 2.x: ``.ocr(frame, cls=False) -> [[(box, (text, score)), ...]]``."""

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


class _PaddleOCR3x:
    """PaddleOCR 3.x: ``.predict(frame) -> [{"rec_texts": [...], "rec_scores": [...]}]``."""

    def __init__(
        self,
        texts: list[str | list[str | tuple[str, float]]],
    ) -> None:
        self.texts = list(texts)

    def predict(self, frame):
        del frame
        value = self.texts.pop(0)
        texts = [value] if isinstance(value, str) else value
        items = [item if isinstance(item, tuple) else (item, 1.0) for item in texts]
        return [
            {
                "rec_texts": [text for text, _ in items],
                "rec_scores": [score for _, score in items],
            },
        ]


_ENGINE_PROTOCOLS = pytest.mark.parametrize(
    "engine_cls",
    [_PaddleOCR2x, _PaddleOCR3x],
    ids=["paddleocr-2x", "paddleocr-3x"],
)


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


def _rendered_text_frame(text: str):
    """White ``text`` on black at a size PaddleOCR reads; skips without a TrueType font.

    PIL's default bitmap font renders "HELLO" at 33x8 px, far below what the
    detector resolves, so the render is pinned to an explicit TrueType face.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((path for path in _TRUETYPE_FONTS if os.path.exists(path)), None)
    if font_path is None:
        pytest.skip("no TrueType font available to render OCR text")
    image = Image.new("RGB", (256, 96), "black")
    ImageDraw.Draw(image).text(
        (16, 16), text, fill="white", font=ImageFont.truetype(font_path, 48)
    )
    return np.asarray(image)


@pytest.mark.skipif(
    not ci_envs.WM_RUN_REAL_MODEL_TESTS,
    reason="set WM_RUN_REAL_MODEL_TESTS=1 to run the real PaddleOCR engine (downloads weights)",
)
@pytest.mark.asyncio
async def test_real_paddleocr_reads_rendered_text_and_ranks_the_matching_target() -> None:
    """The real engine's return layout is what the doubles above hand-write.

    Asserts SHAPE (the adapter reads the rendered word out of the engine output)
    and a differential (matching target beats a non-matching one), never a full
    score: the engine's exact recognition is not this repo's contract.
    """
    import torch

    pytest.importorskip("paddleocr")
    frame = _rendered_text_frame("HELLO")

    engine = _build_paddle_ocr()
    lines = _extract_ocr_lines(_run_paddle_ocr(engine, frame))
    assert lines
    assert "hello" in normalize_ocr_text("".join(line.text for line in lines))

    reward = OCRReward(device="cpu")
    reward._engine = engine
    image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    matching = await reward.score(_make_ocr_sample("HELLO", video_tensor=image))
    mismatching = await reward.score(_make_ocr_sample("ZQXJ", video_tensor=image))
    assert matching > mismatching


@_ENGINE_PROTOCOLS
@pytest.mark.asyncio
async def test_image_ocr_substring_match_gets_full_credit(engine_cls) -> None:
    """The flow_grpo substring shortcut is image-only: a single image whose OCR text
    contains the target scores 1.0 regardless of the surrounding text."""
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = engine_cls(["Cafe Free WiFi Open"])
    sample = _make_ocr_sample(
        "Free WiFi",
        video_tensor=torch.zeros(3, 64, 64),
    )

    score = await reward.score(sample)

    assert score == pytest.approx(1.0)


@_ENGINE_PROTOCOLS
@pytest.mark.asyncio
async def test_video_ocr_never_gets_the_image_only_substring_credit(engine_cls) -> None:
    """A VIDEO whose every frame literally contains the target scores by edit
    distance, not by substring: distance("cafefreewifiopen", "freewifi") == 8 ==
    len(target), so the frame reward is 0 and the mean over positive frames is 0.
    The same text scores 1.0 as an image (above)."""
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = engine_cls(["Cafe Free WiFi Open", "Cafe Free WiFi Open"])
    sample = _make_ocr_sample("Free WiFi", video_tensor=torch.zeros(3, 8, 64, 64))

    assert await reward.score(sample) == pytest.approx(0.0)


def test_column_lines_do_not_silently_truncate_to_the_shorter_column() -> None:
    """A 3.x result whose columns disagree in length fails instead of zip-truncating."""
    with pytest.raises(ValueError, match="lengths differ"):
        _extract_ocr_lines([{"rec_texts": ["ab", "cd"], "rec_scores": [1.0]}])


@pytest.mark.asyncio
async def test_image_ocr_can_require_the_complete_recognized_string() -> None:
    """Exact-text curricula must not reward a target buried in junk text."""
    import torch

    reward = OCRReward(device="cpu", substring_full_credit=False)
    reward._engine = _PaddleOCR2x(["D007"])
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
    legacy._engine = _PaddleOCR2x([["Adopt Me", "OPEN DAILY", "123"]])
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
    reward._engine = _PaddleOCR2x([["Adopt Me", "OPEN DAILY", "123"]])
    assert await reward.score(sample) == pytest.approx(1.0)

    reward = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
    )
    reward._engine = _PaddleOCR2x([["D007", "incidental"]])
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
    exclusive._engine = _PaddleOCR2x([["Adopt Me", "OPEN DAILY"]])
    assert await exclusive.score(sample) == pytest.approx(0.0)

    exclusive = OCRReward(
        device="cpu",
        text_selection="best_complete_line",
        substring_full_credit=False,
        exclusive_alphanumeric_lines=True,
        extra_line_min_confidence=0.5,
    )
    exclusive._engine = _PaddleOCR2x([["Adopt Me", "-"]])
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
    reward._engine = _PaddleOCR2x([["WHIZ", "BANG"]])
    assert await reward.score(sample) == pytest.approx(1.0)

    reward = OCRReward(
        device="cpu",
        text_selection="best_contiguous_lines",
        substring_full_credit=False,
    )
    reward._engine = _PaddleOCR2x([["WHIZ", "OPEN", "BANG"]])
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
    reward._engine = _PaddleOCR2x([lines])
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
    reward._engine = _PaddleOCR2x([["WHIZ", "BANG", "OPEN"]])
    sample = _make_ocr_sample("WHIZBANG", video_tensor=torch.zeros(3, 64, 64))

    assert await reward.score(sample) == pytest.approx(1.0)


def test_ocr_policy_fails_closed() -> None:
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


@pytest.mark.asyncio
async def test_image_ocr_debug_dump_includes_zero_score_sample(tmp_path: Path) -> None:
    """A zero reward must retain the image and OCR output for audit."""
    import torch

    debug_dir = tmp_path / "ocr_debug"
    reward = OCRReward(device="cpu", debug_dir=str(debug_dir))
    reward._engine = _PaddleOCR2x(["JUNK"])
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
    reward._engine = _PaddleOCR2x(["JUNK"])

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
    reward._engine = _PaddleOCR2x(["ABC"])

    with caplog.at_level(logging.WARNING, logger="vrl.rewards.models.ocr"):
        score = await reward.score(
            _make_ocr_sample("ABC", video_tensor=torch.zeros(3, 64, 64)),
        )

    assert score == pytest.approx(1.0)
    assert "Failed to write OCR debug sample 000000" in caplog.text
    assert "debug disk unavailable" in caplog.text


@_ENGINE_PROTOCOLS
@pytest.mark.asyncio
async def test_video_ocr_keeps_flow_grpo_video_edit_distance_behavior(engine_cls) -> None:
    """One inserted character over an 8-character target: reward 1 - 1/8 = 0.875 on
    each of the two sampled frames (``frame_interval=4`` over 8 frames), mean 0.875.
    A loose ``0 < score < 1`` would also accept dividing by ``len(text)`` (0.888)."""
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = engine_cls(["Free WiFiX", "Free WiFiX"])
    sample = _make_ocr_sample(
        "Free WiFi",
        video_tensor=torch.zeros(3, 8, 64, 64),
    )

    score = await reward.score(sample)

    assert score == pytest.approx(0.875)
