"""OCR text-matching reward (flow_grpo-compatible) over the selected transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class OCRReward(ModelRewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation; ``score_key``
    selects that edit similarity (``ocr``) or the exact whole-text match
    fraction (``ocr_match``).

    Transports: in-process (model built eagerly, media in memory, image or
    video tensors alike); ``kind=ray`` (a placement-owned PaddleOCR actor);
    ``kind=http`` (an operator-run service). ``dump`` writes the
    best-scoring frame and the recognized lines.
    """

    model_factory = "vrl.rewards.models.ocr:OCRRewardModel"
    name = "ocr"
    default_score_key = "ocr"
    eager_model = True
    score_keys = ("ocr", "ocr_match")
    dumps_samples = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """PaddleOCR runs CPU-only; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["OCRReward"]
