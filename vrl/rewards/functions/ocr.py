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
    ``kind=http`` (an operator-run service). ``debug_dir`` dumps the
    best-scoring frame and the recognized lines.
    """

    model_factory = "vrl.rewards.models.ocr:OCRRewardModel"
    request_prefix = "ocr"
    debug_basename = "ocr"
    default_reward_name = "ocr"
    default_score_key = "ocr"
    default_artifact_format = "tensor"
    default_media_type = "image"
    eager_model = True
    score_keys = ("ocr", "ocr_match")

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """PaddleOCR runs CPU-only; never claim the resource-resolved GPU."""
        return "cpu"

    def __init__(self, *, debug_dir: str | None = None, **kwargs: Any) -> None:
        # ``debug_dir`` here is the model's frame dump directory, not the
        # reward-level debug sidecar the base owns.
        super().__init__(worker_config={"debug_dir": debug_dir}, **kwargs)

    # Test seam onto the eagerly built in-process engine; a remote transport
    # builds no model here, so the attribute is simply absent.
    @property
    def _engine(self) -> Any:
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        self._model._engine = value


__all__ = ["OCRReward"]
