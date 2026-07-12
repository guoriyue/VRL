"""OCR reward function, behavior mirrors flow_grpo OCR scorers.

Scores generated image/video outputs by how well OCR-detected text matches a
target string provided in rollout metadata. Single-image SD3 rewards mirror
``OcrScorer``; video/multi-frame rewards mirror ``OcrScorer_video_or_image``.

This is a thin ``RewardFunction`` wrapper over ``OCRRewardModel`` driven by the
local (in-process) transport. The substantive scoring logic lives in
``vrl.rewards.models.ocr``.

flow_grpo references:
- ``flow_grpo/ocr.py::OcrScorer``
- ``flow_grpo/ocr.py::OcrScorer_video_or_image``
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.ocr import OCRRewardModel
from vrl.rewards.runtime import InProcessRewardRuntime


class OCRReward(RewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation.

    When ``debug_dir`` is set, dumps the best-scoring frame along with the
    OCR-detected text and target to disk for reward-hacking audit.
    """

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """PaddleOCR runs CPU-only; never claim the resource-resolved GPU."""
        return "cpu"

    def __init__(self, device: str = "cuda", debug_dir: str | None = None) -> None:
        # Build eagerly so debug_dir creation fires now and tests can inject a
        # fake engine via ``reward._engine`` (proxied to the model below).
        model = OCRRewardModel({"device": device, "debug_dir": debug_dir})
        self._model = model
        super().__init__(
            reward_name="ocr",
            score_key="ocr",
            runtime=InProcessRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="image",
            ),
        )

    @property
    def _engine(self) -> Any:
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        self._model._engine = value


__all__ = ["OCRReward"]
