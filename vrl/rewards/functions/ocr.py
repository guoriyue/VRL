"""OCR reward function, behavior mirrors flow_grpo OCR scorers.

Scores generated image/video outputs by how well OCR-detected text matches a
target string provided in sample metadata. The default policy mirrors the
Flow-GRPO scorers; exact-text curricula may preserve and rank complete OCR lines.

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

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.ocr import OCRRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class OCRReward(InferenceRewardFunction):
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

    def __init__(
        self,
        device: str = "cuda",
        debug_dir: str | None = None,
        engine_profile: str = "flow_grpo_compat",
        text_selection: str = "all_text",
        substring_full_credit: bool = True,
        exclusive_alphanumeric_lines: bool = False,
        extra_line_min_confidence: float = 0.5,
        near_duplicate_min_similarity: float | None = None,
    ) -> None:
        # Build eagerly so debug_dir creation fires now and tests can inject a
        # fake engine via ``reward._engine`` (proxied to the model below).
        # ``device`` stays in the RewardFunction constructor contract, while
        # resolve_execution_device above is the sole CPU placement owner.
        del device
        model = OCRRewardModel(
            {
                "debug_dir": debug_dir,
                "engine_profile": engine_profile,
                "text_selection": text_selection,
                "substring_full_credit": substring_full_credit,
                "exclusive_alphanumeric_lines": exclusive_alphanumeric_lines,
                "extra_line_min_confidence": extra_line_min_confidence,
                "near_duplicate_min_similarity": near_duplicate_min_similarity,
            }
        )
        self._model = model
        super().__init__(
            reward_name="ocr",
            score_key="ocr",
            scorer=InProcessRewardScorer(model=model),
        )

    @property
    def _engine(self) -> Any:
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        self._model._engine = value


__all__ = ["OCRReward"]
