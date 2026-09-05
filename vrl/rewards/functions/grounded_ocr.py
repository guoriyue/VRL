"""Framework binding for the atomic grounded-OCR reward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.grounded_ocr import GroundedOCRRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class GroundedOCRReward(InferenceRewardFunction):
    """Score one text rendering only when its requested carrier is valid."""

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """Both PP-OCR and the Codex CLI judge execute outside the trainer GPU."""
        return "cpu"

    def __init__(
        self,
        *,
        ocr: Mapping[str, Any],
        guard: Mapping[str, Any],
        debug_dir: str = "",
        device: str = "cpu",
    ) -> None:
        del device
        model = GroundedOCRRewardModel({"ocr": ocr, "guard": guard})
        super().__init__(
            reward_name="grounded_ocr",
            score_key="grounded_ocr",
            scorer=InProcessRewardScorer(model=model),
            debug_dir=debug_dir,
            request_prefix="grounded-ocr",
            debug_basename="grounded_ocr",
        )


__all__ = ["GroundedOCRReward"]
