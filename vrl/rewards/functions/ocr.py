"""OCR reward function, behavior mirrors flow_grpo OCR scorers.

Scores generated image/video outputs by how well OCR-detected text matches a
target string provided in rollout metadata. Single-image SD3 rewards mirror
``OcrScorer``; video/multi-frame rewards mirror ``OcrScorer_video_or_image``.

This is a thin ``RewardFunction`` wrapper over ``OCRRewardModel`` driven by the
local (in-process) transport. The substantive scoring logic lives in
``vrl.rewards.models.ocr_model``.

flow_grpo references:
- ``flow_grpo/ocr.py::OcrScorer``
- ``flow_grpo/ocr.py::OcrScorer_video_or_image``
"""

from __future__ import annotations

import re
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.ocr_model import OCRRewardModel
from vrl.rewards.runtime import LocalRewardRuntime


def _normalize_text(text: str) -> str:
    """Normalize text for helper-level OCR edit-distance tests."""
    normalized = re.sub(r"[^a-z0-9\s]+", "", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_edit_distance(a: str, b: str) -> float:
    """Return Levenshtein distance normalized by the longer input length."""
    if not a and not b:
        return 0.0
    distance = _edit_distance(a, b)
    return distance / max(len(a), len(b), 1)


def _edit_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein implementation for tests."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


class OCRReward(RewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation.

    When ``debug_dir`` is set, dumps the best-scoring frame along with the
    OCR-detected text and target to disk for reward-hacking audit.
    """

    def __init__(self, device: str = "cuda", debug_dir: str | None = None) -> None:
        # Build eagerly so debug_dir creation fires now and tests can inject a
        # fake engine via ``reward._engine`` (proxied to the model below).
        model = OCRRewardModel({"device": device, "debug_dir": debug_dir})
        self._model = model
        super().__init__(
            reward_name="ocr",
            score_key="ocr",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )

    @property
    def _engine(self) -> Any:
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        self._model._engine = value


__all__ = ["OCRReward"]
