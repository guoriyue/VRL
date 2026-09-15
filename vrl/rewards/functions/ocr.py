"""OCR text-matching reward (flow_grpo-compatible) over the selected transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class OCRReward(DiskArtifactRewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation; ``score_key``
    selects that edit similarity (``ocr``) or the exact whole-text match
    fraction (``ocr_match``).

    Transports: in-process (model built eagerly, media in memory, image or
    video tensors alike); ``kind=service`` (the driver launches a PaddleOCR
    service); ``kind=http`` (an operator-run service). ``debug_dir`` dumps the
    best-scoring frame and the recognized lines.
    """

    model_factory = "vrl.rewards.models.ocr:OCRRewardModel"
    request_prefix = "ocr"
    debug_basename = "ocr"
    default_reward_name = "ocr"
    default_score_key = "ocr"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """PaddleOCR runs CPU-only; never claim the resource-resolved GPU."""
        return "cpu"

    def __init__(
        self,
        device: str = "cuda",
        *,
        debug_dir: str | None = None,
        score_key: str = "ocr",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
    ) -> None:
        if score_key not in {"ocr", "ocr_match"}:
            raise ValueError("OCR score_key must be 'ocr' or 'ocr_match'")
        super().__init__(
            reward_name="ocr",
            score_key=score_key,
            worker_config={"debug_dir": debug_dir},
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )

    # Test seam onto the eagerly built in-process engine; a remote transport
    # builds no model here, so the attribute is simply absent.
    @property
    def _engine(self) -> Any:
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        self._model._engine = value


__all__ = ["OCRReward"]
