"""OCR text-matching reward (flow_grpo-compatible) over the selected transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.models.ocr import OCRRewardModel
from vrl.rewards.protocols import RewardScorer


class OCRReward(DiskArtifactRewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation.

    Transports: in-process (model built eagerly, media in memory, image or
    video tensors alike); ``kind=service`` (the driver launches a PaddleOCR
    service and hands it every knob below); ``kind=http`` (operator-run
    service owns the knobs, so a non-default value here is refused).
    ``debug_dir`` dumps the best-scoring frame and the OCR decision.
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
        engine_profile: str = "flow_grpo_compat",
        text_selection: str = "all_text",
        substring_full_credit: bool = True,
        exclusive_alphanumeric_lines: bool = False,
        extra_line_min_confidence: float = 0.5,
        near_duplicate_min_similarity: float | None = None,
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
        model_config = {
            "debug_dir": debug_dir,
            "engine_profile": engine_profile,
            "text_selection": text_selection,
            "substring_full_credit": substring_full_credit,
            "exclusive_alphanumeric_lines": exclusive_alphanumeric_lines,
            "extra_line_min_confidence": extra_line_min_confidence,
            "near_duplicate_min_similarity": near_duplicate_min_similarity,
        }
        self._model: OCRRewardModel | None = None
        if scorer is not None:
            remote_owned = sorted(
                name
                for name, value in model_config.items()
                if value != _MODEL_CONFIG_DEFAULTS[name]
            )
            if remote_owned:
                raise ValueError(
                    "OCR reward over HTTP inference scores in the standalone reward "
                    f"service; move {remote_owned} into that service's worker_config "
                    "(vrl/config/reward_service/ocr_paddle.yaml) instead of "
                    "reward.kwargs.ocr",
                )
        super().__init__(
            reward_name="ocr",
            score_key=score_key,
            worker_config=model_config,
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )

    @property
    def _engine(self) -> Any:
        if self._model is None:
            raise AttributeError("OCR engine lives in the remote reward service")
        return self._model._engine

    @_engine.setter
    def _engine(self, value: Any) -> None:
        if self._model is None:
            raise AttributeError("OCR engine lives in the remote reward service")
        self._model._engine = value


# The in-process defaults double as the "nothing to forward" check for the
# external HTTP transport, where the operator's service owns these knobs.
_MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "debug_dir": None,
    "engine_profile": "flow_grpo_compat",
    "text_selection": "all_text",
    "substring_full_credit": True,
    "exclusive_alphanumeric_lines": False,
    "extra_line_min_confidence": 0.5,
    "near_duplicate_min_similarity": None,
}


__all__ = ["OCRReward"]
