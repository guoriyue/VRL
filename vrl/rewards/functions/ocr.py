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

from vrl.rewards.artifacts import InMemoryRewardArtifactStore, MediaType, RewardArtifactStore
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.models.ocr import OCRRewardModel
from vrl.rewards.protocols import RewardScorer
from vrl.rewards.runtime import InProcessRewardScorer


class OCRReward(DiskArtifactRewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    the flow_grpo ``OcrScorer_video_or_image`` implementation.

    Two transports, one scorer contract:

    - in-process (default): the PaddleOCR model is built eagerly here, scores
      in the driver, and media rides the request in memory (image or video
      tensors alike, exactly as before);
    - ``reward.inference.ocr.kind=http``: the registry injects the HTTP client,
      media is written to ``artifact_dir`` as ``.pt`` tensors and scored by a
      standalone ``vrl-reward-service`` running ``OCRRewardModel`` on its own
      CPU (``vrl/config/reward_service/ocr_paddle.yaml``), which keeps the
      OCR work off the trainer's launch-bound event loop. The engine/scoring
      knobs then belong to the service's ``worker_config``; setting one here to
      a non-default value is refused rather than silently ignored.

    When ``debug_dir`` is set, dumps the best-scoring frame along with the
    OCR-detected text and target to disk for reward-hacking audit.
    """

    model_factory = "vrl.rewards.models.ocr:OCRRewardModel"
    request_prefix = "ocr"
    debug_basename = "ocr"
    default_reward_name = "ocr"
    default_score_key = "ocr"
    default_artifact_format = "tensor"
    default_media_type = "image"

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
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
    ) -> None:
        if score_key not in {"ocr", "ocr_match"}:
            raise ValueError("OCR score_key must be 'ocr' or 'ocr_match'")
        # ``device`` stays in the RewardFunction constructor contract, while
        # resolve_execution_device above is the sole CPU placement owner.
        del device
        model_config = {
            "debug_dir": debug_dir,
            "engine_profile": engine_profile,
            "text_selection": text_selection,
            "substring_full_credit": substring_full_credit,
            "exclusive_alphanumeric_lines": exclusive_alphanumeric_lines,
            "extra_line_min_confidence": extra_line_min_confidence,
            "near_duplicate_min_similarity": near_duplicate_min_similarity,
        }
        artifact_store: RewardArtifactStore | None
        if scorer is None:
            # Build eagerly so debug_dir creation fires now and tests can inject
            # a fake engine via ``reward._engine`` (proxied to the model below).
            model = OCRRewardModel(model_config)
            self._model: OCRRewardModel | None = model
            scorer = InProcessRewardScorer(model=model)
            artifact_store = InMemoryRewardArtifactStore()
        else:
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
            self._model = None
            artifact_store = None
        super().__init__(
            reward_name="ocr",
            score_key=score_key,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
            scorer=scorer,
            artifact_store=artifact_store,
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
# HTTP transport, where the service owns these knobs.
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
