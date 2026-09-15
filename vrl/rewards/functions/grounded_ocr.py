"""Grounded OCR: PP-OCR text match guarded by a Codex CLI judge (both CPU-side)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class GroundedOCRReward(DiskArtifactRewardFunction):
    """PP-OCR text match plus a Codex CLI judge guard; both execute off-GPU.

    In-process the model is built eagerly and media rides the request in
    memory; ``inference.kind=service`` hands ``ocr``/``guard`` to a
    driver-launched service.
    """

    model_factory = "vrl.rewards.models.grounded_ocr:GroundedOCRRewardModel"
    request_prefix = "grounded-ocr"
    debug_basename = "grounded_ocr"
    default_reward_name = "grounded_ocr"
    default_score_key = "grounded_ocr"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

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
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
    ) -> None:
        super().__init__(
            reward_name="grounded_ocr",
            score_key="grounded_ocr",
            worker_config={"ocr": dict(ocr), "guard": dict(guard)},
            device=device,
            debug_dir=debug_dir,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["GroundedOCRReward"]
