"""Optical-flow motion-dynamics reward over a video."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class MotionDynamicsReward(DiskArtifactRewardFunction):
    """Optical-flow motion-dynamics reward over a video.

    ``worker_config`` is the model's own vocabulary; the resolved device is
    stamped through the ceiling check. In-process the model is built eagerly
    and media rides the request in memory; ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.motion_dynamics:MotionDynamicsModel"
    request_prefix = "motion_dynamics"
    debug_basename = "motion_dynamics"
    default_reward_name = "motion_dynamics"
    default_score_key = "motion_dynamics"
    default_artifact_format = "tensor"
    default_media_type = "video"
    in_process_media = "memory"
    eager_model = True

    def __init__(
        self,
        device: str = "",
        *,
        reward_name: str = "motion_dynamics",
        score_key: str = "motion_dynamics",
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
    ) -> None:
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            worker_config=dict(worker_config or {}),
            device=device or None,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["MotionDynamicsReward"]
