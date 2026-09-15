"""NSFW safety penalty from a Falconsai image classifier over sampled frames."""

from __future__ import annotations

from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class NSFWSafetyReward(DiskArtifactRewardFunction):
    """NSFW safety penalty from a Falconsai image classifier over sampled frames.

    In-process the model is built here and media rides the request in memory;
    ``inference.kind=service`` hands the same kwargs to a driver-launched
    service that scores this reward's ``.pt`` artifacts.
    """

    model_factory = "vrl.rewards.models.nsfw_safety:NSFWSafetyRewardModel"
    request_prefix = "nsfw_safety"
    debug_basename = "nsfw_safety"
    default_reward_name = "nsfw_safety"
    default_score_key = "nsfw_safety"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

    def __init__(
        self,
        device: str = "cuda",
        *,
        score_key: str = "nsfw_safety",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
        **kwargs: Any,
    ) -> None:
        # The classifier model's own injectable callable is also called
        # ``scorer`` (tests hand a ``images -> probabilities`` function). A
        # transport scorer implements ``score_batch``; anything else is the
        # model's kwarg and travels in worker_config.
        if scorer is not None and not hasattr(scorer, "score_batch"):
            kwargs["scorer"] = scorer
            scorer = None
        super().__init__(
            reward_name="nsfw_safety",
            score_key=score_key,
            worker_config=kwargs,
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )

    # The classifier's device key differs from the generic ceiling key.
    device_config_key = "classifier_device"


__all__ = ["NSFWSafetyReward"]
