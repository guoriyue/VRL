"""WD tagger (onnxruntime, CPU) tag-adherence reward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer

WD_TAGGER_SCORE_KEYS = ("wd_tagger_dense", "wd_tagger_recall")


class WDTaggerReward(DiskArtifactRewardFunction):
    """WD tagger (onnxruntime, CPU) tag-adherence reward.

    In-process the model is built here and media rides the request in memory;
    ``inference.kind=service`` hands the same kwargs to a driver-launched
    service that scores this reward's ``.pt`` artifacts.
    """

    model_factory = "vrl.rewards.models.wd_tagger:WDTaggerRewardModel"
    request_prefix = "wd_tagger"
    debug_basename = "wd_tagger"
    default_reward_name = "wd_tagger"
    default_score_key = "wd_tagger_dense"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"

    def __init__(
        self,
        device: str = "cpu",
        *,
        score_key: str = "wd_tagger_dense",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
        **kwargs: Any,
    ) -> None:
        if score_key not in WD_TAGGER_SCORE_KEYS:
            raise ValueError(
                f"wd_tagger score_key must be one of {list(WD_TAGGER_SCORE_KEYS)}, got {score_key!r}"
            )
        super().__init__(
            reward_name="wd_tagger",
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


__all__ = ["WD_TAGGER_SCORE_KEYS", "WDTaggerReward"]
