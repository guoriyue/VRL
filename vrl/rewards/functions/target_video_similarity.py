"""Target-aware Video2World similarity reward."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.target_video_similarity import TargetVideoSimilarityModel
from vrl.rewards.runtime import LocalRewardRuntime


class TargetVideoSimilarityReward(RewardFunction):
    """Local CPU reward comparing generated video to target media from metadata."""

    default_execution = "inline"

    def __init__(
        self,
        device: str = "cpu",
        reward_name: str = "target_video_similarity",
        score_key: str = "target_similarity",
        worker_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        cfg = dict(worker_config or {})
        cfg.setdefault("device", device)
        model = TargetVideoSimilarityModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="video",
            ),
        )


__all__ = ["TargetVideoSimilarityReward"]
