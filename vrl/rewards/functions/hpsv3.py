"""HPSv3 reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` whose runtime loads ``MizzenAI/HPSv3``
(Qwen2-VL-7B + ranknet preference head) and scores a video's frames as
independent images, Flash-GRPO style: the default ``score_key``
``top_frame_mean`` is the mean of the best 30% of per-frame mu scores. This
file only pins the model factory and the HPSv3 defaults; transport and
disk-vs-in-memory wiring are shared.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_HPSV3_MODEL = "vrl.rewards.models.hpsv3:HPSv3Model"


class HPSv3Reward(DiskArtifactRewardFunction):
    """HPSv3 per-frame preference reward scored from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "MizzenAI/HPSv3@main",
        score_key: str = "top_frame_mean",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_factory=_HPSV3_MODEL,
            request_prefix="hpsv3",
            debug_basename="hpsv3",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["HPSv3Reward"]
