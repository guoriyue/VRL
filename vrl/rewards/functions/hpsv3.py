"""HPSv3 reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` whose runtime loads ``MizzenAI/HPSv3``
(Qwen2-VL-7B + ranknet preference head) and scores a video's frames as
independent images, Flash-GRPO style: the default ``score_key``
``top_frame_mean`` is the mean of the best 30% of per-frame mu scores. This
file only pins the model factory and the HPSv3 defaults; transport and
disk-vs-in-memory wiring are shared.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class HPSv3Reward(DiskArtifactRewardFunction):
    """HPSv3 per-frame preference reward scored from disk artifacts."""

    model_factory = "vrl.rewards.models.hpsv3:HPSv3Model"
    request_prefix = "hpsv3"
    debug_basename = "hpsv3"
    default_reward_name = "MizzenAI/HPSv3@main"
    default_score_key = "top_frame_mean"


__all__ = ["HPSv3Reward"]
