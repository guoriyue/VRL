"""HPSv3 reward function.

A ``ModelRewardFunction`` whose runtime loads ``MizzenAI/HPSv3``
(Qwen2-VL-7B + ranknet preference head) and scores a video's frames as
independent images, Flash-GRPO style: the default ``score_key``
``top_frame_mean`` is the mean of the best 30% of per-frame mu scores. This
file only pins the model factory and the HPSv3 defaults; transport and
scorer-local media adaptation are shared.
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class HPSv3Reward(ModelRewardFunction):
    """HPSv3 per-frame preference reward scored through the configured runtime."""

    model_factory = "vrl.rewards.models.hpsv3:HPSv3Model"
    name = "hpsv3"
    default_reward_name = "MizzenAI/HPSv3@main"
    default_score_key = "top_frame_mean"


__all__ = ["HPSv3Reward"]
