"""VideoCon-Physics reward function.

A ``ModelRewardFunction`` whose runtime loads the vendored mPLUG-Owl-Video
backbone with the VideoCon-Physics checkpoint and returns ``physical_commonsense``,
``semantic_adherence``, and ``overall`` sub-scores per artifact. This file only
pins the model factory and the physics-reward defaults.

A ``score_key`` of ``physical_commonsense`` (the default in
``vrl/config/presets/reward/videocon_physics.yaml``) gives GRPO a pure physics signal.
Switch to ``overall`` to also reward caption faithfulness.
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class VideoConPhysicsReward(ModelRewardFunction):
    """VideoCon-Physics reward scored through the configured runtime."""

    model_factory = "vrl.rewards.models.videocon_physics:VideoConPhysicsModel"
    name = "videocon_physics"
    default_reward_name = "videophysics/videocon_physics@main"
    default_score_key = "physical_commonsense"


__all__ = ["VideoConPhysicsReward"]
