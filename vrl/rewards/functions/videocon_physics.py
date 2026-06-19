"""VideoCon-Physics reward function (disk artifacts + Ray-actor transport).

A plain ``RewardFunction`` configured for the disk-artifact path (see
``_init_disk_artifact_reward``) whose worker loads the vendored mPLUG-Owl-Video
backbone with the VideoCon-Physics checkpoint and returns ``physical_commonsense``,
``semantic_adherence``, and ``overall`` sub-scores per artifact. This file only
pins the model factory and the physics-reward defaults.

A ``score_key`` of ``physical_commonsense`` (the default in
``configs/reward/videocon_physics.yaml``) gives GRPO a pure physics signal.
Switch to ``overall`` to also reward caption faithfulness.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction

_VIDEOCON_PHYSICS_MODEL = "vrl.rewards.models.videocon_physics:VideoConPhysicsModel"


class VideoConPhysicsReward(RewardFunction):
    """Reward whose VideoCon-Physics actor runs in a Ray pool on a separate GPU."""

    # Disk-artifact path: always scored by a Ray pool on its own GPU.
    default_execution = "pool"

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_VIDEOCON_PHYSICS_MODEL,
            config_key="videocon_physics",
            request_prefix="videocon-physics",
            debug_basename="videocon_physics",
            default_reward_name="videophysics/videocon_physics@main",
            default_score_key="physical_commonsense",
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["VideoConPhysicsReward"]
