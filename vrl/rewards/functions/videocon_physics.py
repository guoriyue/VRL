"""VideoCon-Physics reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` configured for the disk-artifact path (see
``_init_disk_artifact_reward``) whose runtime loads the vendored mPLUG-Owl-Video
backbone with the VideoCon-Physics checkpoint and returns ``physical_commonsense``,
``semantic_adherence``, and ``overall`` sub-scores per artifact. This file only
pins the model factory and the physics-reward defaults.

A ``score_key`` of ``physical_commonsense`` (the default in
``vrl/config/presets/reward/videocon_physics.yaml``) gives GRPO a pure physics signal.
Switch to ``overall`` to also reward caption faithfulness.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_VIDEOCON_PHYSICS_MODEL = "vrl.rewards.models.videocon_physics:VideoConPhysicsModel"


class VideoConPhysicsReward(DiskArtifactRewardFunction):
    """VideoCon-Physics reward scored from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "videophysics/videocon_physics@main",
        score_key: str = "physical_commonsense",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        self._init_disk_artifact_reward(
            model_factory=_VIDEOCON_PHYSICS_MODEL,
            request_prefix="videocon-physics",
            debug_basename="videocon_physics",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["VideoConPhysicsReward"]
