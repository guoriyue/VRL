"""PhyMotion human-dynamics reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` on the disk-artifact path whose runtime delegates to an
external PhyMotion environment (SMPL + MuJoCo) via a configured command (see
``vrl.rewards.models.phymotion.PhyMotionModel``). Returns
``kinematic_plausibility`` / ``contact_balance`` / ``dynamic_feasibility`` /
``overall``; default ``score_key`` is ``overall``.

PhyMotion is opt-in and external by design — its backend is not a VRL base
dependency, so this reward only loads when ``worker_config.phymotion_cmd`` points
at a working PhyMotion install.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_PHYMOTION_MODEL = "vrl.rewards.models.phymotion:PhyMotionModel"


class PhyMotionReward(DiskArtifactRewardFunction):
    """PhyMotion human-dynamics reward via an external scorer."""

    def __init__(
        self,
        *,
        reward_name: str = "phymotion",
        score_key: str = "overall",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        self._init_disk_artifact_reward(
            model_factory=_PHYMOTION_MODEL,
            request_prefix="phymotion",
            debug_basename="phymotion",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["PhyMotionReward"]
