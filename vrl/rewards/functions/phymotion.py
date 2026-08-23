"""PhyMotion human-dynamics reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` on the disk-artifact path whose runtime delegates to an
external PhyMotion environment (SMPL + MuJoCo) via a configured command (see
``vrl.rewards.models.phymotion.PhyMotionModel``). Returns
``kinematic`` / ``contact`` / ``dynamic`` / ``overall``; default
``score_key`` is ``overall``.

PhyMotion is opt-in and external by design — its backend is not a VRL base
dependency, so this reward only loads when ``worker_config.phymotion_cmd`` points
at a working PhyMotion install.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class PhyMotionReward(DiskArtifactRewardFunction):
    """PhyMotion human-dynamics reward via an external scorer."""

    model_factory = "vrl.rewards.models.phymotion:PhyMotionModel"
    request_prefix = "phymotion"
    debug_basename = "phymotion"
    default_reward_name = "phymotion"
    default_score_key = "overall"


__all__ = ["PhyMotionReward"]
