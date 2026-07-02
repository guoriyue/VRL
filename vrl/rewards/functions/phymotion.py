"""PhyMotion human-dynamics reward function (disk artifacts + Ray-actor transport).

A plain ``RewardFunction`` on the disk-artifact path whose worker delegates to an
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

from vrl.rewards.base import RewardFunction

_PHYMOTION_MODEL = "vrl.rewards.models.phymotion:PhyMotionModel"


class PhyMotionReward(RewardFunction):
    """PhyMotion human-dynamics reward via an external scorer, run in-process."""

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_PHYMOTION_MODEL,
            config_key="phymotion",
            request_prefix="phymotion",
            debug_basename="phymotion",
            default_reward_name="phymotion",
            default_score_key="overall",
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["PhyMotionReward"]
