"""Cosmos3 reasoner reward function (disk artifacts + Ray-actor transport).

A plain ``RewardFunction`` on the disk-artifact path (see
``_init_disk_artifact_reward``) whose worker loads the Cosmos3 reasoner (Qwen3-VL
understanding tower) and returns ``task_success`` / ``contact_realism`` /
``temporal_consistency`` / ``physical_plausibility`` / ``overall`` per artifact.
This file only pins the model factory and the Cosmos3-reasoner defaults;
transport and disk-vs-in-memory wiring are shared.

Default ``score_key`` is ``task_success`` so a robotics compound gets the
goal-completion axis; switch to ``overall`` for a blended physical-AI signal.
The judge needs a pre-remapped reasoner checkpoint (see the model module /
``configs/reward/cosmos3_reasoner.yaml``).
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction

_COSMOS3_REASONER_MODEL = "vrl.rewards.models.cosmos3_reasoner:Cosmos3ReasonerRewardModel"


class Cosmos3ReasonerReward(RewardFunction):
    """Reward whose Cosmos3-reasoner actor runs in a Ray pool on a separate GPU."""

    # Disk-artifact path: always scored by a Ray pool on its own GPU.
    default_execution = "pool"

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_COSMOS3_REASONER_MODEL,
            config_key="cosmos3_reasoner",
            request_prefix="cosmos3_reasoner",
            debug_basename="cosmos3_reasoner",
            default_reward_name="nvidia/Cosmos3-Nano",
            default_score_key="task_success",
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["Cosmos3ReasonerReward"]
