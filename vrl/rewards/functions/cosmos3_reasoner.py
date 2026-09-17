"""Cosmos3 reasoner reward function.

A ``ModelRewardFunction`` whose runtime loads the Cosmos3 reasoner (Qwen3-VL
understanding tower) and returns ``task_success`` / ``contact_realism`` /
``temporal_consistency`` / ``physical_plausibility`` / ``overall`` per artifact.
This file only pins the model factory and the Cosmos3-reasoner defaults;
transport and scorer-local media adaptation are shared.

Default ``score_key`` is ``task_success`` so a robotics compound gets the
goal-completion axis; switch to ``overall`` for a blended physical-AI signal.
The judge needs a pre-remapped reasoner checkpoint (see the model module /
``vrl/config/presets/reward/cosmos3_reasoner.yaml``).
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class Cosmos3ReasonerReward(ModelRewardFunction):
    """Cosmos3-reasoner reward scored through the configured runtime."""

    model_factory = "vrl.rewards.models.cosmos3_reasoner:Cosmos3ReasonerRewardModel"
    name = "cosmos3_reasoner"
    default_reward_name = "nvidia/Cosmos3-Nano"
    default_score_key = "task_success"


__all__ = ["Cosmos3ReasonerReward"]
