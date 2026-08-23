"""Cosmos3 reasoner reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` on the disk-artifact path whose runtime loads the Cosmos3 reasoner (Qwen3-VL
understanding tower) and returns ``task_success`` / ``contact_realism`` /
``temporal_consistency`` / ``physical_plausibility`` / ``overall`` per artifact.
This file only pins the model factory and the Cosmos3-reasoner defaults;
transport and disk-vs-in-memory wiring are shared.

Default ``score_key`` is ``task_success`` so a robotics compound gets the
goal-completion axis; switch to ``overall`` for a blended physical-AI signal.
The judge needs a pre-remapped reasoner checkpoint (see the model module /
``vrl/config/presets/reward/cosmos3_reasoner.yaml``).
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class Cosmos3ReasonerReward(DiskArtifactRewardFunction):
    """Cosmos3-reasoner reward scored from disk artifacts."""

    model_factory = "vrl.rewards.models.cosmos3_reasoner:Cosmos3ReasonerRewardModel"
    request_prefix = "cosmos3_reasoner"
    debug_basename = "cosmos3_reasoner"
    default_reward_name = "nvidia/Cosmos3-Nano"
    default_score_key = "task_success"


__all__ = ["Cosmos3ReasonerReward"]
