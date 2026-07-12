"""Cosmos3 reasoner reward function (disk artifacts + in-process runtime).

A ``CumemRewardFunction`` on the disk-artifact path (see
``_init_disk_artifact_reward``) whose runtime loads the Cosmos3 reasoner (Qwen3-VL
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

from typing import Any

from vrl.rewards.base import CumemRewardFunction

_COSMOS3_REASONER_MODEL = "vrl.rewards.models.cosmos3_reasoner:Cosmos3ReasonerRewardModel"


class Cosmos3ReasonerReward(CumemRewardFunction):
    """Cosmos3-reasoner reward scored in-process from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "nvidia/Cosmos3-Nano",
        score_key: str = "task_success",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        self._init_disk_artifact_reward(
            model_factory=_COSMOS3_REASONER_MODEL,
            request_prefix="cosmos3_reasoner",
            debug_basename="cosmos3_reasoner",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["Cosmos3ReasonerReward"]
