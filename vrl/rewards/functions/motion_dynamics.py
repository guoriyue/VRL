"""RAFT motion-dynamics quality-guard reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
from vrl.rewards.runtime import LocalRewardRuntime


class MotionDynamicsReward(RewardFunction):
    """Local reward scoring generated-video motion magnitude via RAFT optical flow."""

    default_execution = "inline"

    def __init__(
        self,
        device: str = "",
        reward_name: str = "motion_dynamics",
        score_key: str = "motion_dynamics",
        worker_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        cfg = dict(worker_config or {})
        if device:
            cfg.setdefault("device", device)
        model = MotionDynamicsModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="video",
            ),
        )


__all__ = ["MotionDynamicsReward"]
