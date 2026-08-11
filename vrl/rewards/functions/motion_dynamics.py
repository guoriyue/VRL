"""RAFT motion-dynamics quality-guard reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction, reward_worker_config_with_device
from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
from vrl.rewards.runtime import InProcessRewardInferenceRuntime


class MotionDynamicsReward(InferenceRewardFunction):
    """Local reward scoring generated-video motion magnitude via RAFT optical flow."""

    def __init__(
        self,
        device: str = "",
        reward_name: str = "motion_dynamics",
        score_key: str = "motion_dynamics",
        worker_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = reward_worker_config_with_device(worker_config, device=str(device))
        model = MotionDynamicsModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            inference_runtime=InProcessRewardInferenceRuntime(model=model),
        )


__all__ = ["MotionDynamicsReward"]
