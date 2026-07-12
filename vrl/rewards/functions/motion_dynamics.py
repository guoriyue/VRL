"""RAFT motion-dynamics quality-guard reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction, resolve_reward_component_device
from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
from vrl.rewards.runtime import LocalRewardRuntime


class MotionDynamicsReward(RewardFunction):
    """Local reward scoring generated-video motion magnitude via RAFT optical flow."""

    def __init__(
        self,
        device: str = "",
        reward_name: str = "motion_dynamics",
        score_key: str = "motion_dynamics",
        worker_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = dict(worker_config or {})
        if device:
            configured_device = str(cfg.get("device", "")).strip()
            _, effective_device = resolve_reward_component_device(
                resolved_device=str(device),
                overrides=[("worker_config.device", configured_device)],
            )
            cfg["device"] = effective_device
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
