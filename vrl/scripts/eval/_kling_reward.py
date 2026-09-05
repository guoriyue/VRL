"""Shared Kling VideoReward worker-config projection for the eval scripts.

``cosmos_predict25_kling_eval`` and ``video_reward_suite`` both build the same
Kling ``worker_config`` from a loaded config's ``reward.kwargs.kling_video_reward``
subtree. This is the single owner; it reads the resolved reward runtime so the
projection cannot drift from what training itself constructs.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.config.builders import RewardRuntimeConfig


def resolve_kling_worker_config(cfg: DictConfig) -> dict[str, Any]:
    """Project ``reward.kwargs.kling_video_reward`` into a Kling ``worker_config``."""

    reward_cfg = RewardRuntimeConfig.from_cfg(cfg).kwargs.get("kling_video_reward") or {}
    worker_config = dict(reward_cfg.get("worker_config") or {})
    worker_config.setdefault(
        "reward_model_name",
        str(reward_cfg.get("reward_name") or "KlingTeam/VideoReward@main"),
    )
    return worker_config
