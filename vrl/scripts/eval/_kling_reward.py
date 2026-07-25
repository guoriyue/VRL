"""Shared Kling VideoReward worker-config projection for the eval scripts.

``cosmos_predict25_kling_eval`` and ``video_reward_suite`` both build the same
Kling ``worker_config`` from a loaded config's ``reward.kwargs.kling_video_reward``
subtree. This is the single owner. Interpolations are resolved (``resolve=True``)
so no unresolved ``${...}`` literal can leak into the reward loader.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def resolve_kling_worker_config(cfg: DictConfig) -> dict[str, Any]:
    """Project ``reward.kwargs.kling_video_reward`` into a Kling ``worker_config``."""

    reward_cfg = OmegaConf.select(cfg, "reward.kwargs.kling_video_reward", default={})
    reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True) or {}
    if not isinstance(reward_cfg, dict):
        raise ValueError("reward.kwargs.kling_video_reward must be a mapping")
    worker_config = dict(reward_cfg.get("worker_config") or {})
    worker_config.setdefault(
        "reward_model_name",
        str(reward_cfg.get("reward_name") or "KlingTeam/VideoReward@main"),
    )
    return worker_config
