"""Shared reward ``worker_config`` projection for the eval scripts.

Four entrypoints score checkpoints with the same reward a training run used,
and each needs the run's own ``reward.kwargs.<component>.worker_config``. This
is the single owner: it reads the resolved reward runtime, so the projection
cannot drift from what training itself constructs, and ``RewardConfig.from_cfg``
resolves interpolations with ``throw_on_missing=True`` -- an unresolved
``${...}`` literal must never reach a reward loader.

Callers add their own enrichment (the device to score on, a data root) after
this returns; only the parts every caller shares live here.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.config.builders import RewardRuntimeConfig


def resolve_reward_worker_config(
    cfg: DictConfig,
    *,
    component: str,
    default_reward_model_name: str | None = None,
) -> dict[str, Any]:
    """Project ``reward.kwargs.<component>`` into that reward's ``worker_config``.

    ``component`` is the config key, and ``default_reward_model_name`` the
    per-reward contract constant to fall back to when the run named no model.
    """

    reward_cfg = RewardRuntimeConfig.from_cfg(cfg).kwargs.get(component) or {}
    worker_config = dict(reward_cfg.get("worker_config") or {})
    if default_reward_model_name is not None:
        worker_config.setdefault(
            "reward_model_name",
            str(reward_cfg.get("reward_name") or default_reward_model_name),
        )
    return worker_config
