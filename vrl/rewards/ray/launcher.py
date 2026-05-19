"""Launch Ray-backed reward inference runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.ray.runtime import RayActorMethodRuntime
from vrl.rewards.inference import RewardInferenceRuntime
from vrl.rewards.ray.runtime import RewardInferenceActorRuntime
from vrl.rewards.scoring_worker import RewardScoringWorker


def build_reward_ray_runtime(
    cfg: Mapping[str, Any],
    *,
    init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build a Ray-backed reward inference runtime from reward config."""

    worker_config = cfg.get("worker_config")
    if worker_config is None:
        raise ValueError(
            "reward inference_runtime='ray' requires reward.kwargs.video_reward.worker_config",
        )
    if not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping")

    return RewardInferenceActorRuntime(
        RayActorMethodRuntime(
            worker_cls=RewardScoringWorker,
            worker_config=worker_config,
            method_name="score_batch",
            worker_id_prefix="reward",
            num_workers=int(cfg.get("num_workers", 1)),
            cpus_per_worker=float(cfg.get("cpus_per_worker", 0.5)),
            gpus_per_worker=float(cfg.get("gpus_per_worker", 0.0)),
            max_inflight_per_worker=int(cfg.get("max_inflight_batches", 1)),
            startup_method="load_scorer",
            init_ray=init_ray,
            ray_init_kwargs=dict(ray_init_kwargs or {}),
            release_after_call=bool(cfg.get("release_after_score", False)),
        ),
    )


__all__ = ["build_reward_ray_runtime"]
