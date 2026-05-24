"""Launch Ray-backed reward inference runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.ray.runtime import RayActorMethodRuntime
from vrl.rewards.inference import RewardInferenceRuntime
from vrl.rewards.ray.runtime import RewardInferenceActorRuntime
from vrl.rewards.ray.worker import RewardScoringWorker


def build_reward_inference_runtime(
    cfg: Mapping[str, Any],
    *,
    init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build the Ray reward inference runtime selected by config."""

    runtime = str(cfg.get("inference_runtime", ""))
    if runtime != "ray":
        raise ValueError("reward inference_runtime must be 'ray'")
    return build_reward_ray_runtime(
        cfg,
        init_ray=init_ray,
        ray_init_kwargs=ray_init_kwargs,
    )


def build_reward_ray_runtime(
    cfg: Mapping[str, Any],
    *,
    init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build a Ray-backed reward inference runtime from reward config."""

    worker_config_raw = cfg.get("worker_config")
    if worker_config_raw is None:
        raise ValueError(
            "reward inference_runtime='ray' requires explicit reward.kwargs.video_reward.worker_config. "
            "Use worker_config.scorer='import_path' for model-backed rewards, or "
            "worker_config.scorer='tensor_mean' for tensor smoke tests.",
        )
    if not isinstance(worker_config_raw, Mapping):
        raise TypeError("reward worker_config must be a mapping")
    worker_config = dict(worker_config_raw)

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
            placement_strategy=str(cfg.get("placement_strategy", "SPREAD")),
            expected_gpu_ids=tuple(int(gpu_id) for gpu_id in cfg.get("expected_gpu_ids", ())),
            gpu_reservation_count=int(cfg.get("gpu_reservation_count", 0)),
            validate_role="reward",
        ),
    )


__all__ = [
    "build_reward_inference_runtime",
    "build_reward_ray_runtime",
]
