"""Build and drive the reward model actor pool (model-agnostic transport)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from vrl.ray.runtime import RayActorMethodRuntime
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    shard_reward_request,
    validate_reward_results,
)
from vrl.rewards.ray.worker import RewardModelWorker


class RayRewardRuntime:
    """``RewardInferenceRuntime`` backed by a Ray reward actor pool.

    Wraps the actor pool + driver into the score_batch/shutdown transport seam
    so reward callers depend on the contract, not on the Ray plumbing.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        *,
        init_ray: bool = True,
        ray_init_kwargs: dict[str, Any] | None = None,
        actor_runtime: RayActorMethodRuntime | None = None,
    ) -> None:
        if actor_runtime is not None:
            self._actor = actor_runtime
        elif cfg is not None:
            self._actor = self._build_actor_runtime(
                cfg,
                init_ray=init_ray,
                ray_init_kwargs=ray_init_kwargs,
            )
        else:
            raise ValueError("RayRewardRuntime requires either cfg or actor_runtime")

    @property
    def num_workers(self) -> int:
        return self._actor.num_workers

    @property
    def worker_config(self) -> Mapping[str, Any]:
        return self._actor.worker_config

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        if not request.artifacts:
            return []
        shards = shard_reward_request(request, num_shards=self._actor.num_workers)
        queued_at_ns = time.perf_counter_ns()
        shards = [shard.with_metadata({"ray_queued_at_ns": queued_at_ns}) for shard in shards]
        nested = await self._actor.map(shards)
        results = [result for shard_results in nested for result in shard_results]
        return validate_reward_results(request, results)

    async def shutdown(self) -> None:
        await self._actor.shutdown()

    @staticmethod
    def _build_actor_runtime(
        cfg: Mapping[str, Any],
        *,
        init_ray: bool = True,
        ray_init_kwargs: dict[str, Any] | None = None,
    ) -> RayActorMethodRuntime:
        worker_config_raw = cfg.get("worker_config")
        if worker_config_raw is None:
            raise ValueError(
                "reward inference_runtime='ray' requires explicit worker_config. "
                "VideoReward production derives the model from reward_name; tests pass "
                "worker_config.model_factory pointing at a RewardModel factory.",
            )
        if not isinstance(worker_config_raw, Mapping):
            raise TypeError("reward worker_config must be a mapping")

        return RayActorMethodRuntime(
            worker_cls=RewardModelWorker,
            worker_config=dict(worker_config_raw),
            method_name="score_batch",
            worker_id_prefix="reward",
            num_workers=int(cfg.get("num_workers", 1)),
            cpus_per_worker=float(cfg.get("cpus_per_worker", 0.5)),
            gpus_per_worker=float(cfg.get("gpus_per_worker", 0.0)),
            max_inflight_per_worker=int(cfg.get("max_inflight_batches", 1)),
            startup_method="load_model",
            init_ray=init_ray,
            ray_init_kwargs=dict(ray_init_kwargs or {}),
            release_after_call=bool(cfg.get("release_after_score", False)),
            placement_strategy=str(cfg.get("placement_strategy", "SPREAD")),
            expected_gpu_ids=tuple(int(gpu_id) for gpu_id in cfg.get("expected_gpu_ids", ())),
            gpu_reservation_count=int(cfg.get("gpu_reservation_count", 0)),
            validate_role="reward",
        )


__all__ = ["RayRewardRuntime"]
