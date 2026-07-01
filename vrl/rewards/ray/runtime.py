"""Build and drive the reward model actor pool (model-agnostic transport)."""

from __future__ import annotations

import os
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
                "reward execution='pool' requires explicit worker_config. "
                "KlingVideoReward derives the model from reward_name; tests pass "
                "worker_config.model_factory pointing at a RewardModel factory.",
            )
        if not isinstance(worker_config_raw, Mapping):
            raise TypeError("reward worker_config must be a mapping")

        # Run-level placement injected by online.py via the owner; when present
        # the reward actors share that group and never remove it.
        placement = cfg.get("placement")
        placement_group = None
        bundle_indices: tuple[int, ...] = ()
        if placement is not None:
            placement_group = placement.placement_group
            bundle_indices = tuple(placement.bundle_indices)

        # Resident-reward (single-GPU staged pipeline): keep the actor alive and
        # park the model on CPU between scores (see RewardModelWorker). The actor
        # must NOT hold a Ray GPU reservation (it would block the rollout actor on
        # the shared single-GPU bundle), so it schedules as a 0-GPU actor and uses
        # the GPU via CUDA only while scoring; and it must NOT be released after
        # each call (that is the per-step kill+reload this avoids).
        resident = bool(cfg.get("resident", False))
        gpus_per_worker = 0.0 if resident else float(cfg.get("gpus_per_worker", 0.0))
        # The colocated GPU bundle reserves CPU = max(rollout, reward) on the
        # assumption only one of them runs at a time (the reward actor is killed
        # before the rollout relaunches). A resident reward stays alive alongside
        # the rollout, so it must reserve only a SMALL slice (1 CPU) that fits
        # under the max-sized bundle next to the rollout's 1 CPU; reserving the
        # full reward_cpus would push rollout_cpus + reward_cpus past the bundle
        # and deadlock the step-N+1 rollout relaunch on CPU. (A 0-CPU actor also
        # works for the bundle but destabilizes the raylet under heavy n=8 decode;
        # 1 CPU gives Ray normal resource accounting. The worker additionally caps
        # its thread pools so decode/preprocess never starves the raylet.)
        cpus_per_worker = 1.0 if resident else float(cfg.get("cpus_per_worker", 0.5))
        release_after_call = False if resident else bool(cfg.get("release_after_score", False))
        worker_config = dict(worker_config_raw)
        if resident:
            # A 0-GPU Ray actor gets CUDA_VISIBLE_DEVICES="" by default; this opts
            # the reward actor back into seeing the colocated GPU so it can move
            # its model there for scoring.
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            worker_config["resident"] = True

        return RayActorMethodRuntime(
            worker_cls=RewardModelWorker,
            worker_config=worker_config,
            method_name="score_batch",
            worker_id_prefix="reward",
            num_workers=int(cfg.get("num_workers", 1)),
            cpus_per_worker=cpus_per_worker,
            gpus_per_worker=gpus_per_worker,
            max_inflight_per_worker=int(cfg.get("max_inflight_batches", 1)),
            startup_method="load_model",
            init_ray=init_ray,
            ray_init_kwargs=dict(ray_init_kwargs or {}),
            release_after_call=release_after_call,
            placement_strategy=str(cfg.get("placement_strategy", "SPREAD")),
            expected_gpu_ids=(
                tuple(int(gpu_id) for gpu_id in placement.expected_gpu_ids)
                if placement is not None
                else tuple(int(gpu_id) for gpu_id in cfg.get("expected_gpu_ids", ()))
            ),
            placement_group=placement_group,
            bundle_indices=bundle_indices,
            validate_role="reward",
        )


__all__ = ["RayRewardRuntime"]
