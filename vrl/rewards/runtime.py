"""Reward inference runtime wiring and in-process transport."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import (
    RewardInferenceConfig,
    parse_reward_inference_config,
)
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
    RewardMemoryReleaseProof,
    score_artifacts_with_model,
    validate_reward_parking_residual,
)
from vrl.utils.cuda_memory import (
    CumemPool,
    gpu_used_bytes,
    release_cuda_memory_for_parking,
)


class InProcessRewardRuntime:
    """``RewardInferenceRuntime`` that runs a ``RewardModel`` in this process.

    ``worker_config.sleep_offload`` opts a heavyweight model into the same
    sleep/wake semantics the rollout lease uses: the model holds no GPU memory
    between scores — the rollout/trainer own the card then — and comes back
    only for scoring; the caller's step ordering (rollout releases the GPU
    before rewards score) already guarantees the card is free at that point.
    Small in-memory rewards (CLIP-class) leave the knob off and stay resident.

    Offload is cumem-only (vLLM's CuMemAllocator is a hard requirement of
    ``sleep_offload``): the model is built under a backup tag, so process-wide
    ``sleep`` releases physical pages while preserving this reward's contents
    in pinned host RAM, and ``wake`` remaps without cudaMalloc. CuMem tags do
    not isolate sleep operations; the shared-topology preflight therefore
    permits at most one configured GPU reward component per process; zero-weight
    observation-only scorers still execute and therefore count.
    """

    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = True

    def __init__(
        self,
        worker_config: Mapping[str, Any] | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self._worker_config = dict(worker_config or {})
        self._sleep_offload = bool(self._worker_config.get("sleep_offload", False))
        if model is not None and self._sleep_offload:
            raise ValueError(
                "sleep_offload requires the runtime to build the model itself "
                "(worker_config.model_factory) so its allocations land in the "
                "cumem pool; an already-built injected model cannot be pooled.",
            )
        self._model = model
        self._pool: CumemPool | None = None
        self._last_request_id: str | None = None
        self._preload_gpu_used_bytes: int | None = None
        self._parking_residual_bytes_limit = int(
            self._worker_config.get("memory_parking_residual_bytes_limit", 0),
        )
        if self._parking_residual_bytes_limit < 0:
            raise ValueError("reward memory parking residual limit must be >= 0")

    @property
    def requires_memory_parking(self) -> bool:
        """Whether topology/config requires this runtime to release GPU pages."""

        return self._sleep_offload

    async def park_memory(self) -> RewardMemoryReleaseProof:
        """Park reward pages and return request-bound proof; safe to retry."""

        if not self._sleep_offload:
            raise RuntimeError(
                "reward runtime was not configured for complete memory parking",
            )
        request_id = self._last_request_id
        if request_id is None:
            raise RuntimeError(
                "reward runtime cannot park before a score request has started",
            )
        pool = self._pool
        if pool is None:
            raise RuntimeError(
                "reward runtime cannot prove memory parking before its pooled model is built",
            )
        if not pool.asleep:
            # CumemPool marks itself asleep only after allocator.sleep returns.
            # A failure therefore leaves this branch retryable on the next call.
            pool.sleep()
        device = str(self._worker_config.get("device", ""))
        release_cuda_memory_for_parking(device)
        baseline_bytes = self._preload_gpu_used_bytes
        if baseline_bytes is None:
            raise RuntimeError("reward runtime has no pre-load GPU parking baseline")
        proof = RewardMemoryReleaseProof(
            request_id=request_id,
            released=True,
            baseline_gpu_used_bytes=baseline_bytes,
            residual_gpu_used_bytes=gpu_used_bytes(device),
            residual_bytes_limit=self._parking_residual_bytes_limit,
        )
        proof.validate(request_id=request_id)
        return proof

    def _ensure_model(self) -> Any:
        if self._model is None:
            factory_path = str(self._worker_config.get("model_factory", "")).strip()
            if not factory_path:
                raise ValueError(
                    "InProcessRewardRuntime requires worker_config.model_factory "
                    "(import path to a RewardModel factory) or an explicit model",
                )
            module_path, attr = factory_path.split(":", 1)
            factory = getattr(importlib.import_module(module_path), attr)
            if self._sleep_offload:
                # Build inside the pool so every CUDA allocation the factory
                # makes (from_pretrained, .to(device), buffers) is tagged and
                # sleep/wake can release/restore it wholesale.
                self._preload_gpu_used_bytes = gpu_used_bytes(
                    str(self._worker_config.get("device", "")),
                )
                self._pool = CumemPool.require()
                with self._pool.building():
                    self._model = factory(self._worker_config)
            else:
                self._model = factory(self._worker_config)
        return self._model

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        self._last_request_id = request.request_id
        if not request.artifacts:
            return []
        model = self._ensure_model()
        if self._pool is not None:
            self._pool.wake()
        # CuMem's model-building scope is one-shot. Execution uses the normal
        # allocator; park_memory's physical baseline gate rejects any lazy
        # long-lived CUDA allocation that survives scoring.
        return score_artifacts_with_model(
            model,
            request,
            worker_id="local",
            reward_model_version=str(
                self._worker_config.get("reward_model_version", ""),
            ),
        )

    async def shutdown(self) -> None:
        # A slept pool holds pinned host buffers for its pages; wake before
        # dropping the model so freeing the tensors actually returns the
        # pool's memory instead of leaking offloaded copies.
        pool = self._pool
        if pool is not None:
            pool.wake()
        self._model = None
        if pool is not None:
            pool.close()
        # Dedicated CUDA rewards use torch's caching allocator rather than a
        # CuMem pool. Dropping the model alone leaves those physical pages
        # reserved in this long-lived driver process, so terminal cleanup must
        # release the configured device cache for every runtime. The shared
        # path additionally proves the release against its pre-load baseline.
        device = str(self._worker_config.get("device", ""))
        release_cuda_memory_for_parking(device)
        if pool is not None:
            baseline_bytes = self._preload_gpu_used_bytes
            if baseline_bytes is None:
                raise RuntimeError("reward runtime has no pre-load shutdown baseline")
            # A failure retains the pool/baseline so terminal cleanup can retry
            # cache release; the trainer remains parked until this succeeds.
            validate_reward_parking_residual(
                residual_bytes=gpu_used_bytes(device),
                baseline_bytes=baseline_bytes,
                limit_bytes=self._parking_residual_bytes_limit,
                context="reward memory release during shutdown",
            )
        self._pool = None
        self._last_request_id = None
        self._preload_gpu_used_bytes = None


def build_reward_runtime(
    worker_config: Mapping[str, Any] | None = None,
    *,
    inference: Mapping[str, Any] | RewardInferenceConfig | None = None,
) -> RewardInferenceRuntime:
    """Build the runtime selected by the typed inference deployment config."""

    cfg = dict(worker_config or {})
    if "service_url" in cfg:
        raise ValueError(
            "worker_config.service_url was removed; configure "
            "reward.kwargs.<component>.inference.kind=http and inference.endpoint",
        )
    deployment = parse_reward_inference_config(
        inference,
        context="reward inference",
    )
    if deployment.kind == "in_process":
        return InProcessRewardRuntime(cfg)
    if cfg:
        raise ValueError(
            "HTTP reward runtime cannot consume local worker_config; model and "
            "device configuration belong to the external service",
        )
    from vrl.rewards.service.client import HttpRewardRuntime

    return HttpRewardRuntime(deployment)


__all__ = [
    "InProcessRewardRuntime",
    "build_reward_runtime",
]
