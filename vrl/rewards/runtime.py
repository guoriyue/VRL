"""Reward inference runtime wiring and in-process transport."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Mapping
from typing import Any, Literal

from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
    RewardMemoryReleaseProof,
    score_artifacts_with_model,
)
from vrl.utils.cuda_memory import CumemPool


class LocalRewardRuntime:
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
        self._parking_baseline_gpu_used_bytes: int | None = None
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
        self._release_cuda_memory_for_parking()
        baseline_bytes = self._parking_baseline_gpu_used_bytes
        if baseline_bytes is None:
            raise RuntimeError("reward runtime has no pre-load GPU parking baseline")
        proof = RewardMemoryReleaseProof(
            request_id=request_id,
            released=True,
            baseline_gpu_used_bytes=baseline_bytes,
            residual_gpu_used_bytes=self._gpu_used_bytes(),
            residual_bytes_limit=self._parking_residual_bytes_limit,
        )
        proof.validate(request_id=request_id)
        return proof

    def _ensure_model(self) -> Any:
        if self._model is None:
            factory_path = str(self._worker_config.get("model_factory", "")).strip()
            if not factory_path:
                raise ValueError(
                    "LocalRewardRuntime requires worker_config.model_factory "
                    "(import path to a RewardModel factory) or an explicit model",
                )
            module_path, attr = factory_path.split(":", 1)
            factory = getattr(importlib.import_module(module_path), attr)
            if self._sleep_offload:
                # Build inside the pool so every CUDA allocation the factory
                # makes (from_pretrained, .to(device), buffers) is tagged and
                # sleep/wake can release/restore it wholesale.
                self._parking_baseline_gpu_used_bytes = self._gpu_used_bytes()
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
        self._release_cuda_memory_for_parking()
        if pool is not None:
            baseline_bytes = self._parking_baseline_gpu_used_bytes
            if baseline_bytes is None:
                raise RuntimeError("reward runtime has no pre-load shutdown baseline")
            residual_bytes = self._gpu_used_bytes()
            allowed = baseline_bytes + self._parking_residual_bytes_limit
            if residual_bytes > allowed:
                # Retain the pool/baseline so terminal cleanup can retry cache
                # release; the trainer remains parked until this method succeeds.
                raise RuntimeError(
                    "incomplete reward memory release during shutdown: "
                    f"residual={residual_bytes} baseline={baseline_bytes} "
                    f"limit={self._parking_residual_bytes_limit}",
                )
        self._pool = None
        self._last_request_id = None
        self._parking_baseline_gpu_used_bytes = None

    def _gpu_used_bytes(self) -> int:
        """Physical bytes in use on the configured reward CUDA device."""

        device = str(self._worker_config.get("device", ""))
        if not device.startswith("cuda"):
            return 0
        import torch

        if not torch.cuda.is_available():
            return 0
        target = torch.device(device)
        torch.cuda.synchronize(target)
        free_bytes, total_bytes = torch.cuda.mem_get_info(target)
        return int(total_bytes - free_bytes)

    def _release_cuda_memory_for_parking(self) -> None:
        """Release default-allocator pages before publishing the reward proof."""

        gc.collect()
        device = str(self._worker_config.get("device", ""))
        if not device.startswith("cuda"):
            return
        import torch

        if not torch.cuda.is_available():
            return
        target = torch.device(device)
        with torch.cuda.device(target):
            torch.cuda.synchronize(target)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize(target)


def make_reward_runtime(
    execution: Literal["inline"],
    *,
    model_factory: str,
    worker_config: Mapping[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build the in-process reward runtime for a given model factory."""

    worker_cfg = {**dict(worker_config or {}), "model_factory": str(model_factory)}
    runtime = str(execution or "inline")
    if runtime == "inline":
        return LocalRewardRuntime(worker_cfg)
    raise ValueError(
        f"unsupported execution={execution!r}: the Ray reward pool was removed "
        "and rewards score in-process. Drop the key (inline is the default); "
        "shared-GPU parking is derived from distributed resource topology.",
    )


__all__ = [
    "LocalRewardRuntime",
    "make_reward_runtime",
]
