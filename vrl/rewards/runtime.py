"""Reward inference runtime wiring and in-process transport."""

from __future__ import annotations

import importlib
import traceback
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


def _build_prepared_model_in_pool(
    pool: CumemPool,
    factory: Any,
    worker_config: Mapping[str, Any],
) -> Any:
    """Build and prepare one reward model in an isolated CuMem-owned frame.

    The separate frame is a failure-ownership boundary: if lazy preparation
    allocates partial CUDA state and raises, the caller can clear this frame
    from the exception traceback before closing the pool. Keeping the candidate
    in the caller or traceback would leave its tensors live during cleanup.
    """

    with pool.building():
        model = factory(worker_config)
        prepare = getattr(model, "prepare_for_inference", None)
        if callable(prepare):
            # Lazy torch rewards must materialize their CUDA weights here, not
            # on the default allocator during first score.
            prepare()
        return model


def _build_prepared_model(factory: Any, worker_config: Mapping[str, Any]) -> Any:
    """Build one reward model and materialize any lazy accelerator weights."""

    model = factory(worker_config)
    prepare = getattr(model, "prepare_for_inference", None)
    if callable(prepare):
        prepare()
    return model


def _reward_model_mover(model: Any) -> Any:
    """Return the model's complete device-move hook or fail closed."""

    move = getattr(model, "move_to", None)
    if not callable(move):
        move = getattr(model, "to", None)
    if not callable(move):
        raise RuntimeError(
            f"reward model {type(model).__name__} cannot completely park on CPU: "
            "shared-GPU fallback requires move_to(device) or to(device)",
        )
    return move


class InProcessRewardRuntime:
    """``RewardInferenceRuntime`` that runs a ``RewardModel`` in this process.

    ``worker_config.sleep_offload`` opts a heavyweight model into the same
    sleep/wake semantics the rollout lease uses: the model holds no GPU memory
    between scores — the rollout/trainer own the card then — and comes back
    only for scoring; the caller's step ordering (rollout releases the GPU
    before rewards score) already guarantees the card is free at that point.
    Small in-memory rewards (CLIP-class) leave the knob off and stay resident.

    When vLLM's CuMemAllocator is present, the model is built under a backup tag,
    so process-wide ``sleep`` releases physical pages while preserving this
    reward's contents in pinned host RAM. The diffusers/reward environment keeps
    vLLM isolated because it pins a different Torch ABI, so the verified fallback
    moves a model with a complete ``move_to``/``to`` contract to CPU and back.
    Both backends publish the same physical-memory proof before trainer handoff.
    CuMem tags do not isolate sleep operations; the shared-topology preflight
    therefore permits at most one configured GPU reward component per process;
    zero-weight observation-only scorers still execute and therefore count.
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
                "(worker_config.model_factory) so it can capture the pre-load "
                "GPU baseline and validate a complete parking backend.",
            )
        self._model = model
        self._pool: CumemPool | None = None
        self._cpu_parked_device: str | None = None
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
        model = self._model
        if model is None:
            raise RuntimeError(
                "reward runtime cannot prove memory parking before its model is built",
            )
        pool = self._pool
        if pool is not None and not pool.asleep:
            # CumemPool marks itself asleep only after allocator.sleep returns.
            # A failure therefore leaves this branch retryable on the next call.
            pool.sleep()
        elif pool is None and self._cpu_parked_device is None:
            restore_device = str(self._worker_config.get("device", "")).strip()
            if not restore_device.startswith("cuda"):
                raise RuntimeError(
                    "reward CPU parking fallback requires a CUDA execution device, "
                    f"got {restore_device!r}",
                )
            move = _reward_model_mover(model)
            try:
                move("cpu")
            except BaseException as move_error:
                try:
                    move(restore_device)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "reward CPU parking and rollback both failed: "
                        f"move={move_error!r}; rollback={rollback_error!r}",
                    ) from rollback_error
                raise
            self._cpu_parked_device = restore_device
        self._release_cuda_memory_for_parking()
        baseline_bytes = self._preload_gpu_used_bytes
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
                    "InProcessRewardRuntime requires worker_config.model_factory "
                    "(import path to a RewardModel factory) or an explicit model",
                )
            module_path, attr = factory_path.split(":", 1)
            factory = getattr(importlib.import_module(module_path), attr)
            if self._sleep_offload:
                # Build inside the pool so every CUDA allocation the factory
                # makes (from_pretrained, .to(device), buffers) is tagged and
                # sleep/wake can release/restore it wholesale.
                self._preload_gpu_used_bytes = self._gpu_used_bytes()
                pool = CumemPool.try_create()
                if pool is not None:
                    try:
                        model = _build_prepared_model_in_pool(
                            pool,
                            factory,
                            self._worker_config,
                        )
                    except BaseException as load_error:
                        # Commit neither half of a failed model/pool build. Dropping
                        # traceback-held helper locals first lets terminal pool close
                        # release partial CUDA allocations before a future retry.
                        traceback.clear_frames(load_error.__traceback__)
                        try:
                            self._release_cuda_memory_for_parking()
                            pool.close()
                        except BaseException as cleanup_error:
                            self._preload_gpu_used_bytes = None
                            raise RuntimeError(
                                "reward model preparation and CuMem cleanup both failed: "
                                f"load={load_error!r}; cleanup={cleanup_error!r}",
                            ) from cleanup_error
                        self._preload_gpu_used_bytes = None
                        raise
                    self._pool = pool
                    self._model = model
                else:
                    model = None
                    try:
                        model = _build_prepared_model(factory, self._worker_config)
                        _reward_model_mover(model)
                    except BaseException as load_error:
                        traceback.clear_frames(load_error.__traceback__)
                        model = None
                        self._release_cuda_memory_for_parking()
                        self._preload_gpu_used_bytes = None
                        raise
                    self._model = model
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
        elif self._cpu_parked_device is not None:
            restore_device = self._cpu_parked_device
            _reward_model_mover(model)(restore_device)
            self._cpu_parked_device = None
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
        if self._sleep_offload and self._preload_gpu_used_bytes is not None:
            baseline_bytes = self._preload_gpu_used_bytes
            # A failure retains the pool/baseline so terminal cleanup can retry
            # cache release; the trainer remains parked until this succeeds.
            validate_reward_parking_residual(
                residual_bytes=self._gpu_used_bytes(),
                baseline_bytes=baseline_bytes,
                limit_bytes=self._parking_residual_bytes_limit,
                context="reward memory release during shutdown",
            )
        self._pool = None
        self._cpu_parked_device = None
        self._last_request_id = None
        self._preload_gpu_used_bytes = None

    # Instance-assignable test seams over the shared parking bookkeeping in
    # vrl.utils.cuda_memory; rewards measure their configured device only.
    def _gpu_used_bytes(self) -> int:
        return gpu_used_bytes(str(self._worker_config.get("device", "")))

    def _release_cuda_memory_for_parking(self) -> None:
        release_cuda_memory_for_parking(
            str(self._worker_config.get("device", "")),
        )


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
