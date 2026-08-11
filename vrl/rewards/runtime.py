"""Reward runtime implementations and inference transport wiring."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from vrl.config.reward_inference import (
    RewardInferenceConfig,
    parse_reward_inference_config,
)
from vrl.rewards.base import RewardCleanupError, RewardFunction
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
    score_artifacts_with_model,
)
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.config import import_from_path
from vrl.utils.cuda_memory import (
    CumemPool,
    gpu_used_bytes,
    release_cuda_memory_for_parking,
    validate_parking_residual,
)
from vrl.utils.lifecycle import RuntimeLifecycle, RuntimePhase


class RewardFunctionRuntime:
    """Expose a reward function through the collector-facing runtime contract."""

    def __init__(self, reward_function: RewardFunction | None) -> None:
        if reward_function is not None and not isinstance(reward_function, RewardFunction):
            raise TypeError("reward_function must be a RewardFunction or None")
        self._reward_function = reward_function
        self._operation_lock = asyncio.Lock()
        # Same terminal FSM as the generation runtime: RUNNING accepts work,
        # SHUTTING_DOWN closes admission (retryable teardown), TERMINATED is
        # published only after a successful shutdown.
        self.lifecycle = RuntimeLifecycle(owner="reward runtime")

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether scoring yields while every configured component executes."""

        reward_function = self._reward_function
        return bool(reward_function is not None and reward_function.scoring_is_nonblocking)

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work is isolated."""

        reward_function = self._reward_function
        return bool(
            reward_function is None or reward_function.external_accelerator_isolation_verified
        )

    async def preflight(self) -> None:
        """Validate the wrapped reward function before scoring begins."""

        async with self._operation_lock:
            self.lifecycle.require_running("preflight")
            reward_function = self._reward_function
            if reward_function is not None:
                await reward_function.preflight()

    async def score(
        self,
        samples: Sequence[RewardSample],
        *,
        require_memory_release: bool = False,
    ) -> RewardOutput:
        """Score ordered samples while serializing function and memory ownership."""

        async with self._operation_lock:
            self.lifecycle.require_running("score")
            normalized = tuple(samples)
            if not normalized:
                raise ValueError("reward runtime requires at least one sample")
            if not all(isinstance(sample, RewardSample) for sample in normalized):
                raise TypeError("reward runtime samples must contain RewardSample values")
            sample_ids = [sample.sample_id for sample in normalized]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("reward runtime sample_id values must be unique")
            reward_function = self._reward_function
            output: RewardOutput | None = None
            operation_error: BaseException | None = None
            try:
                if reward_function is None:
                    output = RewardOutput(scores=(0.0,) * len(normalized))
                else:
                    output = await reward_function.score_batch(normalized)
                if not isinstance(output, RewardOutput):
                    raise TypeError("reward function score_batch() must return RewardOutput")
                if len(output.scores) != len(normalized):
                    raise ValueError(
                        "reward function returned wrong number of scores: "
                        f"scores={len(output.scores)}, samples={len(normalized)}",
                    )
            except BaseException as error:
                operation_error = error

            parking_error: BaseException | None = None
            if require_memory_release:
                try:
                    await self._park_memory_locked(required=True)
                except BaseException as error:
                    parking_error = error
            if operation_error is not None and parking_error is not None:
                raise RewardCleanupError(
                    "reward scoring and memory parking both failed",
                    [operation_error, parking_error],
                )
            if operation_error is not None:
                raise operation_error
            if parking_error is not None:
                raise parking_error
            assert output is not None
            return output

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> None:
        """Actively park reward owners and enforce the configured gate."""

        async with self._operation_lock:
            self.lifecycle.require_running("park_memory")
            await self._park_memory_locked(required=required)

    async def _park_memory_locked(
        self,
        *,
        required: bool,
    ) -> None:
        reward_function = self._reward_function
        if reward_function is None:
            if required:
                raise RuntimeError(
                    "shared reward topology requires an active memory-parking owner, "
                    "but no reward function is configured",
                )
            return
        parked = await reward_function.park_memory()
        if not isinstance(parked, bool):
            raise TypeError("reward function park_memory() must return bool")
        if required and not parked:
            raise RuntimeError("reward function has no active memory-parking owner")

    async def shutdown(self) -> None:
        """Release the wrapped function exactly once after successful teardown.

        A failed teardown leaves the lifecycle SHUTTING_DOWN (admission stays
        closed, root cause retained) and the next call retries the release.
        """

        async with self._operation_lock:
            if self.lifecycle.phase is RuntimePhase.TERMINATED:
                return
            self.lifecycle.begin_shutdown()
            reward_function = self._reward_function
            if reward_function is not None:
                try:
                    await reward_function.shutdown()
                except BaseException as error:
                    self.lifecycle.fail(error)
                    raise
            self.lifecycle.finish_shutdown()


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
            prepare()
        return model


class InProcessRewardInferenceRuntime:
    """``RewardInferenceRuntime`` that runs a ``RewardModel`` in this process.

    ``worker_config.sleep_offload`` opts a heavyweight model into the same
    sleep/wake semantics the rollout lease uses: the model holds no GPU memory
    between scores — the rollout/trainer own the card then — and comes back
    only for scoring; the caller's step ordering (rollout releases the GPU
    before rewards score) already guarantees the card is free at that point.
    Small in-memory rewards (CLIP-class) leave the knob off and stay resident.

    Parking is CuMem-only. The model is built under a backup tag, so process-wide
    ``sleep`` releases physical pages while preserving this reward's contents in
    pinned host RAM; the runtime requires vLLM's CuMemAllocator and fails loud
    without it rather than degrading to a CPU round trip, which measured 6.2x
    slower per wake/score/sleep cycle and only appears to park.
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

    async def park_memory(self) -> None:
        """Park reward pages and validate the residual-memory gate; safe to retry."""

        if not self._sleep_offload:
            raise RuntimeError(
                "reward runtime was not configured for complete memory parking",
            )
        pool = self._pool
        if pool is None:
            raise RuntimeError(
                "reward runtime cannot prove memory parking before its "
                "CuMem-pooled model is built",
            )
        if not pool.asleep:
            # CumemPool marks itself asleep only after allocator.sleep returns.
            # A failure therefore leaves this branch retryable on the next call.
            pool.sleep()
        self._release_cuda_memory_for_parking()
        baseline_bytes = self._preload_gpu_used_bytes
        if baseline_bytes is None:
            raise RuntimeError("reward runtime has no pre-load GPU parking baseline")
        validate_parking_residual(
            residual_bytes=self._gpu_used_bytes(),
            baseline_bytes=baseline_bytes,
            limit_bytes=self._parking_residual_bytes_limit,
            context="reward memory parking",
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            factory_path = str(self._worker_config.get("model_factory", "")).strip()
            if not factory_path:
                raise ValueError(
                    "InProcessRewardInferenceRuntime requires worker_config.model_factory "
                    "(import path to a RewardModel factory) or an explicit model",
                )
            factory = import_from_path(factory_path)
            if self._sleep_offload:
                # Claim the pool before capturing the baseline so a box without
                # CuMem leaves no phantom baseline behind for shutdown's residual
                # check. get_instance() only does Python bookkeeping, so ordering
                # it first does not perturb the measurement.
                pool = CumemPool.require()
                self._preload_gpu_used_bytes = self._gpu_used_bytes()
                # Build inside the pool so every CUDA allocation the factory
                # makes (from_pretrained, .to(device), buffers) is tagged and
                # sleep/wake can release/restore it wholesale.
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
                self._model = factory(self._worker_config)
        return self._model

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
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
            validate_parking_residual(
                residual_bytes=self._gpu_used_bytes(),
                baseline_bytes=baseline_bytes,
                limit_bytes=self._parking_residual_bytes_limit,
                context="reward memory release during shutdown",
            )
        self._pool = None
        self._preload_gpu_used_bytes = None

    # Instance-assignable test seams over the shared parking bookkeeping in
    # vrl.utils.cuda_memory; rewards measure their configured device only.
    def _gpu_used_bytes(self) -> int:
        return gpu_used_bytes(str(self._worker_config.get("device", "")))

    def _release_cuda_memory_for_parking(self) -> None:
        release_cuda_memory_for_parking(
            str(self._worker_config.get("device", "")),
        )


def build_reward_inference_runtime(
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
        return InProcessRewardInferenceRuntime(cfg)
    if cfg:
        raise ValueError(
            "HTTP reward runtime cannot consume local worker_config; model and "
            "device configuration belong to the external service",
        )
    from vrl.rewards.service.client import HttpRewardInferenceRuntime

    return HttpRewardInferenceRuntime(deployment)


__all__ = [
    "InProcessRewardInferenceRuntime",
    "RewardFunctionRuntime",
    "build_reward_inference_runtime",
]
