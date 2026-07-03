"""Reward inference runtime wiring and in-process transport."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Literal

from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
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
    ``sleep_offload``): the model is BUILT inside a per-runtime tagged pool,
    so ``sleep`` releases the physical pages (weights copied to pinned host
    RAM) while virtual addresses stay mapped, and ``wake`` remaps without
    cudaMalloc — no fragmentation, no per-model ``.to()`` needed, works for
    any reward model. Probe-measured ~6x faster per score cycle than a naive
    ``.to()`` round trip at Kling scale (14GB).
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
        if not request.artifacts:
            return []
        model = self._ensure_model()
        if self._pool is not None:
            self._pool.wake()
        try:
            # The result/artifact mismatch guard lives at the RewardFunction
            # seam (one enforcement point for every runtime), not here.
            return score_artifacts_with_model(
                model,
                request,
                worker_id="local",
                reward_model_version=str(self._worker_config.get("reward_model_version", "")),
            )
        finally:
            if self._pool is not None:
                self._pool.sleep()

    async def shutdown(self) -> None:
        # A slept pool holds pinned host buffers for its pages; wake before
        # dropping the model so freeing the tensors actually returns the
        # pool's memory instead of leaking offloaded copies.
        if self._pool is not None:
            self._pool.wake()
        self._pool = None
        self._model = None


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
        "for a heavyweight model on a shared GPU set the reward's "
        "sleep_offload=true instead.",
    )


__all__ = [
    "LocalRewardRuntime",
    "make_reward_runtime",
]
