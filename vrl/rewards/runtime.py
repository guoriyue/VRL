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


class LocalRewardRuntime:
    """``RewardInferenceRuntime`` that runs a ``RewardModel`` in this process.

    ``worker_config.sleep_offload`` opts a heavyweight model into the same
    sleep/wake semantics the rollout lease uses (generation/execution/worker.py):
    the model is parked on CPU between scores so it holds no GPU memory while the
    rollout/trainer own the card, and moves back only for scoring — the caller's
    step ordering (rollout releases the GPU before rewards score) already
    guarantees the card is free at that point. The model must expose ``.to()``;
    small in-memory rewards (CLIP-class) leave the knob off and stay resident.
    """

    def __init__(
        self,
        worker_config: Mapping[str, Any] | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self._worker_config = dict(worker_config or {})
        self._model = model
        self._sleep_offload = bool(self._worker_config.get("sleep_offload", False))
        self._wake_device: str | None = None

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
            self._model = factory(self._worker_config)
        return self._model

    def _sleep(self) -> None:
        """Park the model on CPU and return its GPU memory to the pool."""

        model = self._model
        move = getattr(model, "to", None)
        if not callable(move):
            raise TypeError(
                "reward worker_config.sleep_offload=true requires the reward "
                f"model to implement .to(device); {type(model).__name__} does not",
            )
        # Capture the scoring device before the move so a model whose ``device``
        # attribute tracks ``.to`` still wakes back onto the right GPU.
        if self._wake_device is None:
            self._wake_device = str(getattr(model, "device", "cuda"))
        move("cpu")
        from vrl.utils.cuda_memory import release_cuda_memory

        release_cuda_memory(gc_collect=True, ipc_collect=True)

    def _wake(self, model: Any) -> None:
        """Restore a parked model onto its scoring device (no rebuild)."""

        if self._wake_device is not None:
            model.to(self._wake_device)

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        if not request.artifacts:
            return []
        model = self._ensure_model()
        if self._sleep_offload:
            self._wake(model)
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
            if self._sleep_offload:
                self._sleep()

    async def shutdown(self) -> None:
        self._model = None
        self._wake_device = None


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
