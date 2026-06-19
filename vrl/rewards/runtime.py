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
    validate_reward_results,
)


class LocalRewardRuntime:
    """``RewardInferenceRuntime`` that runs a ``RewardModel`` in this process."""

    def __init__(
        self,
        worker_config: Mapping[str, Any] | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self._worker_config = dict(worker_config or {})
        self._model = model

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

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        if not request.artifacts:
            return []
        model = self._ensure_model()
        results = score_artifacts_with_model(
            model,
            request,
            worker_id="local",
            reward_model_version=str(self._worker_config.get("reward_model_version", "")),
        )
        return validate_reward_results(request, results)

    async def shutdown(self) -> None:
        self._model = None


def make_reward_runtime(
    execution: Literal["inline", "pool"],
    *,
    model_factory: str,
    worker_config: Mapping[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build the local or ray reward runtime for a given model factory."""

    worker_cfg = {**dict(worker_config or {}), "model_factory": str(model_factory)}
    runtime = str(execution or "inline")
    if runtime == "inline":
        return LocalRewardRuntime(worker_cfg)
    if runtime == "pool":
        # Imported lazily so local-only usage never pulls in Ray.
        from vrl.rewards.ray.runtime import RayRewardRuntime

        # RayRewardRuntime expects the model config under a nested "worker_config"
        # key, plus optional resource knobs at the top level.
        ray_cfg = {**worker_cfg, "worker_config": worker_cfg}
        return RayRewardRuntime(ray_cfg)
    raise ValueError(
        f"unsupported execution={execution!r}; expected 'inline' or 'pool'",
    )


__all__ = [
    "LocalRewardRuntime",
    "make_reward_runtime",
]
