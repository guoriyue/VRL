"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

from vrl.engine import (
    OutputBatch,
    RolloutBackend,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.requests import RolloutRequestBuilder, RolloutRequestPlan
from vrl.rollouts.collector.rewards import RewardScorer


class RolloutCollector:
    """Generic collector: request -> generation runtime -> reward -> trainer batch."""

    def __init__(
        self,
        *,
        model: Any | None,
        config: Any,
        family: str,
        task: str,
        request_builder: RolloutRequestBuilder,
        reward_scorer: RewardScorer,
        default_group_size: int = 1,
        runtime: RolloutBackend | None = None,
        phase_sink: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.family = family
        self.task = task
        self.request_builder = request_builder
        self.reward_scorer = reward_scorer
        self.default_group_size = max(1, int(default_group_size))
        self._runtime = runtime
        self.phase_sink = phase_sink

    def set_runtime(self, runtime: RolloutBackend) -> None:
        if not callable(getattr(runtime, "generate", None)):
            raise TypeError(
                "rollout runtime must implement async generate(request) -> OutputBatch",
            )
        self._runtime = runtime

    @property
    def runtime(self) -> RolloutBackend:
        if self._runtime is None:
            raise RuntimeError(
                "RolloutCollector runtime is not initialized; "
                "call set_runtime(...) before collect(...)",
            )
        return self._runtime

    async def shutdown(self) -> None:
        shutdown = getattr(self._runtime, "shutdown", None)
        if shutdown is not None:
            await shutdown()
        self._runtime = None

    async def release_runtime_memory(self) -> None:
        release = getattr(self._runtime, "release_memory", None)
        if release is not None:
            await release()

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> RolloutBatch:
        group_size = int(kwargs.get("group_size", self.default_group_size))
        plan = self.request_builder.build(prompts, group_size, dict(kwargs))

        profile = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        phases: dict[str, float] = {}
        phase_t = _sync_time() if profile else None

        output = await self.runtime.generate(plan.request)
        if output.error:
            raise RuntimeError(
                f"{self.family}/{self.task} generation failed "
                f"(request_id={plan.request.request_id}): {output.error}",
            )

        if profile and phase_t is not None:
            now = _sync_time()
            phases["collect.engine_generate"] = now - phase_t
            phase_t = now

        batch = await self._output_batch_to_rollout_batch(
            output,
            request_plan=plan,
            phases=phases if profile else None,
            phase_t=phase_t,
        )

        if profile and self.phase_sink is not None:
            self.phase_sink.clear()
            self.phase_sink.update(phases)

        return batch

    async def _output_batch_to_rollout_batch(
        self,
        output: OutputBatch,
        *,
        request_plan: RolloutRequestPlan,
        phases: dict[str, float] | None = None,
        phase_t: float | None = None,
    ) -> RolloutBatch:
        context = RolloutBatchBuildContext(
            metadata=dict(request_plan.pack_metadata),
            device=_device_from_model(self.model),
            kl_reward=float(_config_get(self.config, "kl_reward", 0.0)),
            rescale_to_unit=bool(_config_get(self.config, "rescale_to_unit", False)),
            reward_view_name=_reward_view_name(self.config),
        )
        batch_builder = TrajectoryRolloutBatchBuilder(output, context)

        with _record_function("collector.reward_score"):
            rewards = await self.reward_scorer.score(
                batch_builder.reward_scoring_input(request_plan.reward_metadata),
            )

        if phases is not None and phase_t is not None:
            phases["collect.reward_score"] = _sync_time() - phase_t

        return batch_builder.build(rewards)


def _device_from_model(model: Any | None) -> Any | None:
    return getattr(model, "device", None)


def _config_get(config: Any, name: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _reward_view_name(config: Any) -> str | None:
    for name in ("reward_view", "reward_view_name"):
        value = _config_get(config, name, None)
        if value:
            return str(value)
    return None


def _record_function(name: str) -> Any:
    try:
        from vrl.trainers.profiling import record_function
    except ImportError:
        return nullcontext()
    return record_function(name)


def _sync_time() -> float:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


__all__ = ["RolloutCollector"]
