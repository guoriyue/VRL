"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

from vrl.generation import GenerationOutput, GenerationRuntime
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.artifacts import (
    release_reward_artifact_if_needed,
    reward_artifact_policy_from_cfg,
)
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.config import RolloutConfig
from vrl.rollouts.collector.requests import (
    CollectorRequest,
    GenerationRequestBuilder,
)
from vrl.rollouts.collector.rewards import RewardScorer
from vrl.rollouts.families import get_rollout_family_entry
from vrl.trajectory import trajectory_storage_policy_from_cfg


class RolloutCollector:
    """Generic collector: request -> generation runtime -> reward -> trainer batch."""

    def __init__(
        self,
        *,
        model: Any | None,
        config: Any,
        family: str,
        task: str,
        request_builder: Any,
        reward_scorer: RewardScorer,
        default_group_size: int = 1,
        runtime: GenerationRuntime | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.family = family
        self.task = task
        self.request_builder = request_builder
        self.reward_scorer = reward_scorer
        self.default_group_size = max(1, int(default_group_size))
        self._runtime = runtime
        self.last_collect_phases: dict[str, float] = {}

    def set_runtime(self, runtime: GenerationRuntime) -> None:
        if not callable(getattr(runtime, "generate", None)):
            raise TypeError(
                "generation runtime must implement async generate(request) -> GenerationOutput",
            )
        self._runtime = runtime

    @property
    def runtime(self) -> GenerationRuntime:
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
        collector_request = self.request_builder.build(prompts, group_size, dict(kwargs))

        profile = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        phases: dict[str, float] = {}
        phase_t = _sync_time() if profile else None

        output = await self.runtime.generate(collector_request.request)
        if output.error:
            raise RuntimeError(
                f"{self.family}/{self.task} generation failed "
                f"(request_id={collector_request.request.request_id}): {output.error}",
            )

        if self._should_release_runtime_before_reward_model():
            # Shared single-GPU reward runs must drop rollout actors before reward
            # model actors can reserve the same GPU.
            await self.release_runtime_memory()

        if profile and phase_t is not None:
            now = _sync_time()
            phases["collect.engine_generate"] = now - phase_t
            phase_t = now

        batch = await self._output_batch_to_rollout_batch(
            output,
            collector_request=collector_request,
            phases=phases if profile else None,
            phase_t=phase_t,
        )

        if profile:
            self.last_collect_phases.clear()
            self.last_collect_phases.update(phases)

        return batch

    async def _output_batch_to_rollout_batch(
        self,
        output: GenerationOutput,
        *,
        collector_request: CollectorRequest,
        phases: dict[str, float] | None = None,
        phase_t: float | None = None,
    ) -> RolloutBatch:
        context = RolloutBatchBuildContext(
            metadata=dict(collector_request.metadata),
            device=_device_from_model(self.model),
            kl_reward=float(_config_get(self.config, "kl_reward", 0.0)),
            rescale_to_unit=bool(_config_get(self.config, "rescale_to_unit", False)),
            reward_view_name=_reward_view_name(self.config),
            trajectory_storage_policy=trajectory_storage_policy_from_cfg(
                _config_get(self.config, "trajectory_storage", None),
            ),
            reward_artifact_policy=reward_artifact_policy_from_cfg(
                _config_get(self.config, "reward_artifact", None),
            ),
        )
        batch_builder = TrajectoryRolloutBatchBuilder(output, context)

        with _record_function("collector.reward_score"):
            rewards = await self.reward_scorer.score(
                batch_builder.reward_scoring_input(collector_request.metadata),
            )

        if phases is not None and phase_t is not None:
            phases["collect.reward_score"] = _sync_time() - phase_t

        batch = batch_builder.build(rewards)
        release_reward_artifact_if_needed(batch, context.reward_artifact_policy)
        release_reward_artifact_if_needed(output, context.reward_artifact_policy)
        return batch

    def _should_release_runtime_before_reward_model(self) -> bool:
        runtime = self._runtime
        config = getattr(runtime, "config", None)
        if not bool(getattr(config, "release_before_reward_model", False)):
            return False
        resources = getattr(config, "resources", None)
        return bool(getattr(resources, "reward_shared_with_rollout", False))


def build_rollout_collector(
    family: str,
    *,
    model: Any | None,
    reward_fn: Any | None,
    config: RolloutConfig | None = None,
    runtime: GenerationRuntime | None = None,
) -> RolloutCollector:
    """Build a rollout collector from the canonical family registry."""

    entry = get_rollout_family_entry(family)
    if config is None:
        raise ValueError(
            f"{entry.family} collector requires resolved rollout config; "
            "build them from YAML before constructing the collector",
        )
    collector = entry.collector
    if collector.request_prefix is None:
        raise ValueError(f"{entry.family} collector registry entry is incomplete")

    return RolloutCollector(
        model=model,
        config=config,
        family=entry.family,
        task=entry.task,
        request_builder=GenerationRequestBuilder(
            family=entry.family,
            task=entry.task,
            request_prefix=collector.request_prefix,
            config=config,
            return_artifacts=collector.return_artifacts,
            default_task_type=collector.default_task_type,
            metadata_key=collector.metadata_key,
        ),
        reward_scorer=RewardScorer(reward_fn),
        default_group_size=(
            1 if collector.kind == "diffusion" else int(config.require("n_samples_per_prompt"))
        ),
        runtime=runtime,
    )


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
        from vrl.utils.profiling import record_function
    except ImportError:
        return nullcontext()
    return record_function(name)


def _sync_time() -> float:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


__all__ = [
    "RolloutCollector",
    "build_rollout_collector",
]
