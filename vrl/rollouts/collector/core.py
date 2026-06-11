"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from vrl.utils.config import cfg_get


@dataclass(slots=True)
class UnscoredRollout:
    """A generated-but-unscored prompt group awaiting deferred reward scoring."""

    output: GenerationOutput
    collector_request: CollectorRequest
    profile: bool = False
    phases: dict[str, float] = field(default_factory=dict)


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
        unscored = await self.collect_unscored(prompts, **kwargs)
        return (await self.score_rollouts([unscored]))[0]

    async def collect_unscored(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> UnscoredRollout:
        """Generate one prompt group without scoring it.

        Deferred-scoring half of collect(): the generation runtime stays
        resident so several groups can be generated back to back; scoring (and
        the rollout release shared-GPU reward runs need before it) happens in
        score_rollouts().
        """

        group_size = int(kwargs.get("group_size", self.default_group_size))
        collector_request = self.request_builder.build(prompts, group_size, dict(kwargs))

        profile = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        phase_t = _sync_time() if profile else None

        output = await self.runtime.generate(collector_request.request)
        if output.error:
            raise RuntimeError(
                f"{self.family}/{self.task} generation failed "
                f"(request_id={collector_request.request.request_id}): {output.error}",
            )

        unscored = UnscoredRollout(
            output=output,
            collector_request=collector_request,
            profile=profile,
        )
        if profile and phase_t is not None:
            unscored.phases["collect.engine_generate"] = _sync_time() - phase_t
        return unscored

    async def score_rollouts(self, unscored: list[UnscoredRollout]) -> list[RolloutBatch]:
        """Score unscored groups through one reward call and build their batches.

        Rollout prompts/metadata are per-sample, so all groups score in a
        single reward_scorer.score_many call — model-backed rewards with
        release_after_score pay one actor cold start per call instead of one
        per group. Batches return in input order.
        """

        if not unscored:
            return []
        if self._should_release_runtime_before_reward_model():
            # Shared single-GPU reward runs must drop rollout actors before reward
            # model actors can reserve the same GPU.
            await self.release_runtime_memory()

        contexts = []
        builders = []
        for rollout in unscored:
            context = RolloutBatchBuildContext(
                metadata=dict(rollout.collector_request.metadata),
                device=getattr(self.model, "device", None),
                kl_reward=float(cfg_get(self.config, "kl_reward", 0.0)),
                reward_view_name=_reward_view_name(self.config),
                trajectory_storage_policy=trajectory_storage_policy_from_cfg(
                    cfg_get(self.config, "trajectory_storage", None),
                ),
                reward_artifact_policy=reward_artifact_policy_from_cfg(
                    cfg_get(self.config, "reward_artifact", None),
                ),
            )
            contexts.append(context)
            builders.append(TrajectoryRolloutBatchBuilder(rollout.output, context))

        profile = any(rollout.profile for rollout in unscored)
        phase_t = _sync_time() if profile else None
        with _record_function("collector.reward_score"):
            rewards = await self.reward_scorer.score_many(
                [
                    builder.reward_scoring_input(rollout.collector_request.metadata)
                    for builder, rollout in zip(builders, unscored, strict=True)
                ],
            )
        reward_score_s = _sync_time() - phase_t if phase_t is not None else None

        batches: list[RolloutBatch] = []
        for builder, context, rollout, group_rewards in zip(
            builders, contexts, unscored, rewards, strict=True,
        ):
            batch = builder.build(group_rewards)
            release_reward_artifact_if_needed(batch, context.reward_artifact_policy)
            release_reward_artifact_if_needed(rollout.output, context.reward_artifact_policy)
            if rollout.profile and reward_score_s is not None:
                rollout.phases["collect.reward_score"] = reward_score_s
                self.last_collect_phases.clear()
                self.last_collect_phases.update(rollout.phases)
            batches.append(batch)
        return batches

    def _should_release_runtime_before_reward_model(self) -> bool:
        # Ask the runtime (GenerationRuntime protocol) instead of probing its
        # config internals; the release decision is the runtime's own knowledge.
        runtime = self._runtime
        if runtime is None:
            return False
        return bool(runtime.should_release_memory_before_reward())


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


def _reward_view_name(config: Any) -> str | None:
    for name in ("reward_view", "reward_view_name"):
        value = cfg_get(config, name, None)
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
    "UnscoredRollout",
    "build_rollout_collector",
]
