"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.ray.resources import RayLifecyclePlan

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
from vrl.utils.profiling import record_function


@dataclass(slots=True)
class UnscoredRollout:
    """A generated-but-unscored prompt group awaiting deferred reward scoring."""

    output: GenerationOutput
    collector_request: CollectorRequest
    profile: bool = False
    phases: dict[str, float] = field(default_factory=dict)
    reward_timing_ms: dict[str, float] = field(default_factory=dict)


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
        runtime: GenerationRuntime | None = None,
        lifecycle: RayLifecyclePlan | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.family = family
        self.task = task
        self.request_builder = request_builder
        self.reward_scorer = reward_scorer
        self._runtime = runtime
        # Topology-derived release policy (vrl/ray/resources.py). None means no
        # shared GPU, so rollout never releases before reward. Read here instead
        # of asking the runtime, which is now just transport.
        self._lifecycle = lifecycle

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
        release = getattr(self._runtime, "release", None)
        if release is not None:
            await release()

    async def collect(
        self,
        prompts: list[str],
        *,
        group_size: int,
        **kwargs: Any,
    ) -> RolloutBatch:
        unscored = await self.collect_unscored(prompts, group_size=group_size, **kwargs)
        return (await self.score_rollouts([unscored]))[0]

    async def collect_unscored(
        self,
        prompts: list[str],
        *,
        group_size: int,
        **kwargs: Any,
    ) -> UnscoredRollout:
        """Generate one prompt group without scoring it.

        Deferred-scoring half of collect(): the generation runtime stays
        resident so several groups can be generated back to back; scoring (and
        the rollout release shared-GPU reward runs need before it) happens in
        score_rollouts().
        """

        collector_request = self.request_builder.build(prompts, int(group_size), dict(kwargs))

        profile = os.environ.get("VRL_PROFILE") == "1"
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
                kl_reward_coef=float(cfg_get(self.config, "kl_reward_coef", 0.0)),
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
        with record_function("collector.reward_score"):
            rewards = await self.reward_scorer.score_many(
                [
                    builder.reward_scoring_input(rollout.collector_request.metadata)
                    for builder, rollout in zip(builders, unscored, strict=True)
                ],
            )
        reward_timing_ms = dict(
            getattr(self.reward_scorer, "last_reward_timing_ms", {}) or {},
        )
        if reward_timing_ms:
            unscored[0].reward_timing_ms.update(reward_timing_ms)
        reward_score_s = _sync_time() - phase_t if phase_t is not None else None

        build_t = _sync_time() if profile else None
        batches: list[RolloutBatch] = []
        for builder, context, rollout, group_rewards in zip(
            builders, contexts, unscored, rewards, strict=True,
        ):
            batch = builder.build(group_rewards)
            release_reward_artifact_if_needed(batch, context.reward_artifact_policy)
            release_reward_artifact_if_needed(rollout.output, context.reward_artifact_policy)
            batches.append(batch)
        if reward_score_s is not None and build_t is not None:
            # One score_many call and one build pass cover every group, so the
            # call-level timings live on the first group only: a caller summing
            # phases over groups must not multiply the same wall time. The
            # phases stay on the rollouts (caller-owned) so concurrent collects
            # never share mutable collector state.
            unscored[0].phases["collect.reward_score"] = reward_score_s
            unscored[0].phases["collect.batch_build"] = _sync_time() - build_t
        return batches

    def _should_release_runtime_before_reward_model(self) -> bool:
        # The release decision is derived once from GPU topology into the
        # lifecycle plan (vrl/ray/resources.py), not re-decided per call by the
        # runtime. None plan = no shared GPU = never release before reward.
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_rollout_before_reward


def build_rollout_collector(
    family: str,
    *,
    model: Any | None,
    reward_fn: Any | None,
    config: RolloutConfig | None = None,
    runtime: GenerationRuntime | None = None,
    lifecycle: RayLifecyclePlan | None = None,
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
        runtime=runtime,
        lifecycle=lifecycle,
    )


def _reward_view_name(config: Any) -> str | None:
    value = cfg_get(config, "reward_view", None)
    return str(value) if value else None


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
