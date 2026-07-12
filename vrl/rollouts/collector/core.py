"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vrl.ray.resources import RayLifecyclePlan

from vrl.generation import GenerationOutput, GenerationRuntime
from vrl.rewards.base import RewardCleanupError
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
        config: Any,
        family: str,
        task: str,
        request_builder: Any,
        reward_scorer: RewardScorer,
        runtime: GenerationRuntime | None = None,
        lifecycle: RayLifecyclePlan | None = None,
    ) -> None:
        self.config = config
        self.family = family
        self.task = task
        self.request_builder = request_builder
        self.reward_scorer = reward_scorer
        self._runtime: GenerationRuntime | None = None
        if runtime is not None:
            self.set_runtime(runtime)
        # Topology-derived handoff policy (vrl/ray/resources.py). None means no
        # shared GPU, so rollout never offloads before reward. Read here instead
        # of asking the runtime, which is now just transport.
        self._lifecycle = lifecycle
        self._reward_phase_started = False
        self._reward_shutdown_complete = False

    def set_runtime(self, runtime: GenerationRuntime) -> None:
        if not isinstance(runtime, GenerationRuntime):
            raise TypeError(
                "generation runtime must implement the complete GenerationRuntime "
                "protocol (activate/generate/offload/shutdown/topology)",
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
        """Release generation before waking/destroying the reward owner."""

        errors: list[BaseException] = []
        runtime = self._runtime
        if runtime is not None:
            try:
                await runtime.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._runtime = None
        if errors:
            # A slept reward pool must not remap pages while rollout ownership is
            # unknown. Retain it asleep and let the next shutdown retry finish
            # generation first.
            raise errors[0]
        if not self._reward_shutdown_complete:
            try:
                await self.reward_scorer.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._reward_shutdown_complete = True
        if errors:
            raise errors[0]

    async def activate_runtime(self) -> None:
        await self.runtime.activate()

    async def offload_runtime_memory(self) -> None:
        errors: list[BaseException] = []
        try:
            await self._offload_runtime_transport()
        except BaseException as error:
            errors.append(error)
        if self._requires_reward_memory_release() and self._reward_phase_started:
            # Phase-final gate: actively invoke the idempotent park operation.
            # This retries a first sleep failure from score_many and never lets
            # a cached proof from an earlier request authorize trainer restore.
            try:
                await self.reward_scorer.park_memory(
                    required=True,
                )
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RewardCleanupError(
                "rollout and reward memory parking both failed",
                errors,
            )

    async def _offload_runtime_transport(self) -> None:
        await self.runtime.offload()

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
        the rollout offload shared-GPU reward runs need before it) happens in
        score_rollouts().
        """

        collector_request = self.request_builder.build(prompts, int(group_size), dict(kwargs))
        # A new generation phase has not activated reward memory yet. This also
        # prevents a previous iteration's proof from authorizing a later handoff.
        self._reward_phase_started = False

        profile = os.environ.get("VRL_PROFILE") == "1"
        phase_t = _wall_time() if profile else None

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
            unscored.phases["collect.engine_generate"] = _wall_time() - phase_t
        return unscored

    async def score_rollouts(self, unscored: list[UnscoredRollout]) -> list[RolloutBatch]:
        """Score unscored groups through one reward call and build their batches.

        Rollout prompts/metadata are per-sample, so all groups score in a
        single reward_scorer.score_many call — model-backed rewards with
        phase parking pay one model activation per call instead of one per
        group. Batches return in input order.
        """

        if not unscored:
            return []
        if self._should_release_runtime_before_reward_model():
            # Shared single-GPU reward runs park rollout model memory before the
            # in-process reward model takes over the physical GPU.
            await self._offload_runtime_transport()

        contexts = []
        builders = []
        for rollout in unscored:
            context = RolloutBatchBuildContext(
                metadata=dict(rollout.collector_request.metadata),
                # Collector/continuous-owner output is host-owned. The trainer
                # moves the completed batch to its device later; creating reward
                # tensors here on the trainer GPU would race backward.
                device="cpu",
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
        phase_t = _wall_time() if profile else None
        require_reward_release = self._requires_reward_memory_release()
        self._reward_phase_started = require_reward_release
        reward_inputs = [
            builder.reward_scoring_input(rollout.collector_request.metadata)
            for builder, rollout in zip(builders, unscored, strict=True)
        ]
        with record_function("collector.reward_score"):
            score_result = await self.reward_scorer.score_many(
                reward_inputs,
                require_memory_release=require_reward_release,
            )
        reward_timing_ms = dict(score_result.timing_ms)
        if reward_timing_ms:
            unscored[0].reward_timing_ms.update(reward_timing_ms)
        reward_score_s = _wall_time() - phase_t if phase_t is not None else None

        build_t = _wall_time() if profile else None
        batches: list[RolloutBatch] = []
        component_offset = 0
        for builder, context, rollout, group_rewards in zip(
            builders,
            contexts,
            unscored,
            score_result.scores,
            strict=True,
        ):
            batch = builder.build(group_rewards)
            component_stop = component_offset + int(group_rewards.shape[0])
            if score_result.components:
                # Raw component values travel with their exact rollout rows.
                # A continuous producer may already be scoring a future group
                # when this batch is consumed, so trainer metrics must never
                # read a mutable last-result cache from the shared reward model.
                batch.extras["reward_components"] = {
                    name: torch.tensor(
                        values[component_offset:component_stop],
                        dtype=torch.float32,
                    )
                    for name, values in score_result.components.items()
                }
            component_offset = component_stop
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
            unscored[0].phases["collect.batch_build"] = _wall_time() - build_t
        return batches

    def _should_release_runtime_before_reward_model(self) -> bool:
        # The release decision is derived once from GPU topology into the
        # lifecycle plan (vrl/ray/resources.py), not re-decided per call by the
        # runtime. None plan = no shared GPU = never release before reward.
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_rollout_before_reward

    def _requires_reward_memory_release(self) -> bool:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_reward_after_score

    @property
    def requires_runtime_offload_before_reward(self) -> bool:
        """Whether scoring introduces a mid-iteration GPU handoff."""

        return self._should_release_runtime_before_reward_model()

    @property
    def requires_driver_model_offload_for_reward(self) -> bool:
        """Whether reward scoring borrows the trainer's in-process GPU."""

        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_trainer_before_reward


def build_rollout_collector(
    family: str,
    *,
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


def _wall_time() -> float:
    """Owner-safe timing that never synchronizes the trainer's CUDA device."""

    return time.perf_counter()


__all__ = [
    "RolloutCollector",
    "UnscoredRollout",
    "build_rollout_collector",
]
