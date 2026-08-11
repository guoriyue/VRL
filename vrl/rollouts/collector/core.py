"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vrl.ray.resources import RayLifecyclePlan

from vrl.generation import GenerationOutput, GenerationRuntime
from vrl.models.families.registry import ModelFamilyEntry
from vrl.rewards import RewardRuntime
from vrl.rewards.base import RewardCleanupError
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import (
    CollectorRequest,
    GenerationRequestBuilder,
)
from vrl.utils.profiling import profile_range


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
        config: RolloutCollectorConfig,
        request_builder: GenerationRequestBuilder,
        reward_runtime: RewardRuntime,
        generation_runtime: GenerationRuntime | None = None,
        lifecycle: RayLifecyclePlan | None = None,
        trajectory_layout: str | None = None,
    ) -> None:
        self.config = config
        self.request_builder = request_builder
        if not isinstance(reward_runtime, RewardRuntime):
            raise TypeError(
                "reward runtime must implement the complete RewardRuntime protocol",
            )
        self.reward_runtime = reward_runtime
        # Resolved family PolicySemantics.trajectory_layout (source of truth for
        # multisegment routing), threaded into every RolloutBatchBuildContext.
        self._trajectory_layout = trajectory_layout
        self._generation_runtime: GenerationRuntime | None = None
        if generation_runtime is not None:
            self.set_generation_runtime(generation_runtime)
        # Topology-derived handoff policy (vrl/ray/resources.py). None means no
        # shared GPU, so rollout never offloads before reward. Read here instead
        # of asking the runtime, which is now just transport.
        self._lifecycle = lifecycle
        self._reward_phase_started = False
        self._reward_shutdown_complete = False

    def set_generation_runtime(self, runtime: GenerationRuntime) -> None:
        if not isinstance(runtime, GenerationRuntime):
            raise TypeError(
                "generation runtime must implement the complete GenerationRuntime "
                "protocol (activate/generate/offload/shutdown/topology)",
            )
        self._generation_runtime = runtime

    @property
    def generation_runtime(self) -> GenerationRuntime:
        if self._generation_runtime is None:
            raise RuntimeError(
                "RolloutCollector generation runtime is not initialized; "
                "call set_generation_runtime(...) before collect(...)",
            )
        return self._generation_runtime

    async def shutdown(self) -> None:
        """Release generation before waking/destroying the reward owner."""

        errors: list[BaseException] = []
        runtime = self._generation_runtime
        if runtime is not None:
            try:
                await runtime.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._generation_runtime = None
        if errors:
            # A slept reward pool must not remap pages while rollout ownership is
            # unknown. Retain it asleep and let the next shutdown retry finish
            # generation first.
            raise errors[0]
        if not self._reward_shutdown_complete:
            try:
                await self.reward_runtime.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._reward_shutdown_complete = True
        if errors:
            raise errors[0]

    async def activate_generation_runtime(self) -> None:
        await self.generation_runtime.activate()

    async def offload_generation_runtime_memory(self) -> None:
        errors: list[BaseException] = []
        try:
            await self.generation_runtime.offload()
        except BaseException as error:
            errors.append(error)
        if self._requires_reward_memory_release() and self._reward_phase_started:
            # Phase-final gate: actively invoke the idempotent park operation.
            # This retries a first sleep failure from score(); successful return
            # is the phase-final gate before trainer restore.
            try:
                await self.reward_runtime.park_memory(
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

    async def collect_unscored(
        self,
        inputs: list[Any],
        *,
        group_size: int,
        metadata: Mapping[str, Any] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
    ) -> UnscoredRollout:
        """Generate one group of ``GenerationInput`` (or bare prompt) conditioning.

        Generation half of the collection flow: the runtime stays
        resident so several groups can be generated back to back; scoring (and
        the rollout offload shared-GPU reward runs need before it) happens in
        score_rollouts(). ``metadata`` is the group's opaque reward-scoring
        payload (see ``PromptExample.reward_metadata``).
        """

        collector_request = self.request_builder.build(
            inputs,
            int(group_size),
            metadata=metadata,
            request_overrides=request_overrides,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        # A new generation phase has not activated reward memory yet. This also
        # prevents a previous iteration's state from authorizing a later handoff.
        self._reward_phase_started = False

        profile = os.environ.get("VRL_PROFILE") == "1"
        phase_t = time.perf_counter() if profile else None

        output = await self.generation_runtime.generate(collector_request.request)
        unscored = UnscoredRollout(
            output=output,
            collector_request=collector_request,
            profile=profile,
        )
        if profile and phase_t is not None:
            unscored.phases["collect.engine_generate"] = time.perf_counter() - phase_t
        return unscored

    async def score_rollouts(self, unscored: list[UnscoredRollout]) -> list[RolloutBatch]:
        """Score unscored groups through one reward call and build their batches.

        Rollout prompts/metadata are per-sample, so all groups score in a
        single reward_runtime.score call — model-backed rewards with
        phase parking pay one model activation per call instead of one per
        group. Batches return in input order.
        """

        if not unscored:
            return []
        if self.requires_generation_offload_before_reward:
            # Shared single-GPU reward runs park rollout model memory before the
            # in-process reward model takes over the physical GPU.
            await self.generation_runtime.offload()

        # self.config already holds these as resolved, typed fields (frozen for the
        # batch). Read them directly — feeding the typed trajectory_storage policy
        # back through trajectory_storage_policy_from_cfg raised TypeError.
        kl_reward_coef = self.config.kl_reward_coef
        trajectory_storage_policy = self.config.trajectory_storage
        builders = []
        for rollout in unscored:
            context = RolloutBatchBuildContext(
                metadata=dict(rollout.collector_request.metadata),
                # Collector/continuous-owner output is host-owned. The trainer
                # moves the completed batch to its device later; creating reward
                # tensors here on the trainer GPU would race backward.
                device="cpu",
                kl_reward_coef=kl_reward_coef,
                trajectory_storage_policy=trajectory_storage_policy,
                trajectory_layout=self._trajectory_layout,
            )
            builders.append(TrajectoryRolloutBatchBuilder(rollout.output, context))

        profile = any(rollout.profile for rollout in unscored)
        phase_t = time.perf_counter() if profile else None
        require_reward_release = self._requires_reward_memory_release()
        self._reward_phase_started = require_reward_release
        reward_samples = [builder.reward_samples() for builder in builders]
        samples = tuple(sample for group_samples in reward_samples for sample in group_samples)
        with profile_range("collector.reward_score"):
            score_result = await self.reward_runtime.score(
                samples,
                require_memory_release=require_reward_release,
            )
        reward_timing_ms = dict(score_result.timing_ms)
        if reward_timing_ms:
            unscored[0].reward_timing_ms.update(reward_timing_ms)
        reward_score_s = time.perf_counter() - phase_t if phase_t is not None else None

        build_t = time.perf_counter() if profile else None
        batches: list[RolloutBatch] = []
        sample_offset = 0
        for builder, group_samples in zip(
            builders,
            reward_samples,
            strict=True,
        ):
            sample_stop = sample_offset + len(group_samples)
            group_rewards = torch.tensor(
                score_result.scores[sample_offset:sample_stop],
                dtype=torch.float32,
                device=builder.context.device or "cpu",
            )
            batch = builder.build(group_rewards)
            if score_result.components:
                # Raw component values travel with their exact rollout rows.
                # A continuous producer may already be scoring a future group
                # when this batch is consumed, so trainer metrics must never
                # read a mutable last-result cache from the shared reward model.
                batch.extras["reward_components"] = {
                    name: torch.tensor(
                        values[sample_offset:sample_stop],
                        dtype=torch.float32,
                    )
                    for name, values in score_result.components.items()
                }
            sample_offset = sample_stop
            batches.append(batch)
        if reward_score_s is not None and build_t is not None:
            # One score call and one build pass cover every group, so the
            # call-level timings live on the first group only: a caller summing
            # phases over groups must not multiply the same wall time. The
            # phases stay on the rollouts (caller-owned) so concurrent collects
            # never share mutable collector state.
            unscored[0].phases["collect.reward_score"] = reward_score_s
            unscored[0].phases["collect.batch_build"] = time.perf_counter() - build_t
        return batches

    def _requires_reward_memory_release(self) -> bool:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_reward_after_score

    @property
    def requires_generation_offload_before_reward(self) -> bool:
        """Whether scoring introduces a mid-iteration GPU handoff."""

        # The release decision is derived once from GPU topology into the
        # lifecycle plan (vrl/ray/resources.py), not re-decided per call by the
        # runtime. None plan = no shared GPU = never release before reward.
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_rollout_before_reward

    @property
    def requires_driver_model_offload_for_reward(self) -> bool:
        """Whether reward scoring borrows the trainer's in-process GPU."""

        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.handoff.release_trainer_before_reward

    @property
    def supports_reward_generation_overlap(self) -> bool:
        """Whether strict collection can stream score N beside generation N+1."""

        return bool(
            self._reward_accelerator_isolation_verified
            and self.reward_runtime.scoring_is_nonblocking
        )

    @property
    def supports_continuous_reward_execution(self) -> bool:
        """Whether reward placement is safe beside continuous trainer/rollout work."""

        return self._reward_accelerator_isolation_verified

    @property
    def _reward_accelerator_isolation_verified(self) -> bool:
        """Combine local topology with proof for out-of-plan accelerators."""

        return bool(
            not self.requires_generation_offload_before_reward
            and not self.requires_driver_model_offload_for_reward
            and self.reward_runtime.external_accelerator_isolation_verified
        )


def build_rollout_collector(
    entry: ModelFamilyEntry,
    *,
    reward_runtime: RewardRuntime,
    config: RolloutCollectorConfig,
    generation_runtime: GenerationRuntime | None = None,
    lifecycle: RayLifecyclePlan | None = None,
) -> RolloutCollector:
    """Build a rollout collector from an already resolved family entry."""

    return RolloutCollector(
        config=config,
        request_builder=GenerationRequestBuilder(entry=entry, config=config),
        reward_runtime=reward_runtime,
        generation_runtime=generation_runtime,
        lifecycle=lifecycle,
        trajectory_layout=entry.policy_semantics.trajectory_layout,
    )


__all__ = [
    "RolloutCollector",
    "UnscoredRollout",
    "build_rollout_collector",
]
