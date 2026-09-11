"""Shared rollout collector orchestration.

The one place the two engine protocols meet: ``RolloutCollector`` drives
``GenerationRuntime.generate`` and ``RewardRuntime.score`` for a prompt group
and owns the phase seam between them — reading the topology-derived
``RayLifecyclePlan`` (``vrl/ray/resources.py``) to decide which role must park
its GPU before the next phase, and exposing those decisions as capabilities
(overlap, continuous execution) instead of letting schedules re-derive them.
Prompt grouping, deferred scoring, task cleanup, and result accounting also
live on the collector, shared by strict and continuous schedules. Admission,
weight synchronization, and staleness remain schedule responsibilities.
Trajectory packing details live in ``batch_builder``; request construction in
``requests``.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vrl.ray.resources import RayLifecyclePlan

from vrl.generation import GenerationInput, GenerationOutput, GenerationRuntime
from vrl.models.families.registry import ModelFamilyEntry
from vrl.rewards import RewardRuntime
from vrl.rewards.base import RewardCleanupError
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import remap_group_ids_, split_batch_by_group
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import (
    CollectorRequest,
    GenerationRequestBuilder,
)
from vrl.rollouts.stats import RolloutStats
from vrl.utils.profiling import profile_range


@dataclass(slots=True)
class UnscoredRollout:
    """A generated-but-unscored prompt group awaiting deferred reward scoring."""

    output: GenerationOutput
    collector_request: CollectorRequest
    profile: bool = False
    phases: dict[str, float] = field(default_factory=dict)
    reward_timing_ms: dict[str, float] = field(default_factory=dict)


class RewardCollectionMode(str, Enum):  # noqa: UP042
    """How prompt collection interleaves group generation and reward scoring.

    Strict collection picks between ``BATCHED_SERIAL`` and
    ``PER_GROUP_STREAMING`` from the collector's overlap capability. Continuous
    collection already dispatches one task per group, so it uses
    ``BATCHED_SERIAL`` inside each task instead of creating an inner task that
    cannot add overlap. ``PER_GROUP_SERIAL`` is the acceptance control arm
    required by ``docs/sprints/done/SPRINT_reward_service.md``: it moves strict
    scoring to per-group granularity *without* overlap, so the per-group
    call/transport tax can be measured separately from the overlap gain.
    """

    BATCHED_SERIAL = "batched_serial"
    PER_GROUP_SERIAL = "per_group_serial"
    PER_GROUP_STREAMING = "per_group_streaming"


class PromptCollectionCleanupError(RuntimeError):
    """Generation failed while an in-flight reward task also failed to settle."""

    def __init__(
        self,
        root_cause: BaseException,
        cleanup_errors: list[BaseException],
    ) -> None:
        self.root_cause = root_cause
        self.cleanup_errors = tuple(cleanup_errors)
        cleanup = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
        super().__init__(
            f"prompt collection root cause: {type(root_cause).__name__}: {root_cause}; "
            f"in-flight reward cleanup failures: {cleanup}",
        )


@dataclass(slots=True)
class GeneratedPromptGroup:
    """Generation receipt retained until scoring and prompt remapping finish.

    Both schedules use this handoff so deferred scoring preserves example
    metadata and the original prompt indices without reconstructing requests.
    Times use the local performance clock, never a remote worker clock.
    """

    unscored: UnscoredRollout
    prompt_indices: list[int]
    started_at: float
    completed_at: float


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

    @classmethod
    def from_family(
        cls,
        entry: ModelFamilyEntry,
        *,
        reward_runtime: RewardRuntime,
        config: RolloutCollectorConfig,
        generation_runtime: GenerationRuntime | None = None,
        lifecycle: RayLifecyclePlan | None = None,
    ) -> RolloutCollector:
        """Build a rollout collector from an already resolved family entry."""

        return cls(
            config=config,
            request_builder=GenerationRequestBuilder(entry=entry, config=config),
            reward_runtime=reward_runtime,
            generation_runtime=generation_runtime,
            lifecycle=lifecycle,
            trajectory_layout=entry.policy_semantics.trajectory_layout,
        )

    def set_generation_runtime(self, runtime: GenerationRuntime) -> None:
        self._generation_runtime = runtime

    @property
    def generation_runtime(self) -> GenerationRuntime | None:
        """Return the attached runtime, or None during setup."""
        return self._generation_runtime

    def _require_generation_runtime(self) -> GenerationRuntime:
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
        await self._require_generation_runtime().activate()

    async def offload_generation_runtime_memory(self) -> None:
        errors: list[BaseException] = []
        try:
            await self._require_generation_runtime().offload()
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
            group_size,
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

        output = await self._require_generation_runtime().generate(collector_request.request)
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
            await self._require_generation_runtime().offload()
        # Symmetric handoff to the generation side's activate(): pre-warm the
        # reward model here so its build/wake latency is not billed to the
        # measured scoring phase below.
        await self.reward_runtime.activate()

        # self.config already holds these as resolved, typed fields (frozen for the
        # batch). Read them directly — feeding the typed trajectory_storage policy
        # back through TrajectoryStoragePolicy.from_config raised TypeError.
        kl_reward_coef = self.config.kl_reward_coef
        trajectory_storage_policy = self.config.trajectory_storage
        builders = []
        for rollout in unscored:
            reward_metadata = dict(rollout.collector_request.metadata)
            policy_version = rollout.collector_request.request.policy_version
            if policy_version is not None:
                # The scorer owns rollout audit artifacts before the trainer has
                # an epoch number. Policy version is the stable update identity
                # across supervisor resume; keep it beside the scored media.
                reward_metadata["rollout_policy_version"] = int(policy_version)
            context = RolloutBatchBuildContext(
                metadata=reward_metadata,
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
        return lifecycle.release_reward_after_score

    @property
    def requires_generation_offload_before_reward(self) -> bool:
        """Whether scoring introduces a mid-iteration GPU handoff."""

        # The release decision is derived once from GPU topology into the
        # lifecycle plan (vrl/ray/resources.py), not re-decided per call by the
        # runtime. None plan = no shared GPU = never release before reward.
        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.release_rollout_before_reward

    @property
    def requires_driver_model_offload_for_reward(self) -> bool:
        """Whether reward scoring borrows the trainer's in-process GPU."""

        lifecycle = self._lifecycle
        if lifecycle is None:
            return False
        return lifecycle.release_trainer_before_reward

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

    async def generate_prompt_groups(
        self,
        *,
        prompts: list[Any],
        group_size: int,
        runtime_debug: bool,
        policy_version: int | None,
    ) -> AsyncIterator[GeneratedPromptGroup]:
        """Yield generated groups without acquiring or running the reward runtime.

        Consecutive plain prompts retain their batched request; structured examples
        retain their own metadata and overrides. Iteration is demand-driven: the
        caller owns admission, each yielded receipt, and any downstream scoring.
        """

        pending_prompts: list[str] = []
        pending_indices: list[int] = []

        async def generate(
            inputs: list[Any], indices: list[int], **kwargs: Any
        ) -> GeneratedPromptGroup:
            started = time.perf_counter()
            unscored = await self.collect_unscored(
                inputs,
                group_size=group_size,
                runtime_debug=runtime_debug,
                policy_version=policy_version,
                **kwargs,
            )
            return GeneratedPromptGroup(unscored, indices, started, time.perf_counter())

        for prompt_idx, item in enumerate(prompts):
            if not isinstance(item, (str, bytes)) and hasattr(item, "generation_input"):
                if pending_prompts:
                    yield await generate(
                        [GenerationInput(prompt=prompt) for prompt in pending_prompts],
                        list(pending_indices),
                    )
                    pending_prompts.clear()
                    pending_indices.clear()
                yield await generate(
                    [item.generation_input()],
                    [prompt_idx],
                    metadata=item.reward_metadata(),
                    request_overrides=dict(item.request_overrides or {}),
                )
            else:
                pending_prompts.append(str(item))
                pending_indices.append(prompt_idx)
        if pending_prompts:
            yield await generate(
                [GenerationInput(prompt=prompt) for prompt in pending_prompts],
                list(pending_indices),
            )

    async def collect_prompt_groups(
        self,
        *,
        prompts: list[Any],
        group_size: int,
        runtime_debug: bool,
        policy_version: int | None,
        stats: RolloutStats,
        reward_mode: RewardCollectionMode | None = None,
    ) -> list[RolloutBatch]:
        """Collect every trainer prompt's sample group and return per-group batches.

        Collectors without an explicit overlap capability keep two strict phases:
        generate every prompt group, then score all groups through one reward call.
        A capable collector may score group N while generating group N+1. The
        streaming path owns at most one scoring task, so reward work has bounded
        backpressure and deterministic cleanup.

        ``reward_mode`` overrides that derived choice for acceptance measurement
        only; see :class:`RewardCollectionMode`. A capable collector may be forced
        onto either control arm, while an incapable collector stays on the batched
        baseline.

        ``stats`` accumulates this call's collect phase timings
        (``collect.engine_generate`` / ``collect.reward_score`` /
        ``collect.batch_build``) plus any reward-inference timings. The accumulator
        is owned by this call, so concurrent collects never overwrite each other.
        """

        if not prompts:
            return []

        collection_started = time.perf_counter()
        reward_intervals: list[tuple[float, float]] = []

        generated_groups: list[GeneratedPromptGroup] = []
        scored_batches: list[RolloutBatch] = []
        # The collector combines topology and reward-runtime execution semantics.
        # Only its capability may enable per-group collection: the acceptance
        # override can restrict a capable collector, but cannot grant the runtime
        # isolation needed to alternate generation and scoring safely.
        overlap_capable = bool(self.supports_reward_generation_overlap)
        if reward_mode is None:
            mode = (
                RewardCollectionMode.PER_GROUP_STREAMING
                if overlap_capable
                else RewardCollectionMode.BATCHED_SERIAL
            )
        elif reward_mode is not RewardCollectionMode.BATCHED_SERIAL and not overlap_capable:
            raise ValueError(
                f"reward collection mode {reward_mode.value!r} requires the collector's "
                "reward/generation overlap capability (async scoring plus verified "
                "accelerator isolation); it cannot be forced on",
            )
        else:
            mode = reward_mode
        per_group_scoring = mode is not RewardCollectionMode.BATCHED_SERIAL
        score_task: asyncio.Task[list[RolloutBatch]] | None = None

        async def score_unscored(groups: list[UnscoredRollout]) -> list[RolloutBatch]:
            started = time.perf_counter()
            try:
                return await self.score_rollouts(groups)
            finally:
                reward_intervals.append((started, time.perf_counter()))

        def accept_single_batch(batches: list[RolloutBatch]) -> None:
            if len(batches) != 1:
                raise RuntimeError(
                    "per-group reward scoring must return one batch for one unscored group, "
                    f"got {len(batches)}",
                )
            scored_batches.extend(batches)

        async def drain_score_task() -> None:
            nonlocal score_task
            task = score_task
            if task is None:
                return
            # Clear ownership before awaiting so a task failure is not mistaken for
            # a second cleanup failure by the outer exception handler.
            score_task = None
            accept_single_batch(await task)

        async def record_generated(group: GeneratedPromptGroup) -> None:
            nonlocal score_task
            generated_groups.append(group)
            unscored = group.unscored
            if not per_group_scoring:
                return
            if mode is RewardCollectionMode.PER_GROUP_SERIAL:
                # Control arm: same per-group call granularity as streaming, but the
                # score completes before the next generation starts. The measured
                # difference against streaming is overlap alone.
                accept_single_batch(await score_unscored([unscored]))
                return
            # Generation of this group ran while the previous scoring task was in
            # flight. Drain it before starting this group's task: at most one reward
            # call can own service/model state at a time.
            await drain_score_task()
            score_task = asyncio.create_task(
                score_unscored([unscored]),
                name="rollout-reward-score",
            )

        try:
            async for generated in self.generate_prompt_groups(
                prompts=prompts,
                group_size=group_size,
                runtime_debug=runtime_debug,
                policy_version=policy_version,
            ):
                await record_generated(generated)
            if not generated_groups:
                return []

            if per_group_scoring:
                # PER_GROUP_SERIAL already drained inline; only streaming can still
                # own a task here.
                await drain_score_task()
                batches = scored_batches
            else:
                batches = await score_unscored(
                    [group.unscored for group in generated_groups],
                )
        except BaseException as root_cause:
            cleanup_errors: list[BaseException] = []
            task = score_task
            score_task = None
            if task is not None:
                # Do not wait indefinitely for remote scoring after generation has
                # already failed. Reward cancellation owns request deletion and
                # artifact/model cleanup; gather keeps that cleanup attached to this
                # collection call.
                task.cancel()
                cleanup_results = await asyncio.gather(task, return_exceptions=True)
                cleanup_errors.extend(
                    result
                    for result in cleanup_results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                )
            if cleanup_errors:
                raise PromptCollectionCleanupError(
                    root_cause,
                    cleanup_errors,
                ) from root_cause
            raise

        all_batches = self.finish_scored_prompt_groups(generated_groups, batches, stats)
        stats.add_collection_timing(
            wall_s=time.perf_counter() - collection_started,
            generation_intervals=[
                (group.started_at, group.completed_at) for group in generated_groups
            ],
            reward_intervals=reward_intervals,
        )
        stats.add_counter("collect.group_count", len(all_batches))
        stats.add_counter(
            "collect.sample_count",
            sum(int(batch.rewards.shape[0]) for batch in all_batches),
        )
        return all_batches

    def finish_scored_prompt_groups(
        self,
        generated_groups: list[GeneratedPromptGroup],
        batches: list[RolloutBatch],
        stats: RolloutStats,
    ) -> list[RolloutBatch]:
        """Account collector timings and restore prompt identities after scoring."""

        # Per-call phases live on the unscored groups (collector writes the
        # call-level score/build timings and reward inference timings on the
        # first group only).
        for group in generated_groups:
            unscored = group.unscored
            stats.add_phases(unscored.phases)
            reward_timing_ms = unscored.reward_timing_ms
            if reward_timing_ms:
                standard_keys = {"latency_ms", "queue_wait_ms", "inference_ms"}
                stats.fold_reward_timing(
                    latency_ms=reward_timing_ms.get("latency_ms"),
                    queue_wait_ms=reward_timing_ms.get("queue_wait_ms"),
                    inference_ms=reward_timing_ms.get("inference_ms"),
                    extra_ms={
                        str(name): float(value)
                        for name, value in reward_timing_ms.items()
                        if name not in standard_keys and str(name).endswith("_ms")
                    },
                )

        all_batches: list[RolloutBatch] = []
        for batch, group in zip(batches, generated_groups, strict=True):
            remap_group_ids_(batch, group.prompt_indices)
            all_batches.extend(split_batch_by_group(batch))
        return all_batches


__all__ = [
    "GeneratedPromptGroup",
    "PromptCollectionCleanupError",
    "RewardCollectionMode",
    "RolloutCollector",
    "UnscoredRollout",
]
