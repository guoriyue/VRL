"""Shared rollout collector orchestration.

The one place the two engine protocols meet: ``RolloutCollector`` drives
``GenerationRuntime.generate`` and ``RewardRuntime.score`` for a prompt group
and owns the phase seam between them — reading the topology-derived
``RoleLifecycle`` (``vrl/ray/resources.py``) to decide which role must park
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
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from vrl.ray.resources import RoleLifecycle

from vrl.generation import GenerationInput, GenerationOutput, GenerationRuntime
from vrl.models.families.registry import ModelFamilyEntry
from vrl.rewards import RewardOutput, RewardRuntime
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
from vrl.utils.profiling import TimeIntervals, profile_range
from vrl.utils.stats import count, elapsed, timed


@dataclass(slots=True)
class UnscoredRollout:
    """A generated-but-unscored prompt group awaiting deferred reward scoring."""

    output: GenerationOutput
    collector_request: CollectorRequest


@dataclass(slots=True)
class RolloutEvaluation:
    """Request-local rewards and trajectory builders awaiting batch assembly."""

    rollouts: list[UnscoredRollout]
    builders: list[TrajectoryRolloutBatchBuilder]
    sample_counts: list[int]
    rewards: RewardOutput


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
class RolloutGenerationResult:
    """Unscored rollout plus original prompt indices and local generation timings.

    Both schedules use this handoff so deferred scoring preserves example
    metadata and the original prompt indices without reconstructing requests.
    Times use the local performance clock, never a remote worker clock.
    """

    unscored: UnscoredRollout
    prompt_indices: list[int]
    started_at: float
    completed_at: float


@runtime_checkable
class OwnedCollection(Protocol):
    """A prompt item that runs its own generation plan through the collector.

    ``prepare_training_batches`` hands such items the collector and returns the
    per-group batches they produce, each group's ids local to that item. The
    item may issue any number of ordinary requests through
    ``request_builder`` / ``generate_rollout`` / ``evaluate_rollout`` and must
    finish with ``finish_scored_prompt_groups``. The framework does not know
    what the plan is; multi-step edit chains (the sibling ``agentic`` repo) are one.
    Its phases and counters go to the caller's bound stats (``vrl.utils.stats``).
    """

    async def collect(
        self,
        collector: RolloutCollector,
        *,
        group_size: int,
        policy_version: int | None,
    ) -> list[RolloutBatch]: ...


class RolloutCollector:
    """Generic collector: request -> generation runtime -> reward -> trainer batch."""

    def __init__(
        self,
        *,
        request_builder: GenerationRequestBuilder,
        reward_runtime: RewardRuntime,
        lifecycle: RoleLifecycle,
    ) -> None:
        self.request_builder = request_builder
        self.reward_runtime = reward_runtime
        # The recipe attaches the generation runtime once it is launched; the
        # collector exists first so its shutdown covers the launch.
        self._generation_runtime: GenerationRuntime | None = None
        # Topology-derived handoff policy (vrl/ray/resources.py), read here
        # instead of asking the runtime, which is just transport.
        self.lifecycle = lifecycle
        self._reward_shutdown_complete = False

    @classmethod
    def from_family(
        cls,
        entry: ModelFamilyEntry,
        *,
        reward_runtime: RewardRuntime,
        config: RolloutCollectorConfig,
        lifecycle: RoleLifecycle,
    ) -> RolloutCollector:
        """Build a rollout collector from an already resolved family entry."""

        return cls(
            request_builder=GenerationRequestBuilder(entry=entry, config=config),
            reward_runtime=reward_runtime,
            lifecycle=lifecycle,
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

        runtime = self._generation_runtime
        if runtime is not None:
            # A slept reward pool must not remap pages while rollout ownership is
            # unknown. Retain it asleep and let the next shutdown retry finish
            # generation first.
            await runtime.shutdown()
            self._generation_runtime = None
        if not self._reward_shutdown_complete:
            await self.reward_runtime.shutdown()
            self._reward_shutdown_complete = True

    async def activate_generation_runtime(self) -> None:
        await self._require_generation_runtime().activate()

    async def offload_generation_runtime_memory(self) -> None:
        errors: list[BaseException] = []
        try:
            await self._require_generation_runtime().offload()
        except BaseException as error:
            errors.append(error)
        # Phase-final gate: a reward that shares a GPU must be parked before the
        # trainer restores; this retries a park that failed after scoring.
        if self.reward_yields_gpu:
            try:
                await self.reward_runtime.park_memory()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RewardCleanupError(
                "rollout and reward memory parking both failed",
                errors,
            )

    async def generate_rollout(self, collector_request: CollectorRequest) -> UnscoredRollout:
        """Execute one prepared generation request without acquiring rewards."""

        # Starts at submission: a prefetched group's time overlaps the running
        # one's, so the sum over groups may exceed the call's wall time.
        with timed("collect.engine_generate"):
            output = await self._require_generation_runtime().generate(collector_request.request)
        return UnscoredRollout(output=output, collector_request=collector_request)

    async def evaluate_rollout(self, unscored: list[UnscoredRollout]) -> RolloutEvaluation | None:
        """Evaluate generated samples through one reward call, without assembling batches."""

        if not unscored:
            return None
        if self.requires_generation_offload_before_reward:
            # Shared single-GPU reward runs park rollout model memory before the
            # in-process reward model takes over the physical GPU.
            await self._require_generation_runtime().offload()
        # Symmetric handoff to the generation side's activate(): pre-warm the
        # reward model here so its build/wake latency is not billed to the
        # measured scoring phase below.
        await self.reward_runtime.activate()

        # self.config already holds these as resolved, typed fields (frozen for the
        # batch); the schema parsed the trajectory_storage policy once.
        trajectory_storage_policy = self.request_builder.config.trajectory_storage
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
                trajectory_storage_policy=trajectory_storage_policy,
            )
            builders.append(TrajectoryRolloutBatchBuilder(rollout.output, context))

        reward_samples = [builder.reward_samples() for builder in builders]
        samples = tuple(sample for group_samples in reward_samples for sample in group_samples)
        score_error: BaseException | None = None
        with timed("collect.reward_score"):
            try:
                with profile_range("collector.reward_score"):
                    score_result = await self.reward_runtime.score(samples)
            except BaseException as error:
                score_error = error
            if self.reward_yields_gpu:
                # The reward shares a GPU: it yields it as soon as scoring is over,
                # whether or not scoring succeeded, exactly as rollout yields
                # before scoring.
                try:
                    await self.reward_runtime.park_memory()
                except BaseException as park_error:
                    if score_error is None:
                        raise
                    raise RewardCleanupError(
                        "reward scoring and memory parking both failed",
                        [score_error, park_error],
                    ) from score_error
        if score_error is not None:
            raise score_error
        return RolloutEvaluation(
            rollouts=unscored,
            builders=builders,
            sample_counts=[len(samples) for samples in reward_samples],
            rewards=score_result,
        )

    def assemble_training_batches(
        self, evaluation: RolloutEvaluation | None
    ) -> list[RolloutBatch]:
        """Assemble trajectories and evaluated rewards into trainer-owned batches."""

        if evaluation is None:
            return []
        score_result = evaluation.rewards
        batches: list[RolloutBatch] = []
        with timed("collect.batch_build"):
            sample_offset = 0
            for builder, sample_count in zip(
                evaluation.builders,
                evaluation.sample_counts,
                strict=True,
            ):
                sample_stop = sample_offset + sample_count
                group_rewards = torch.tensor(
                    score_result.scores[sample_offset:sample_stop],
                    dtype=torch.float32,
                    # Collector output is host-owned; the trainer moves the batch
                    # to its device later, so reward tensors never race backward.
                    device="cpu",
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
        return batches

    @property
    def requires_generation_offload_before_reward(self) -> bool:
        """Whether scoring introduces a mid-iteration GPU handoff."""

        # The release decision is derived once from GPU topology into the
        # lifecycle plan (vrl/ray/resources.py), not re-decided per call by the
        # runtime. None plan = no shared GPU = never release before reward.
        return self.lifecycle.park_rollout_for_reward

    @property
    def reward_yields_gpu(self) -> bool:
        """Whether the reward shares a GPU and must park after scoring (plan bit)."""

        return self.lifecycle.offload_reward

    @property
    def reward_isolation_verified(self) -> bool:
        """Whether the reward shares no GPU with generation or training.

        Read off the lifecycle plan. An HTTP service is outside the plan and
        stays resident on its own device, so it never blocks overlap.
        """

        return not self.lifecycle.offload_reward

    def build_generation_requests(
        self,
        *,
        prompts: list[Any],
        group_size: int,
        policy_version: int | None,
    ) -> Iterator[tuple[CollectorRequest, list[int]]]:
        """Build requests and original prompt indices without executing generation.

        Consecutive plain prompts share a request; structured examples retain
        their own metadata and overrides. Requests are built on demand.
        """

        pending_prompts: list[str] = []
        pending_indices: list[int] = []

        def build(inputs: list[Any], indices: list[int], **kwargs: Any):
            request = self.request_builder.build(
                inputs,
                group_size=group_size,
                policy_version=policy_version,
                reward_media_refs=True,
                **kwargs,
            )
            return request, indices

        for prompt_idx, item in enumerate(prompts):
            if isinstance(item, OwnedCollection):
                raise ValueError(
                    "an owned collection is collected through prepare_training_batches"
                )
            if not isinstance(item, (str, bytes)) and hasattr(item, "generation_input"):
                if pending_prompts:
                    yield build(
                        [GenerationInput(prompt=prompt) for prompt in pending_prompts],
                        list(pending_indices),
                    )
                    pending_prompts.clear()
                    pending_indices.clear()
                yield build(
                    [item.generation_input()],
                    [prompt_idx],
                    metadata=item.reward_metadata(),
                    request_overrides=dict(item.request_overrides or {}),
                )
            else:
                pending_prompts.append(str(item))
                pending_indices.append(prompt_idx)
        if pending_prompts:
            yield build(
                [GenerationInput(prompt=prompt) for prompt in pending_prompts],
                list(pending_indices),
            )

    async def prepare_training_batches(
        self,
        *,
        prompts: list[Any],
        group_size: int,
        policy_version: int | None,
    ) -> list[RolloutBatch]:
        """Collect every trainer prompt's sample group and return per-group batches.

        Collectors without an explicit overlap capability keep two strict phases:
        generate every prompt group, then score all groups through one reward call.
        A capable collector may score group N while generating group N+1. The
        streaming path owns at most one scoring task, so reward work has bounded
        backpressure and deterministic cleanup. Either way the next group's
        generation request is submitted before the current one is awaited, so
        the engine never idles between two groups.

        The call's collect phase timings (``collect.engine_generate`` /
        ``collect.reward_score`` / ``collect.batch_build`` / ``collect.wall``
        ...) and counters land in the caller's bound stats
        (``vrl.utils.stats``), as does what the generation and reward work for
        this call records, in this process or returned from an actor. A caller
        running several collects concurrently binds one scope per call.
        """

        if not prompts:
            return []
        owned = [isinstance(item, OwnedCollection) for item in prompts]
        if any(owned):
            if not all(owned):
                raise ValueError("owned collections and one-shot prompts cannot share a call")
            batches: list[RolloutBatch] = []
            for item in prompts:
                # Group ids only need to be distinct within this call; downstream
                # consumers renumber them. Offset each item's local ids.
                offset = len(batches)
                for batch in await item.collect(
                    self,
                    group_size=group_size,
                    policy_version=policy_version,
                ):
                    batch.group_ids = batch.group_ids + offset
                    batches.append(batch)
            return batches

        collection_started = time.perf_counter()
        reward_intervals: list[tuple[float, float]] = []

        generated_groups: list[RolloutGenerationResult] = []
        scored_batches: list[RolloutBatch] = []
        # Scoring runs beside generation only when its accelerator needs no GPU
        # handoff; online rewards are Ray actors or HTTP services, so scoring
        # itself always yields the event loop.
        per_group_scoring = self.reward_isolation_verified
        score_task: asyncio.Task[list[RolloutBatch]] | None = None

        async def score_unscored(groups: list[UnscoredRollout]) -> list[RolloutBatch]:
            started = time.perf_counter()
            try:
                return self.assemble_training_batches(await self.evaluate_rollout(groups))
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

        async def record_generated(group: RolloutGenerationResult) -> None:
            nonlocal score_task
            generated_groups.append(group)
            unscored = group.unscored
            if not per_group_scoring:
                return
            # Generation of this group ran while the previous scoring task was in
            # flight. Drain it before starting this group's task: at most one reward
            # call can own service/model state at a time.
            await drain_score_task()
            score_task = asyncio.create_task(
                score_unscored([unscored]),
                name="rollout-reward-score",
            )

        # Keep one request submitted ahead so the engine never idles between two
        # groups. The engine admits one request at a time, so this only removes
        # the gap: the next request starts once the current one's batches are
        # staged, while the current one is merged and scored.
        pending_generation: tuple[asyncio.Task[UnscoredRollout], float] | None = None

        def start_generation(
            request: CollectorRequest,
        ) -> tuple[asyncio.Task[UnscoredRollout], float]:
            return (
                asyncio.create_task(self.generate_rollout(request), name="rollout-generate"),
                time.perf_counter(),
            )

        try:
            planned = list(
                self.build_generation_requests(
                    prompts=prompts,
                    group_size=group_size,
                    policy_version=policy_version,
                ),
            )
            for index, (request, prompt_indices) in enumerate(planned):
                if pending_generation is None:
                    generation, started = start_generation(request)
                else:
                    generation, started = pending_generation
                    pending_generation = None
                if index + 1 < len(planned):
                    pending_generation = start_generation(planned[index + 1][0])
                unscored = await generation
                generated = RolloutGenerationResult(
                    unscored, prompt_indices, started, time.perf_counter()
                )
                await record_generated(generated)
            if not generated_groups:
                return []

            if per_group_scoring:
                await drain_score_task()
                batches = scored_batches
            else:
                batches = await score_unscored(
                    [group.unscored for group in generated_groups],
                )
        except BaseException as root_cause:
            cleanup_errors: list[BaseException] = []
            prefetched = pending_generation
            pending_generation = None
            if prefetched is not None:
                # The group after the failed one may already be generating.
                # Cancel it so the engine is not left running a request nobody
                # will score; its own failure is a cleanup error, not the cause.
                prefetched_task, _ = prefetched
                prefetched_task.cancel()
                prefetch_results = await asyncio.gather(prefetched_task, return_exceptions=True)
                cleanup_errors.extend(
                    result
                    for result in prefetch_results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                )
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

        all_batches = self.finish_scored_prompt_groups(generated_groups, batches)
        generation_timing = TimeIntervals(
            (group.started_at, group.completed_at) for group in generated_groups
        )
        reward_timing = TimeIntervals(reward_intervals)
        elapsed("collect.wall", time.perf_counter() - collection_started)
        elapsed("collect.generation_wall", generation_timing.duration)
        elapsed("collect.reward_wall", reward_timing.duration)
        elapsed("collect.generation_reward_overlap", generation_timing.overlap(reward_timing))
        count("collect.group_count", len(all_batches))
        count("collect.sample_count", sum(int(batch.rewards.shape[0]) for batch in all_batches))
        return all_batches

    def finish_scored_prompt_groups(
        self,
        generated_groups: list[RolloutGenerationResult],
        batches: list[RolloutBatch],
    ) -> list[RolloutBatch]:
        """Restore prompt identities after scoring and split per group."""

        all_batches: list[RolloutBatch] = []
        for batch, group in zip(batches, generated_groups, strict=True):
            batch.remap_group_ids_(group.prompt_indices)
            all_batches.extend(batch.split_by_group())
        return all_batches


__all__ = [
    "OwnedCollection",
    "PromptCollectionCleanupError",
    "RolloutCollector",
    "RolloutEvaluation",
    "RolloutGenerationResult",
    "UnscoredRollout",
]
