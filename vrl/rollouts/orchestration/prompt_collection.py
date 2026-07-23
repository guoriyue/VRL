"""Prompt collection helpers for rollout schedules."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from vrl.generation import GenerationInput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import remap_group_ids_, split_batch_by_group
from vrl.rollouts.orchestration.types import RewardCollectionMode
from vrl.utils.stats import RolloutStats


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


async def collect_prompt_batches(
    *,
    collector: Any,
    prompts: list[Any],
    group_size: int,
    runtime_debug: bool,
    policy_version: int | None,
    stats: RolloutStats | None = None,
    reward_mode: RewardCollectionMode | None = None,
) -> list[RolloutBatch]:
    """Collect trainer prompts through ``RolloutCollector`` and split by group.

    Collectors without an explicit overlap capability keep two strict phases:
    generate every prompt group, then score all groups through one reward call.
    A capable collector may score group N while generating group N+1. The
    streaming path owns at most one scoring task, so reward work has bounded
    backpressure and deterministic cleanup.

    ``reward_mode`` overrides that derived choice for acceptance measurement
    only; see :class:`RewardCollectionMode`. A capable collector may be forced
    onto either control arm, while an incapable collector stays on the batched
    baseline.

    ``stats`` (when given) accumulates this call's collect phase timings
    (``collect.engine_generate`` / ``collect.reward_score`` /
    ``collect.batch_build``) plus any reward-inference timings. The accumulator
    is owned by this call, so concurrent collects never overwrite each other.
    """

    if not prompts:
        return []

    collection_started = time.perf_counter()
    generation_intervals: list[tuple[float, float]] = []
    reward_intervals: list[tuple[float, float]] = []

    # (unscored group, group-id remap: per-sample indices for plain-string
    # batches, a single prompt index for PromptExample groups)
    unscored_groups: list[tuple[Any, list[int] | int]] = []
    scored_batches: list[RolloutBatch] = []
    pending_prompts: list[str] = []
    pending_indices: list[int] = []
    # The collector combines topology and reward-runtime execution semantics.
    # Only its capability may enable per-group collection: the acceptance
    # override can restrict a capable collector, but cannot grant the runtime
    # isolation needed to alternate generation and scoring safely.
    overlap_capable = bool(
        getattr(collector, "supports_reward_generation_overlap", False),
    )
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

    async def collect_unscored(
        inputs: list[Any],
        **kwargs: Any,
    ) -> Any:
        started = time.perf_counter()
        try:
            return await collector.collect_unscored(inputs, **kwargs)
        finally:
            generation_intervals.append((started, time.perf_counter()))

    async def score_unscored(groups: list[Any]) -> list[RolloutBatch]:
        started = time.perf_counter()
        try:
            return await collector.score_rollouts(groups)
        finally:
            reward_intervals.append((started, time.perf_counter()))

    def take_single(batches: list[RolloutBatch]) -> None:
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
        take_single(await task)

    async def record_unscored(
        unscored: Any,
        remap: list[int] | int,
    ) -> None:
        nonlocal score_task
        unscored_groups.append((unscored, remap))
        if not per_group_scoring:
            return
        if mode is RewardCollectionMode.PER_GROUP_SERIAL:
            # Control arm: same per-group call granularity as streaming, but the
            # score completes before the next generation starts. The measured
            # difference against streaming is overlap alone.
            take_single(await score_unscored([unscored]))
            return
        # Generation of this group ran while the previous scoring task was in
        # flight. Drain it before starting this group's task: at most one reward
        # call can own service/model state at a time.
        await drain_score_task()
        score_task = asyncio.create_task(
            score_unscored([unscored]),
            name="rollout-reward-score",
        )

    async def flush_pending_prompts() -> None:
        if not pending_prompts:
            return
        unscored = await collect_unscored(
            [GenerationInput(prompt=prompt) for prompt in pending_prompts],
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        await record_unscored(unscored, list(pending_indices))
        pending_prompts.clear()
        pending_indices.clear()

    try:
        for prompt_idx, item in enumerate(prompts):
            if not isinstance(item, (str, bytes)) and hasattr(item, "generation_input"):
                await flush_pending_prompts()
                # The example itself owns the field mapping (generation_input /
                # reward_metadata) — no untyped kwargs relay in between.
                unscored = await collect_unscored(
                    [item.generation_input()],
                    group_size=group_size,
                    metadata=item.reward_metadata(),
                    request_overrides=dict(item.request_overrides or {}),
                    runtime_debug=runtime_debug,
                    policy_version=policy_version,
                )
                await record_unscored(unscored, prompt_idx)
            else:
                pending_prompts.append(str(item))
                pending_indices.append(prompt_idx)

        await flush_pending_prompts()
        if not unscored_groups:
            return []

        if per_group_scoring:
            # PER_GROUP_SERIAL already drained inline; only streaming can still
            # own a task here.
            await drain_score_task()
            batches = scored_batches
        else:
            batches = await score_unscored(
                [unscored for unscored, _ in unscored_groups],
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

    if stats is not None:
        # Per-call phases live on the unscored groups (collector writes the
        # call-level score/build timings and reward inference timings on the
        # first group only).
        for unscored, _ in unscored_groups:
            stats.add_phases(getattr(unscored, "phases", {}))
            reward_timing_ms = getattr(unscored, "reward_timing_ms", {}) or {}
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
    for batch, (_, remap) in zip(batches, unscored_groups, strict=True):
        global_prompt_indices = remap if isinstance(remap, list) else [remap]
        remap_group_ids_(batch, global_prompt_indices)
        all_batches.extend(split_batch_by_group(batch))
    if stats is not None:
        collection_wall_s = time.perf_counter() - collection_started
        generation_wall_s = sum(end - start for start, end in generation_intervals)
        reward_wall_s = sum(end - start for start, end in reward_intervals)
        overlap_s = _interval_overlap_seconds(generation_intervals, reward_intervals)
        stats.add_phases(
            {
                "collect.wall": collection_wall_s,
                "collect.generation_wall": generation_wall_s,
                "collect.reward_wall": reward_wall_s,
                "collect.generation_reward_overlap": overlap_s,
            },
        )
        stats.add_counter("collect.group_count", len(all_batches))
        stats.add_counter(
            "collect.sample_count",
            sum(int(batch.rewards.shape[0]) for batch in all_batches),
        )
    return all_batches


def _interval_overlap_seconds(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    """Return the wall time covered by both ordered, non-overlapping timelines."""

    left_intervals = sorted(left)
    right_intervals = sorted(right)
    left_index = 0
    right_index = 0
    overlap = 0.0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


__all__ = [
    "PromptCollectionCleanupError",
    "collect_prompt_batches",
]
