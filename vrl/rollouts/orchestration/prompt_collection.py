"""Prompt collection helpers for rollout schedules."""

from __future__ import annotations

import asyncio
from typing import Any

from vrl.generation import GenerationInput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import remap_group_ids_, split_batch_by_group
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
) -> list[RolloutBatch]:
    """Collect trainer prompts through ``RolloutCollector`` and split by group.

    Collectors without an explicit overlap capability keep two strict phases:
    generate every prompt group, then score all groups through one reward call.
    A capable collector may score group N while generating group N+1. The
    streaming path owns at most one scoring task, so reward work has bounded
    backpressure and deterministic cleanup.

    ``stats`` (when given) accumulates this call's collect phase timings
    (``collect.engine_generate`` / ``collect.reward_score`` /
    ``collect.batch_build``) plus any reward-inference timings. The accumulator
    is owned by this call, so concurrent collects never overwrite each other.
    """

    if not prompts:
        return []

    # (unscored group, group-id remap: per-sample indices for plain-string
    # batches, a single prompt index for PromptExample groups)
    unscored_groups: list[tuple[Any, list[int] | int]] = []
    scored_batches: list[RolloutBatch] = []
    pending_prompts: list[str] = []
    pending_indices: list[int] = []
    # The collector combines topology and reward-runtime execution semantics.
    # Missing capability must preserve the batched serial path: topology alone
    # cannot prove that scoring yields the event loop or retains throughput.
    overlap_reward_scoring = bool(
        getattr(collector, "supports_reward_generation_overlap", False),
    )
    score_task: asyncio.Task[list[RolloutBatch]] | None = None

    async def drain_score_task() -> None:
        nonlocal score_task
        task = score_task
        if task is None:
            return
        # Clear ownership before awaiting so a task failure is not mistaken for
        # a second cleanup failure by the outer exception handler.
        score_task = None
        batches = await task
        if len(batches) != 1:
            raise RuntimeError(
                "streamed reward scoring must return one batch for one unscored group, "
                f"got {len(batches)}",
            )
        scored_batches.extend(batches)

    async def record_unscored(
        unscored: Any,
        remap: list[int] | int,
    ) -> None:
        nonlocal score_task
        unscored_groups.append((unscored, remap))
        if not overlap_reward_scoring:
            return
        # Generation of this group ran while the previous scoring task was in
        # flight. Drain it before starting this group's task: at most one reward
        # call can own service/model state at a time.
        await drain_score_task()
        score_task = asyncio.create_task(
            collector.score_rollouts([unscored]),
            name="rollout-reward-score",
        )

    async def flush_pending_prompts() -> None:
        if not pending_prompts:
            return
        unscored = await collector.collect_unscored(
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
            if _is_prompt_example(item):
                await flush_pending_prompts()
                # The example itself owns the field mapping (generation_input /
                # reward_metadata) — no untyped kwargs relay in between.
                unscored = await collector.collect_unscored(
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

        if overlap_reward_scoring:
            await drain_score_task()
            batches = scored_batches
        else:
            batches = await collector.score_rollouts(
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
                stats.fold_reward_timing(
                    latency_ms=reward_timing_ms.get("latency_ms"),
                    queue_wait_ms=reward_timing_ms.get("queue_wait_ms"),
                    inference_ms=reward_timing_ms.get("inference_ms"),
                )

    all_batches: list[RolloutBatch] = []
    for batch, (_, remap) in zip(batches, unscored_groups, strict=True):
        if isinstance(remap, list):
            remap_group_ids_(batch, remap)
        else:
            batch.group_ids[:] = remap
        all_batches.extend(split_batch_by_group(batch))
    return all_batches


def _is_prompt_example(item: Any) -> bool:
    return not isinstance(item, (str, bytes)) and hasattr(item, "generation_input")


__all__ = ["PromptCollectionCleanupError", "collect_prompt_batches"]
