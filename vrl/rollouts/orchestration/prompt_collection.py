"""Prompt collection helpers for rollout schedules."""

from __future__ import annotations

from typing import Any

from vrl.generation import GenerationInput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import remap_group_ids_, split_batch_by_group
from vrl.utils.stats import RolloutStats


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

    Two phases: generate every prompt group first (the generation runtime stays
    resident throughout), then score all groups through one reward call. Shared
    single-GPU reward runs therefore pay the rollout release and the reward
    actor cold start once per call instead of once per group.

    ``stats`` (when given) accumulates this call's collect phase timings
    (``collect.engine_generate`` / ``collect.reward_score`` /
    ``collect.batch_build``) plus any reward-inference timings. The accumulator
    is owned by this call, so concurrent collects never overwrite each other.
    """

    # (unscored group, group-id remap: per-sample indices for plain-string
    # batches, a single prompt index for PromptExample groups)
    unscored_groups: list[tuple[Any, list[int] | int]] = []
    pending_prompts: list[str] = []
    pending_indices: list[int] = []

    async def flush_pending_prompts() -> None:
        if not pending_prompts:
            return
        unscored = await collector.collect_unscored(
            [GenerationInput(prompt=prompt) for prompt in pending_prompts],
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        unscored_groups.append((unscored, list(pending_indices)))
        pending_prompts.clear()
        pending_indices.clear()

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
            unscored_groups.append((unscored, prompt_idx))
        else:
            pending_prompts.append(str(item))
            pending_indices.append(prompt_idx)

    await flush_pending_prompts()
    if not unscored_groups:
        return []

    batches = await collector.score_rollouts(
        [unscored for unscored, _ in unscored_groups],
    )

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


__all__ = ["collect_prompt_batches"]
