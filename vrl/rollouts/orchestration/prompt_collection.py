"""Prompt collection helpers for rollout schedules."""

from __future__ import annotations

from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import remap_group_ids_, split_batch_by_group


async def collect_prompt_batches(
    *,
    collector: Any,
    prompts: list[Any],
    group_size: int,
    runtime_debug: bool,
    policy_version: int | None,
) -> list[RolloutBatch]:
    """Collect trainer prompts through ``RolloutCollector`` and split by group."""

    all_batches: list[RolloutBatch] = []
    pending_prompts: list[str] = []
    pending_indices: list[int] = []

    async def flush_pending_prompts() -> None:
        if not pending_prompts:
            return
        collect_kwargs = _base_collect_kwargs(
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        batch = await collector.collect(list(pending_prompts), **collect_kwargs)
        remap_group_ids_(batch, pending_indices)
        all_batches.extend(split_batch_by_group(batch))
        pending_prompts.clear()
        pending_indices.clear()

    for prompt_idx, item in enumerate(prompts):
        if _is_prompt_example(item):
            await flush_pending_prompts()
            collect_kwargs = _prompt_example_kwargs(
                item,
                group_size=group_size,
                runtime_debug=runtime_debug,
                policy_version=policy_version,
            )
            batch = await collector.collect(
                [str(item.prompt)],
                **collect_kwargs,
            )
            batch.group_ids[:] = prompt_idx
            all_batches.extend(split_batch_by_group(batch))
        else:
            pending_prompts.append(str(item))
            pending_indices.append(prompt_idx)

    await flush_pending_prompts()
    return all_batches


def _base_collect_kwargs(
    *,
    group_size: int,
    runtime_debug: bool,
    policy_version: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"group_size": int(group_size)}
    if runtime_debug:
        kwargs["runtime_debug"] = True
    if policy_version is not None:
        kwargs["policy_version"] = int(policy_version)
    return kwargs


def _prompt_example_kwargs(
    item: Any,
    *,
    group_size: int,
    runtime_debug: bool,
    policy_version: int | None,
) -> dict[str, Any]:
    kwargs = {
        **_base_collect_kwargs(
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        ),
        "target_text": getattr(item, "target_text", None),
        "references": getattr(item, "references", None),
        "task_type": getattr(item, "task_type", None),
        "request_overrides": getattr(item, "request_overrides", None),
        "sample_metadata": getattr(item, "metadata", None),
    }
    reference_image = getattr(item, "reference_image", None)
    if reference_image:
        kwargs["reference_image"] = reference_image
    reference_video = getattr(item, "reference_video", None)
    if reference_video:
        kwargs["reference_video"] = reference_video
    return kwargs


def _is_prompt_example(item: Any) -> bool:
    return not isinstance(item, (str, bytes)) and hasattr(item, "prompt")


__all__ = ["collect_prompt_batches"]
