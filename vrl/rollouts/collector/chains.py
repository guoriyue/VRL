"""Collect edit chains: one prompt group per declared step, each conditioned on the last.

Step ``k`` of a chain is an ordinary one-shot prompt group whose reference image
is one sample drawn from step ``k-1``. The parent is drawn uniformly from the
driver RNG, never selected by the reward, so later steps train on the policy's
own state distribution rather than on judge-picked prefixes. Every step is
scored on its own conditioning (the parent image), so the editor is paid for
doing the requested edit and for keeping the parent's content, including the
edits made before it. No terminal reward is propagated backwards: this is
per-step credit under a fixed schedule, not a learned multi-step policy.

Generation is sequential within a chain and scoring happens once for every
group at the end, like ``RewardCollectionMode.BATCHED_SERIAL``.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.edit_chains import EditChain
from vrl.utils.media import write_png
from vrl.utils.media_reference import MediaReference

if TYPE_CHECKING:
    from vrl.rollouts.collector.core import RolloutCollector


def sample_image(output: Any, index: int) -> torch.Tensor:
    """One decoded image from a generation output: a tensor batch or boxed references."""

    media = output[index]
    if isinstance(media, MediaReference):
        media = media.resolve()
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"edit chain parent must be an image tensor, got {type(media).__name__}")
    if media.ndim == 4 and media.shape[1] == 1:
        # A one-frame [C, T, H, W] image family output.
        media = media[:, 0]
    if media.ndim != 3:
        raise ValueError(f"edit chain parent must be one image, got shape {tuple(media.shape)}")
    return media


async def collect_edit_chains(
    collector: RolloutCollector,
    chains: list[EditChain],
    *,
    group_size: int,
    runtime_debug: bool,
    policy_version: int | None,
    stats: RolloutStats,
) -> list[RolloutBatch]:
    """Generate every step of every chain, then score all groups in one reward call."""

    from vrl.rollouts.collector.core import RolloutGenerationResult

    media_dir = collector.config.edit_chain_media_dir
    if media_dir is None:
        raise ValueError(
            "edit chain collection needs edit_chain_media_dir (derived from trainer.output_dir)"
        )
    collection_started = time.perf_counter()
    generated: list[RolloutGenerationResult] = []
    for chain in chains:
        parent, parent_sample_id = chain.source, None
        for step in range(len(chain.steps)):
            example = chain.step_example(step, parent, parent_sample_id=parent_sample_id)
            request = collector.request_builder.build(
                [example.generation_input()],
                group_size=group_size,
                metadata=example.reward_metadata(),
                request_overrides=dict(example.request_overrides or {}),
                runtime_debug=runtime_debug,
                policy_version=policy_version,
                reward_media_refs=True,
            )
            started = time.perf_counter()
            unscored = await collector.generate_rollout(request)
            # Group ids only need to be distinct within this call; the trainer
            # and the continuous consumer renumber them.
            generated.append(
                RolloutGenerationResult(unscored, [len(generated)], started, time.perf_counter())
            )
            if step + 1 == len(chain.steps):
                continue
            rows = unscored.output.sample_rows
            if len(rows) != group_size:
                raise RuntimeError(
                    f"edit chain step returned {len(rows)} samples for a group of {group_size}"
                )
            # The driver RNG is checkpointed with the run, so a resumed run
            # branches from the same samples as an uninterrupted one.
            chosen = random.randrange(group_size)
            row = rows[chosen]
            path = media_dir / chain.chain_id / f"step{step:02d}-{request.request.request_id}.png"
            write_png(sample_image(unscored.output.output, chosen), path)
            parent, parent_sample_id = str(path), row.sample_id

    reward_started = time.perf_counter()
    batches = collector.assemble_training_batches(
        await collector.evaluate_rollout([group.unscored for group in generated])
    )
    reward_wall = time.perf_counter() - reward_started
    all_batches = collector.finish_scored_prompt_groups(generated, batches, stats)
    stats.add_phases(
        {
            "collect.wall": time.perf_counter() - collection_started,
            "collect.generation_wall": sum(
                group.completed_at - group.started_at for group in generated
            ),
            "collect.reward_wall": reward_wall,
            "collect.generation_reward_overlap": 0.0,
        }
    )
    stats.add_counter("collect.group_count", len(all_batches))
    stats.add_counter(
        "collect.sample_count", sum(int(batch.rewards.shape[0]) for batch in all_batches)
    )
    return all_batches


__all__ = ["collect_edit_chains", "sample_image"]
