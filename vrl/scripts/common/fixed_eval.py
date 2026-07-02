"""Fixed (held-out) evaluation over a frozen prompt+seed grid, as a global phase.

Extracted from ``online.py``: a self-contained held-out-eval concern driven
entirely by the ``collector`` / ``reward_fn`` / ``training_context`` that
``run_online_recipe``'s ``_fixed_eval_and_log`` hook passes in. It reuses the
training sampling path but never touches the trainer (no collect_training_batch,
backward, optimizer/EMA, or CSV write), so it lives beside the recipe runner as
its own module rather than inline in it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch

from vrl.trainers.distributed import DistributedTrainingContext


@dataclass(frozen=True, slots=True)
class _FixedEvalResult:
    """Aggregated fixed-eval reward over the held-out prompt+seed grid (GLOBAL).

    Under multi-rank eval this is the all-reduced result every rank sees after
    ``_merge_fixed_eval_stats``: reward_mean/std/stderr and component_means are
    over the WHOLE eval set, not this rank's local shard. rank0 writes it to CSV.
    """

    reward_mean: float
    reward_std: float
    reward_stderr: float
    n: int
    component_means: dict[str, float]


@dataclass(frozen=True, slots=True)
class _FixedEvalLocalStats:
    """One rank's reward sufficient statistics over its eval prompt shard.

    Sum / sum-of-squares / count (not mean/std) so ``_merge_fixed_eval_stats`` can
    all-reduce SUM across ranks and recover the global population mean/std exactly —
    averaging per-rank means would be wrong for unequal shard sizes. Component
    stats are summed and counted the same way (§3.3): a mean of per-rank means is
    NOT valid, so we carry sum+count per component and divide after the all-reduce.
    """

    reward_sum: float
    reward_sumsq: float
    reward_count: int
    component_sums: dict[str, float]
    component_counts: dict[str, int]


def _iter_fixed_eval_shard(
    examples: list[Any],
    max_prompts: int,
    rank: int,
    world_size: int,
) -> list[tuple[int, Any]]:
    """This rank's disjoint slice of the eval grid as ``(global_index, item)``.

    Truncates to ``max_prompts`` in the GLOBAL prompt order first, then strides by
    ``rank`` (every ``world_size``-th prompt). The returned ``global_index`` — used
    to derive the seed — is the index in the truncated global list, NOT a
    shard-local index, so the same prompt keeps the same seed regardless of
    ``world_size`` (fixed eval stays fixed when the GPU count changes; §3.1). The
    union over all ranks is exactly the truncated eval set, and shards are disjoint.
    """

    all_examples = examples[:max_prompts] if max_prompts > 0 else list(examples)
    return [
        (global_index, item)
        for global_index, item in enumerate(all_examples)
        if global_index % world_size == rank
    ]


def _fixed_eval_group_seed(base_seed: int, global_index: int, samples_per_prompt: int) -> int:
    """Deterministic per-prompt seed on the fixed grid (world_size-independent).

    Keyed by the GLOBAL eval index so a prompt's seed never moves when the rank
    count changes; identical to the original rank0-only grid (base + i*samples).
    """

    return int(base_seed) + int(global_index) * int(samples_per_prompt)


def _fixed_eval_collect_kwargs(item: Any, *, group_size: int, seed: int) -> dict[str, Any]:
    """Collect kwargs for one eval prompt group at a fixed seed.

    Mirrors prompt_collection._prompt_example_kwargs so image/caption/reference
    eval prompts carry their conditioning, and adds the deterministic ``seed``
    the request builder threads into sampling (requests.py).
    """

    kwargs: dict[str, Any] = {"group_size": int(group_size), "seed": int(seed)}
    if not isinstance(item, str):
        for key, attr in (
            ("target_text", "target_text"),
            ("references", "references"),
            ("task_type", "task_type"),
            ("request_overrides", "request_overrides"),
            ("sample_metadata", "metadata"),
        ):
            value = getattr(item, attr, None)
            if value is not None:
                kwargs[key] = value
        reference_image = getattr(item, "reference_image", None)
        if reference_image:
            kwargs["reference_image"] = reference_image
        reference_video = getattr(item, "reference_video", None)
        if reference_video:
            kwargs["reference_video"] = reference_video
        target_image = getattr(item, "target_image", None)
        if target_image:
            kwargs["target_image"] = target_image
        target_video = getattr(item, "target_video", None)
        if target_video:
            kwargs["target_video"] = target_video
    return kwargs


async def _run_local_fixed_eval(
    collector: Any,
    reward_fn: Any,
    examples: list[Any],
    *,
    samples_per_prompt: int,
    base_seed: int,
    max_prompts: int,
    component_names: tuple[str, ...],
    rank: int,
    world_size: int,
) -> _FixedEvalLocalStats:
    """Score THIS rank's eval prompt shard — local reward sufficient statistics.

    Reuses the training ``collector``/runtime/``reward_fn`` (same sampling path as
    training) but does NOT touch the trainer: no collect_training_batch, no
    backward, no optimizer/EMA/previous-policy sync, no trainer.prompts mutation,
    no CSV write. Each prompt gets ``samples_per_prompt`` videos at a deterministic
    seed derived from its GLOBAL index, so the grid reproduces across epochs,
    resumes, and rank counts. An empty shard (max_prompts < world_size) returns
    zero-count stats and still participates in the caller's all-reduce (§6.2).
    Rollout/reward artifacts are released before returning to avoid host-RAM creep.
    """

    local_examples = _iter_fixed_eval_shard(examples, max_prompts, rank, world_size)

    # Snapshot only this eval's reward components, not training's.
    reward_fn.reset_components()
    unscored: list[Any] = []
    batches: list[Any] = []
    try:
        for global_index, item in local_examples:
            seed = _fixed_eval_group_seed(base_seed, global_index, samples_per_prompt)
            prompt = str(getattr(item, "prompt", item))
            unscored.append(
                await collector.collect_unscored(
                    [prompt],
                    **_fixed_eval_collect_kwargs(item, group_size=samples_per_prompt, seed=seed),
                ),
            )
        if unscored:
            batches = await collector.score_rollouts(unscored)
            rewards = torch.cat(
                [b.rewards.detach().double().reshape(-1).cpu() for b in batches],
            )
        else:
            rewards = torch.zeros(0, dtype=torch.float64)
        reward_count = int(rewards.numel())
        reward_sum = float(rewards.sum().item()) if reward_count else 0.0
        reward_sumsq = float(rewards.mul(rewards).sum().item()) if reward_count else 0.0
        last = getattr(reward_fn, "last_components", {}) or {}
        component_sums = {
            name: float(sum(last.get(name) or [])) for name in component_names
        }
        component_counts = {
            name: len(last.get(name) or []) for name in component_names
        }
        return _FixedEvalLocalStats(
            reward_sum=reward_sum,
            reward_sumsq=reward_sumsq,
            reward_count=reward_count,
            component_sums=component_sums,
            component_counts=component_counts,
        )
    finally:
        # Drop rollout/reward artifact refs before the next training epoch.
        del batches, unscored
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _merge_fixed_eval_stats(
    local: _FixedEvalLocalStats,
    *,
    component_names: tuple[str, ...],
    device: torch.device,
) -> _FixedEvalResult:
    """All-reduce SUM the per-rank stats into the GLOBAL fixed-eval result.

    Packs reward sum/sumsq/count and per-component sum/count into one float64
    tensor and reduces once (mirrors ``_global_reward_stats`` in the trainer). With
    no process group / world_size==1 the local stats already ARE the global stats,
    so the reduce is skipped and the SAME arithmetic runs — single-rank output is
    unchanged (P3). Population std from the summed moments (var = E[x^2] - E[x]^2),
    stderr = std / sqrt(count); component means are global_sum / global_count, NOT
    a mean of per-rank means (§3.3). Must be called on every rank with the same
    ``component_names`` so the collective stays balanced.
    """

    ncomp = len(component_names)
    flat = [local.reward_sum, local.reward_sumsq, float(local.reward_count)]
    flat += [float(local.component_sums.get(name, 0.0)) for name in component_names]
    flat += [float(local.component_counts.get(name, 0)) for name in component_names]
    stats = torch.tensor(flat, dtype=torch.float64)

    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # NCCL collectives require GPU tensors; gloo (tests/CPU) reduces CPU directly.
        if dist.get_backend() == "nccl":
            stats = stats.to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    stats = stats.cpu()

    g_sum = float(stats[0])
    g_sumsq = float(stats[1])
    g_count = float(stats[2])
    n = round(g_count)
    if n <= 0:
        mean, std, stderr = float("nan"), 0.0, 0.0
    elif n == 1:
        mean, std, stderr = g_sum / g_count, 0.0, 0.0
    else:
        mean = g_sum / g_count
        var = g_sumsq / g_count - mean * mean
        std = float(max(var, 0.0) ** 0.5)
        stderr = std / (g_count**0.5)

    comp_sums = stats[3 : 3 + ncomp]
    comp_counts = stats[3 + ncomp : 3 + 2 * ncomp]
    component_means = {
        name: (float(comp_sums[i]) / float(comp_counts[i]))
        if float(comp_counts[i]) > 0
        else float("nan")
        for i, name in enumerate(component_names)
    }
    return _FixedEvalResult(mean, std, stderr, n, component_means)


async def _run_distributed_fixed_eval(
    collector: Any,
    reward_fn: Any,
    eval_examples: list[Any],
    *,
    samples_per_prompt: int,
    base_seed: int,
    max_prompts: int,
    component_names: tuple[str, ...],
    training_context: DistributedTrainingContext,
) -> _FixedEvalResult:
    """Run fixed eval as a GLOBAL phase: every rank scores a disjoint prompt shard.

    Controller-free disaggregation shape (§0/§2): the rollout side (every training
    rank's colocated runtime) splits the eval generation/reward work by global
    index, then a single stats all-reduce gives every rank the same global
    ``_FixedEvalResult`` — the learning signal over the WHOLE eval set, not rank0's
    local slice. The caller writes/logs it on rank0 only. MUST be called on EVERY
    rank (the all-reduce inside ``_merge_fixed_eval_stats`` is collective); an empty
    shard still participates with count=0 (§6.2). Walks only the rollout collector
    path, never the FSDP/DDP model forward, so it issues no trainer-replay
    collectives that could interleave with the stats reduce (§6.4). An exception on
    one rank propagates here (and is logged by the run's error path) but will leave
    peers waiting at the reduce — acceptable for v1; no fault-tolerance yet (§6.3).
    """

    truncated = eval_examples[:max_prompts] if max_prompts > 0 else eval_examples
    if not truncated:
        raise ValueError(
            "fixed eval has no prompts (check data.eval_manifest and trainer.eval.max_prompts)",
        )
    local = await _run_local_fixed_eval(
        collector,
        reward_fn,
        list(eval_examples),
        samples_per_prompt=samples_per_prompt,
        base_seed=base_seed,
        max_prompts=max_prompts,
        component_names=component_names,
        rank=training_context.rank,
        world_size=training_context.world_size,
    )
    return _merge_fixed_eval_stats(
        local,
        component_names=component_names,
        device=training_context.device,
    )


__all__ = [
    "_FixedEvalLocalStats",
    "_FixedEvalResult",
    "_fixed_eval_collect_kwargs",
    "_fixed_eval_group_seed",
    "_iter_fixed_eval_shard",
    "_merge_fixed_eval_stats",
    "_run_distributed_fixed_eval",
    "_run_local_fixed_eval",
]
