"""Fixed eval is a GLOBAL, rank-sharded phase — not a rank0-only bypass.

Three layers, matching SPRINT_distributed_fixed_eval.md §5:

* ``_iter_fixed_eval_shard`` / ``_fixed_eval_group_seed`` — pure sharding: every
  eval prompt lands on exactly one rank, the union is the whole (truncated) eval
  set, and a prompt's seed is keyed by its GLOBAL index so it does NOT move when
  the GPU count changes (fixed eval stays fixed).
* ``_merge_fixed_eval_stats`` — the global mean/std/stderr and component means come
  from summed sufficient statistics, never a mean of per-rank means.
* a real 2-rank gloo group — the all-reduce wiring gives every rank the SAME
  global ``_FixedEvalResult``, distinct from either rank's local slice.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.scripts.common.online import (
    _fixed_eval_group_seed,
    _FixedEvalLocalStats,
    _FixedEvalResult,
    _iter_fixed_eval_shard,
    _merge_fixed_eval_stats,
)

# --------------------------------------------------------------------------- #
# §5.1 pure sharding
# --------------------------------------------------------------------------- #


def test_shard_is_disjoint_and_covers_the_eval_set() -> None:
    examples = [f"p{i}" for i in range(10)]
    world_size = 3
    shards = {
        rank: _iter_fixed_eval_shard(examples, 0, rank, world_size)
        for rank in range(world_size)
    }

    # Exact split from the sprint: rank r owns every world_size-th prompt.
    assert [gi for gi, _ in shards[0]] == [0, 3, 6, 9]
    assert [gi for gi, _ in shards[1]] == [1, 4, 7]
    assert [gi for gi, _ in shards[2]] == [2, 5, 8]

    all_indices = [gi for rank in range(world_size) for gi, _ in shards[rank]]
    # Disjoint (no prompt evaluated twice) and a complete cover of the eval set.
    assert sorted(all_indices) == list(range(10))
    assert len(all_indices) == len(set(all_indices))

    # The item travels with its global index, not a shard-local one.
    assert shards[0][1] == (3, "p3")


def test_max_prompts_truncates_in_global_order_before_sharding() -> None:
    examples = [f"p{i}" for i in range(10)]
    shards = [_iter_fixed_eval_shard(examples, 4, rank, 2) for rank in range(2)]
    assert [gi for gi, _ in shards[0]] == [0, 2]
    assert [gi for gi, _ in shards[1]] == [1, 3]
    covered = sorted(gi for s in shards for gi, _ in s)
    assert covered == [0, 1, 2, 3]  # only the first max_prompts prompts


def test_empty_shard_when_max_prompts_below_world_size() -> None:
    # max_prompts=2 < world_size=3: rank2 gets nothing but must still be callable.
    examples = [f"p{i}" for i in range(10)]
    assert _iter_fixed_eval_shard(examples, 2, 2, 3) == []
    assert [gi for gi, _ in _iter_fixed_eval_shard(examples, 2, 0, 3)] == [0]
    assert [gi for gi, _ in _iter_fixed_eval_shard(examples, 2, 1, 3)] == [1]


def test_seed_depends_only_on_global_index_not_world_size() -> None:
    # The same prompt (global index 4) must get the same seed at any world size.
    base_seed, samples = 100, 2
    expected = base_seed + 4 * samples
    assert _fixed_eval_group_seed(base_seed, 4, samples) == expected

    examples = [f"p{i}" for i in range(10)]
    for world_size in (1, 2, 3, 4):
        for rank in range(world_size):
            for global_index, _ in _iter_fixed_eval_shard(examples, 0, rank, world_size):
                seed = _fixed_eval_group_seed(base_seed, global_index, samples)
                assert seed == base_seed + global_index * samples


# --------------------------------------------------------------------------- #
# §5.2 aggregation arithmetic (sum/count, never mean-of-means)
# --------------------------------------------------------------------------- #


def test_merge_uses_summed_moments_not_mean_of_means() -> None:
    # rank0 rewards [1, 3], rank1 rewards [5]. The combined sufficient statistics
    # (what the all-reduce SUM produces) are what merge must consume.
    combined = _FixedEvalLocalStats(
        reward_sum=1.0 + 3.0 + 5.0,
        reward_sumsq=1.0 + 9.0 + 25.0,
        reward_count=3,
        # component "kling": rank0 sum=4 count=2, rank1 sum=5 count=1.
        component_sums={"kling": 9.0},
        component_counts={"kling": 3},
    )
    result = _merge_fixed_eval_stats(
        combined,
        component_names=("kling",),
        device=torch.device("cpu"),
    )

    assert result.n == 3
    assert result.reward_mean == pytest.approx(3.0)
    pop_std = math.sqrt(((1 - 3) ** 2 + (3 - 3) ** 2 + (5 - 3) ** 2) / 3)
    assert result.reward_std == pytest.approx(pop_std)
    assert result.reward_stderr == pytest.approx(pop_std / math.sqrt(3))
    # global_sum/global_count = 9/3 = 3, NOT mean-of-means (2 + 5)/2 = 3.5.
    assert result.component_means["kling"] == pytest.approx(3.0)
    assert result.component_means["kling"] != pytest.approx(3.5)


def test_merge_empty_eval_yields_nan_mean_zero_spread() -> None:
    empty = _FixedEvalLocalStats(0.0, 0.0, 0, {"kling": 0.0}, {"kling": 0})
    result = _merge_fixed_eval_stats(
        empty,
        component_names=("kling",),
        device=torch.device("cpu"),
    )
    assert result.n == 0
    assert math.isnan(result.reward_mean)
    assert result.reward_std == 0.0
    assert result.reward_stderr == 0.0
    assert math.isnan(result.component_means["kling"])


# --------------------------------------------------------------------------- #
# §5.3 real 2-rank gloo all-reduce
# --------------------------------------------------------------------------- #

# rank0 evaluates eval indices {0, 2} -> rewards [1, 3]; rank1 -> {1, 3} -> [5, 7].
_RANK_LOCAL = [
    _FixedEvalLocalStats(
        reward_sum=1.0 + 3.0,
        reward_sumsq=1.0 + 9.0,
        reward_count=2,
        component_sums={"kling": 1.0 + 3.0},
        component_counts={"kling": 2},
    ),
    _FixedEvalLocalStats(
        reward_sum=5.0 + 7.0,
        reward_sumsq=25.0 + 49.0,
        reward_count=2,
        component_sums={"kling": 5.0 + 7.0},
        component_counts={"kling": 2},
    ),
]


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        result = _merge_fixed_eval_stats(
            _RANK_LOCAL[rank],
            component_names=("kling",),
            device=torch.device("cpu"),
        )
        q.put((rank, result.reward_mean, result.reward_std, result.reward_stderr,
               result.n, result.component_means["kling"]))
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_get_identical_global_result() -> None:
    all_rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    g_mean = float(all_rewards.mean())
    g_std = float(all_rewards.std(unbiased=False))
    g_stderr = g_std / math.sqrt(all_rewards.numel())

    # The global view must differ from either rank's local slice.
    assert abs(float(torch.tensor([1.0, 3.0]).mean()) - g_mean) > 1.0
    assert abs(float(torch.tensor([5.0, 7.0]).mean()) - g_mean) > 1.0

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, 29571, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, mean, std, stderr, n, kling = q.get(timeout=60)
        results[rank] = (mean, std, stderr, n, kling)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    # Both ranks computed the SAME global metrics over all 4 eval samples.
    for rank in (0, 1):
        mean, std, stderr, n, kling = results[rank]
        assert n == 4
        assert mean == pytest.approx(g_mean, rel=1e-6)
        assert std == pytest.approx(g_std, rel=1e-6)
        assert stderr == pytest.approx(g_stderr, rel=1e-6)
        assert kling == pytest.approx(g_mean, rel=1e-6)
    assert results[0] == results[1]


def test_single_rank_merge_matches_local_population_stats() -> None:
    """No process group: merge falls back to local stats == global (P3)."""
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    local = _FixedEvalLocalStats(
        reward_sum=float(rewards.sum()),
        reward_sumsq=float(rewards.mul(rewards).sum()),
        reward_count=int(rewards.numel()),
        component_sums={"kling": float(rewards.sum())},
        component_counts={"kling": int(rewards.numel())},
    )
    result = _merge_fixed_eval_stats(
        local,
        component_names=("kling",),
        device=torch.device("cpu"),
    )
    assert isinstance(result, _FixedEvalResult)
    assert result.reward_mean == pytest.approx(float(rewards.mean()))
    assert result.reward_std == pytest.approx(float(rewards.std(unbiased=False)))
