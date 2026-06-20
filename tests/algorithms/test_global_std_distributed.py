"""DDP correctness for ``global_std`` advantage normalization.

Under DDP each rank holds only its own prompt slice, so a naive
``rewards.std()`` is per-rank, not global. ``_population_std_across_ranks``
all-reduces the sufficient statistics so ``global_std=True`` uses the true
cross-rank std. These tests spawn a real (gloo, CPU) 2-rank process group and
assert every rank computes the std over the *concatenated* rewards — and that
this differs from what either rank would get from its local slice alone.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.algorithms.advantages import (
    _population_std_across_ranks,
    group_relative_advantages,
)

# rank0 and rank1 hold deliberately different-scale slices so the local stds are
# far apart and a per-rank bug would be obvious.
_RANK_REWARDS = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        local = torch.tensor(_RANK_REWARDS[rank])
        std = float(_population_std_across_ranks(local))
        # also exercise the full advantage path: one group per rank, global_std on
        gids = torch.zeros(local.numel(), dtype=torch.long)
        adv = group_relative_advantages(
            local, gids, eps=1e-8, adv_clip_max=10.0, global_std=True
        )
        # advantage = (r - local_group_mean) / global_std → recover the denominator
        recovered_denom = float((local[0] - local.mean()) / adv[0])
        q.put((rank, std, recovered_denom))
    finally:
        dist.destroy_process_group()


def test_global_std_is_cross_rank_not_per_rank() -> None:
    all_rewards = torch.tensor([r for slice_ in _RANK_REWARDS for r in slice_])
    expected_global = float(all_rewards.std(unbiased=False))
    local_std_rank0 = float(torch.tensor(_RANK_REWARDS[0]).std(unbiased=False))
    local_std_rank1 = float(torch.tensor(_RANK_REWARDS[1]).std(unbiased=False))
    # sanity: the global std must be meaningfully different from either local std
    assert abs(expected_global - local_std_rank0) > 1.0
    assert abs(expected_global - local_std_rank1) > 1.0

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_run_rank, args=(rank, 2, 29555, q)) for rank in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, std, denom = q.get(timeout=50)
        results[rank] = (std, denom)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0, f"rank process exited with {p.exitcode}"

    # both ranks must agree on the SAME, TRUE global std (not their local one)
    for rank in (0, 1):
        std, denom = results[rank]
        assert std == pytest.approx(expected_global, rel=1e-5), (
            f"rank{rank} std={std} != global {expected_global}"
        )
        assert denom == pytest.approx(expected_global, rel=1e-4), (
            f"rank{rank} advantage denom={denom} != global {expected_global}"
        )


def test_global_std_falls_back_to_local_without_process_group() -> None:
    # No process group (single-GPU / unit-test path): local population std.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(_population_std_across_ranks(rewards)) == pytest.approx(
        float(rewards.std(unbiased=False))
    )
