"""The logged reward mean/std must be cross-rank, not rank0-local.

Each DDP rank holds only its own prompt slice, and rank0 alone writes the metric
row — so a plain ``rewards.mean()`` logs a per-rank number, half the true
32-prompt objective. ``_global_reward_stats`` all-reduces the sufficient
statistics. This spawns a real gloo 2-rank group and asserts every rank reports
the concatenated mean/std, distinct from its local slice.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.trainers.online.trainer import _global_reward_stats

_RANK_REWARDS = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        mean, std = _global_reward_stats(torch.tensor(_RANK_REWARDS[rank]))
        q.put((rank, mean, std))
    finally:
        dist.destroy_process_group()


def test_logged_reward_stats_are_cross_rank() -> None:
    allr = torch.tensor([r for s in _RANK_REWARDS for r in s])
    g_mean = float(allr.mean())
    g_std = float(allr.std(unbiased=False))
    # the global mean/std must differ from either rank's local view
    for s in _RANK_REWARDS:
        t = torch.tensor(s)
        assert abs(float(t.mean()) - g_mean) > 1.0

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, 29566, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, mean, std = q.get(timeout=50)
        results[rank] = (mean, std)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    for rank in (0, 1):
        mean, std = results[rank]
        assert mean == pytest.approx(g_mean, rel=1e-5)
        assert std == pytest.approx(g_std, rel=1e-5)


def test_falls_back_to_local_without_process_group() -> None:
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mean, std = _global_reward_stats(r)
    assert mean == pytest.approx(float(r.mean()))
    assert std == pytest.approx(float(r.std()))
