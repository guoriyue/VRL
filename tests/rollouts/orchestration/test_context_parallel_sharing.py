"""CP peers receive one owned rollout; DP peers retain independent samples."""

import asyncio
import random
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.schedule import collect_context_parallel_iteration
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.trainers.data.prompt_sampler import PromptBatchSampler
from vrl.trainers.distributed import create_context_parallel_groups
from vrl.trajectory import TrajectoryBatch


def _worker(rank, rendezvous, spool_dir, cuda):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        groups = create_context_parallel_groups(2)
        sampler = PromptBatchSampler(
            generator=torch.Generator().manual_seed(551),
            num_examples=16,
            prompts_per_rank=2,
            num_replicas=groups.dp_size,
            rank=groups.dp_rank,
            strategy="random_without_replacement",
        )
        indices = sampler.sample()
        indices_by_rank = [None] * 4
        dist.all_gather_object(indices_by_rank, indices)
        assert indices_by_rank[0] == indices_by_rank[1]
        assert indices_by_rank[2] == indices_by_rank[3]
        assert set(indices_by_rank[0]).isdisjoint(indices_by_rank[2])
        calls = 0
        seed = 100 + rank
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        random.seed(seed)
        np.random.seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        sampler_state = sampler.generator.get_state().clone()

        async def collect():
            nonlocal calls
            calls += 1
            assert groups.cp_rank == 0
            torch.rand(3)
            random.random()
            np.random.rand(3)
            if cuda:
                torch.rand(3, device=device)
            trajectory = TrajectoryBatch(
                request_id=f"dp-{groups.dp_rank}",
                family="cosmos-predict2.5",
                task="text-to-video",
                sample_rows=[],
                axes={},
                segments={},
                context={
                    "latents": torch.full(
                        (2, 3, 4), float(groups.dp_rank), device=device, dtype=torch.bfloat16
                    )
                },
            )
            return RolloutIteration(
                batches=[
                    RolloutBatch(
                        rewards=torch.tensor(indices, device=device, dtype=torch.float32),
                        group_ids=torch.arange(2, device=device),
                        trajectory=trajectory,
                        context={"indices": indices, "policy_version": 3},
                        extras={"nested": [torch.arange(5, device=device)]},
                    )
                ]
            )

        with (
            patch(
                "torch.cuda.get_rng_state_all", side_effect=AssertionError("foreign CUDA RNG read")
            ),
            patch(
                "torch.cuda.set_rng_state_all",
                side_effect=AssertionError("foreign CUDA RNG write"),
            ),
        ):
            result = asyncio.run(
                collect_context_parallel_iteration(
                    collect if groups.cp_rank == 0 else None, groups=groups, spool_dir=spool_dir
                )
            )
        leader_seed = 100 + groups.dp_rank * 2
        expected_cpu = torch.Generator().manual_seed(leader_seed)
        torch.rand(3, generator=expected_cpu)
        torch.testing.assert_close(
            torch.randperm(16), torch.randperm(16, generator=expected_cpu), rtol=0, atol=0
        )
        expected_python = random.Random(leader_seed)
        expected_python.random()
        assert random.random() == expected_python.random()
        expected_numpy = np.random.RandomState(leader_seed)
        expected_numpy.rand(3)
        np.testing.assert_array_equal(np.random.rand(2), expected_numpy.rand(2))
        if cuda:
            expected_cuda = torch.Generator(device=device).manual_seed(leader_seed)
            torch.rand(3, device=device, generator=expected_cuda)
            torch.testing.assert_close(
                torch.rand(5, device=device),
                torch.rand(5, device=device, generator=expected_cuda),
                rtol=0,
                atol=0,
            )
        assert torch.equal(sampler.generator.get_state(), sampler_state)
        batch = result.batches[0]
        assert batch.context == {"indices": indices, "policy_version": 3}
        torch.testing.assert_close(batch.rewards, torch.tensor(indices, dtype=torch.float32))
        assert batch.trajectory.request_id == f"dp-{groups.dp_rank}"
        assert batch.trajectory.context["latents"].device.type == "cpu"
        assert batch.trajectory.context["latents"].dtype == torch.bfloat16
        assert batch.trajectory.context["latents"].eq(groups.dp_rank).all()
        torch.testing.assert_close(batch.extras["nested"][0], torch.arange(5))
        assert calls == (1 if groups.cp_rank == 0 else 0)

        async def fail_collect():
            if groups.dp_rank == 0:
                raise RuntimeError("owner collection failed")
            return await collect()

        with pytest.raises(RuntimeError, match="owner collection failed"):
            asyncio.run(
                collect_context_parallel_iteration(
                    fail_collect if groups.cp_rank == 0 else None,
                    groups=groups,
                    spool_dir=spool_dir,
                )
            )
        original_load = torch.load

        def load(*args, **kwargs):
            if rank == 1:
                raise OSError("follower read failed")
            return original_load(*args, **kwargs)

        with patch("torch.load", load), pytest.raises(RuntimeError, match="follower read failed"):
            asyncio.run(
                collect_context_parallel_iteration(
                    collect if groups.cp_rank == 0 else None, groups=groups, spool_dir=spool_dir
                )
            )
        before = calls
        with pytest.raises(RuntimeError, match="FileNotFoundError"):
            asyncio.run(
                collect_context_parallel_iteration(
                    collect if groups.cp_rank == 0 else None,
                    groups=groups,
                    spool_dir=Path(spool_dir) / "missing",
                )
            )
        assert calls == before
        dist.barrier()
        assert not list(Path(spool_dir).glob(".cp-rollout-*"))
    finally:
        dist.destroy_process_group()


def test_context_parallel_rollout_sharing_cpu(tmp_path):
    mp.spawn(
        _worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path), False), nprocs=4, join=True
    )


@pytest.mark.distributed
def test_context_parallel_rollout_sharing_nccl(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    mp.spawn(
        _worker, args=((tmp_path / "nccl").as_uri(), str(tmp_path), True), nprocs=4, join=True
    )
