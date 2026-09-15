"""Training exchanges must preserve token/head layout and distributed gradients."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vrl.trainers.distributed import (
    context_parallel_heads_to_tokens,
    context_parallel_tokens_to_heads,
)


def _worker(rank, world, rendezvous, use_cuda=False):
    torch.set_num_threads(1)
    if use_cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        if world == 4:
            even = dist.new_group([0, 2])
            odd = dist.new_group([1, 3])
            group = even if rank % 2 == 0 else odd
            seed = 100 + rank % 2
        else:
            group, seed = dist.group.WORLD, 100
        local_rank = dist.get_rank(group)
        torch.manual_seed(seed)
        full = torch.randn(2, 4, 6, 3, dtype=torch.float64, device=device)
        local = full.chunk(2, dim=2)[local_rank].detach().requires_grad_()
        heads = context_parallel_tokens_to_heads(local, group=group)
        torch.testing.assert_close(heads, full.chunk(2, dim=1)[local_rank], rtol=0, atol=0)
        result = context_parallel_heads_to_tokens(heads, group=group)
        torch.testing.assert_close(result, local, rtol=0, atol=0)
        weights = torch.randn_like(full)
        (result * weights.chunk(2, dim=2)[local_rank]).sum().backward()
        torch.testing.assert_close(local.grad, weights.chunk(2, dim=2)[local_rank], rtol=0, atol=0)

        reference = [torch.randn_like(full).requires_grad_() for _ in range(3)]
        shards = [
            value.detach().chunk(2, dim=2)[local_rank].requires_grad_() for value in reference
        ]
        expected = F.scaled_dot_product_attention(*reference)
        exchanged = [context_parallel_tokens_to_heads(value, group=group) for value in shards]
        actual = context_parallel_heads_to_tokens(
            F.scaled_dot_product_attention(*exchanged), group=group
        )
        torch.testing.assert_close(
            actual, expected.chunk(2, dim=2)[local_rank], rtol=1e-10, atol=1e-10
        )
        (expected * weights).sum().backward()
        (actual * weights.chunk(2, dim=2)[local_rank]).sum().backward()
        for shard, value in zip(shards, reference, strict=True):
            torch.testing.assert_close(
                shard.grad, value.grad.chunk(2, dim=2)[local_rank], rtol=1e-10, atol=1e-10
            )

        with pytest.raises(ValueError, match="heads must be divisible"):
            context_parallel_tokens_to_heads(torch.zeros(1, 3, 2, 2), group=group)
        with pytest.raises(ValueError, match="token count must be divisible"):
            context_parallel_heads_to_tokens(torch.zeros(1, 2, 3, 2), group=group)
        with pytest.raises(ValueError, match="nonempty"):
            context_parallel_tokens_to_heads(torch.zeros(1, 2, 0, 2), group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_context_parallel_exchange_and_attention_backward(tmp_path, world):
    mp.spawn(_worker, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)


def test_context_parallel_exchange_requires_process_group():
    if dist.is_initialized():
        pytest.skip("test requires no ambient process group")
    with pytest.raises(RuntimeError, match="initialized process group"):
        context_parallel_tokens_to_heads(torch.zeros(1, 2, 2, 2), group=None)


@pytest.mark.distributed
def test_context_parallel_exchange_nccl(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_worker, args=(2, (tmp_path / "nccl-rendezvous").as_uri(), True), nprocs=2, join=True)
