"""Four ranks: DP2 x CP2 must match full-batch accumulated AdamW updates."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock

from vrl.models.families.cosmos import CosmosContextParallelAttnProcessor
from vrl.models.precision import fixed_row_linear_compute
from vrl.trainers.distributed import (
    context_parallel_gather_tokens,
    create_context_parallel_groups,
    reduce_context_parallel_gradients,
)


def _worker(rank, rendezvous, cuda):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        world_size=4,
        rank=rank,
        timeout=timedelta(seconds=120),
    )
    try:
        for invalid in (0, 3, True, 2.5):
            with pytest.raises(ValueError, match="dividing world size"):
                create_context_parallel_groups(invalid)
        groups = create_context_parallel_groups(2)
        assert (groups.dp_rank, groups.cp_rank) == (rank // 2, rank % 2)
        torch.manual_seed(991)
        reference = CosmosTransformerBlock(4, 8, 32).attn1.to(device)
        reference.register_parameter("unused", torch.nn.Parameter(torch.ones(1, device=device)))
        reference.register_parameter(
            "conditional", torch.nn.Parameter(torch.ones(1, device=device))
        )
        candidate = copy.deepcopy(reference)
        candidate.set_processor(CosmosContextParallelAttnProcessor(groups.cp_group))
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-4) for model in (reference, candidate)
        ]
        with fixed_row_linear_compute(reference), fixed_row_linear_compute(candidate):
            for _update in range(2):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                inputs = torch.randn(2, 2, 1, 12, 32, device=device)
                weights = torch.randn_like(inputs)
                for micro in range(2):
                    for dp in range(2):
                        output = reference(inputs[micro, dp])
                        loss = output.mul(weights[micro, dp]).sum()
                        if dp == 1:
                            loss = loss + reference.conditional.square().sum()
                        (loss / 4).backward()
                    local = inputs[micro, groups.dp_rank].chunk(2, dim=1)[groups.cp_rank]
                    output = context_parallel_gather_tokens(
                        candidate(local), group=groups.cp_group
                    )
                    loss = output.mul(weights[micro, groups.dp_rank]).sum()
                    if groups.dp_rank == 1:
                        loss = loss + candidate.conditional.square().sum()
                    (loss / 4).backward()
                reduce_context_parallel_gradients(candidate.parameters(), groups=groups)
                for expected, actual in zip(
                    reference.parameters(), candidate.parameters(), strict=True
                ):
                    if expected.grad is None:
                        assert actual.grad is None
                    else:
                        torch.testing.assert_close(
                            actual.grad, expected.grad, atol=1e-5, rtol=3e-5
                        )
                for optimizer in optimizers:
                    optimizer.step()
                for expected, actual in zip(
                    reference.parameters(), candidate.parameters(), strict=True
                ):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    left, right = (
                        optimizers[0].state.get(expected),
                        optimizers[1].state.get(actual),
                    )
                    if left is None:
                        assert right is None
                    else:
                        for key in left:
                            torch.testing.assert_close(right[key], left[key], atol=2e-6, rtol=5e-5)
        assert candidate.unused.item() == 1
        assert candidate.conditional.item() != 1
        if rank == 0:
            candidate.unused.grad = torch.sparse_coo_tensor([[0]], [1.0], (1,), device=device)
        with pytest.raises(ValueError, match="dense gradients"):
            reduce_context_parallel_gradients([candidate.unused], groups=groups)
    finally:
        dist.destroy_process_group()


def test_dp_cp_accumulated_updates_cpu(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), False), nprocs=4, join=True)


@pytest.mark.distributed
def test_dp_cp_accumulated_updates_nccl(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    mp.spawn(_worker, args=((tmp_path / "nccl").as_uri(), True), nprocs=4, join=True)
