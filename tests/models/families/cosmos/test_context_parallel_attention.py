"""Compare actual Cosmos projections, QK norms and RoPE through CP backward."""

import copy
from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock
from torch.utils.checkpoint import checkpoint

from vrl.models.families.cosmos import CosmosContextParallelSelfAttnProcessor


def _worker(rank, rendezvous, cuda):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(481)
        reference = CosmosTransformerBlock(4, 8, 32).attn1.to(device)
        candidate = copy.deepcopy(reference)
        candidate.set_processor(CosmosContextParallelSelfAttnProcessor(dist.group.WORLD))
        for use_rope, recompute in product((False, True), repeat=2):
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            full = torch.randn(2, 12, 32, device=device, requires_grad=True)
            local = full.detach().chunk(2, dim=1)[rank].requires_grad_()
            weights = torch.randn_like(full)
            phases = torch.randn(12, 4, device=device).repeat(1, 2)
            rope = (phases.cos(), phases.sin()) if use_rope else None
            local_rope = tuple(t.chunk(2, dim=0)[rank] for t in rope) if rope else None
            if recompute:
                expected = checkpoint(reference, full, image_rotary_emb=rope, use_reentrant=False)
                actual = checkpoint(
                    candidate, local, image_rotary_emb=local_rope, use_reentrant=False
                )
            else:
                expected = reference(full, image_rotary_emb=rope)
                actual = candidate(local, image_rotary_emb=local_rope)
            torch.testing.assert_close(
                actual, expected.chunk(2, dim=1)[rank], atol=2e-6, rtol=2e-5
            )
            (expected * weights).sum().backward()
            (actual * weights.chunk(2, dim=1)[rank]).sum().backward()
            torch.testing.assert_close(
                local.grad, full.grad.chunk(2, dim=1)[rank], atol=2e-6, rtol=2e-5
            )
            for (name, parameter), (other_name, other) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert name == other_name
                assert parameter.grad is not None and other.grad is not None
                dist.all_reduce(other.grad)
                torch.testing.assert_close(other.grad, parameter.grad, atol=1e-5, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_cosmos_cp_attention_cpu(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), False), nprocs=2, join=True)


@pytest.mark.distributed
def test_cosmos_cp_attention_nccl(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_worker, args=((tmp_path / "nccl").as_uri(), True), nprocs=2, join=True)


@pytest.mark.parametrize("kwarg", ["encoder_hidden_states", "attention_mask"])
def test_cosmos_cp_attention_rejects_unsupported_inputs(kwarg):
    processor = CosmosContextParallelSelfAttnProcessor(None)
    with pytest.raises(ValueError, match="unmasked self-attention"):
        processor(None, None, **{kwarg: torch.zeros(1)})
