"""Real multi-block Cosmos forward/backward, including native checkpoint calls."""

import copy
from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import CosmosTransformer3DModel

from vrl.models.families.cosmos.context_parallel import cosmos_context_parallel


def _worker(rank, rendezvous, cuda, extra_pos, cross_projection):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(194)
        reference = CosmosTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=8,
            max_size=(4, 16, 16),
            patch_size=(1, 2, 2),
            concat_padding_mask=False,
            extra_pos_embed_type="learnable" if extra_pos else None,
            use_crossattn_projection=cross_projection,
            crossattn_proj_in_channels=64,
            encoder_hidden_states_channels=32,
        ).to(device)
        candidate = copy.deepcopy(reference)
        original_processors = [b.attn1.processor for b in candidate.transformer_blocks]
        for recompute, frame_time in product((False, True), repeat=2):
            for model in (reference, candidate):
                model.zero_grad(set_to_none=True)
                if recompute:
                    model.enable_gradient_checkpointing()
                else:
                    model.disable_gradient_checkpointing()
            full = torch.randn(1, 4, 3, 4, 4, device=device, requires_grad=True)
            local = full.detach().clone().requires_grad_()
            text = torch.randn(
                1, 3, 64 if cross_projection else 32, device=device, requires_grad=True
            )
            local_text = text.detach().clone().requires_grad_()
            timestep = torch.tensor([0.4], device=device)
            if frame_time:
                timestep = torch.tensor([0.3, 0.6, 0.2], device=device).view(1, 1, 3, 1, 1)
            expected = reference(full, timestep, text).sample
            weight = torch.randn_like(expected)
            (expected * weight).sum().backward()
            with cosmos_context_parallel(candidate, group=dist.group.WORLD):
                with (
                    pytest.raises(ValueError, match="already installed"),
                    cosmos_context_parallel(candidate, group=dist.group.WORLD),
                ):
                    pass
                actual = candidate(local, timestep, local_text).sample
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                ((actual * weight).sum() / 2).backward()
            for attention, original in zip(
                candidate.transformer_blocks, original_processors, strict=True
            ):
                assert attention.attn1.processor is original
            for actual_grad, expected_grad in (
                (local.grad, full.grad),
                (local_text.grad, text.grad),
            ):
                dist.all_reduce(actual_grad)
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=3e-5)
            for (name, parameter), (other_name, other) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert name == other_name
                if parameter.grad is None:
                    assert other.grad is None
                    continue
                assert other.grad is not None
                dist.all_reduce(other.grad)
                torch.testing.assert_close(
                    other.grad,
                    parameter.grad,
                    atol=2e-5,
                    rtol=5e-5,
                    msg=lambda message, label=(name, recompute, frame_time): f"{label}: {message}",
                )
        torch.testing.assert_close(candidate(local, timestep, local_text).sample, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("extra_pos", [False, True])
@pytest.mark.parametrize("cross_projection", [False, True])
def test_cosmos_context_parallel_model_cpu(tmp_path, extra_pos, cross_projection):
    mp.spawn(
        _worker,
        args=((tmp_path / "gloo").as_uri(), False, extra_pos, cross_projection),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("extra_pos", [False, True])
@pytest.mark.parametrize("cross_projection", [False, True])
def test_cosmos_context_parallel_model_nccl(tmp_path, extra_pos, cross_projection):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(
        _worker,
        args=((tmp_path / "nccl").as_uri(), True, extra_pos, cross_projection),
        nprocs=2,
        join=True,
    )
