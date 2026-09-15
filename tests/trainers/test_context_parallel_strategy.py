"""Exercise the real trainer clip/step boundary with a DP2 x CP2 strategy."""

import copy
import math
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import CosmosTransformer3DModel

from tests.trainers._strategy_policies import FakePolicy
from vrl.models.precision import apply_float32_precision, fixed_row_linear_compute
from vrl.trainers.checkpointing import _require_equal_tensor_tree
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import ContextParallelStrategy


def _worker(rank, rendezvous, cuda):
    torch.set_num_threads(1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    apply_float32_precision("ieee")
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    strategy = ContextParallelStrategy(
        DistributedTrainingContext("context_parallel", rank, 4, device), cp_size=2
    )
    try:
        torch.manual_seed(911)
        reference = CosmosTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=1,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=8,
            max_size=(4, 16, 16),
            patch_size=(1, 2, 2),
            concat_padding_mask=False,
            extra_pos_embed_type=None,
            use_crossattn_projection=True,
            crossattn_proj_in_channels=64,
            encoder_hidden_states_channels=32,
        ).to(device)
        candidate = copy.deepcopy(reference)
        with torch.no_grad():
            next(candidate.parameters()).add_(rank * 0.1)
        policy = FakePolicy(candidate)
        assert strategy.prepare_model(policy) is policy
        for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="already prepared"):
            strategy.prepare_model(policy)
        with pytest.raises(NotImplementedError, match="GradScaler"):
            strategy.backward(torch.ones((), device=device), grad_scaler=object())
        with pytest.raises(NotImplementedError, match="parking"):
            strategy.validate_training_state_parking()
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-4) for model in (reference, candidate)
        ]
        trainer = SimpleNamespace(
            model=candidate,
            config=SimpleNamespace(max_norm=1.0),
            _strategy=strategy,
            _grad_scaler=None,
        )
        with fixed_row_linear_compute(reference):
            for _update in range(2):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                inputs = torch.randn(2, 2, 1, 4, 3, 4, 4, device=device)
                texts = torch.randn(2, 2, 1, 3, 64, device=device)
                timestep = torch.tensor([0.4], device=device)
                for micro in range(2):
                    for dp in range(2):
                        output = reference(inputs[micro, dp], timestep, texts[micro, dp]).sample
                        (output.square().mean() / 4).backward()
                    dp = strategy.groups.dp_rank
                    output = candidate(inputs[micro, dp], timestep, texts[micro, dp]).sample
                    strategy.backward(output.square().mean() / 2)
                expected_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0))
                optimizers[0].step()
                actual_norm, stepped = OnlineTrainer._clip_and_step(trainer, optimizers[1])
                assert stepped and math.isclose(
                    actual_norm, expected_norm, rel_tol=2e-5, abs_tol=1e-6
                )
                for left, right in zip(
                    reference.parameters(), candidate.parameters(), strict=True
                ):
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
                    a, b = optimizers[0].state.get(left), optimizers[1].state.get(right)
                    if a is None:
                        assert b is None
                    else:
                        for key in a:
                            torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=3e-5)
            state = copy.deepcopy(strategy.export_optimizer_state(candidate, optimizers[1]))
            fresh = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
            strategy.load_optimizer_state(candidate, fresh, state)
            assert len(fresh.state) == len(optimizers[1].state) > 0
            _require_equal_tensor_tree(state, fresh.state_dict(), label="restored CP optimizer")
    finally:
        strategy.shutdown()
    assert not candidate.proj_out._forward_hooks
    assert "forward" not in candidate.proj_out.__dict__


def test_context_parallel_strategy_trainer_boundary_cpu(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), False), nprocs=4, join=True)


@pytest.mark.distributed
def test_context_parallel_strategy_trainer_boundary_nccl(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    mp.spawn(_worker, args=((tmp_path / "nccl").as_uri(), True), nprocs=4, join=True)
