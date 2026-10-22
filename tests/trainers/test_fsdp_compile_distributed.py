"""A compiled FSDP2 replay trains like the same compiled transformer unsharded.

Two Gloo ranks shard a tiny diffusers SD3 transformer in the online build
order (compile, then full or selective activation checkpointing, then
``FSDPStrategy.prepare_model`` with reshard-after-forward), each rank on its
half of one batch, for several AdamW steps. Regional and root compile must
match one process training the same compiled transformer on the whole batch:
the reshard-after-forward all-gathers and the gradient reduce-scatter run
outside the compiled graphs, so sharding adds only fp32 reduction-order noise.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from tests.trainers._strategy_policies import run_ranks

STEPS = 3
# (compile, checkpointing): root compile traces the selective policy itself.
VARIANTS = (("regional", "full"), ("root", "full"), ("root", "selective"))


def _transformer(variant: tuple[str, str]) -> nn.Module:
    from diffusers import SD3Transformer2DModel

    from vrl.nn.optimization.regional_compile import compile_repeated_blocks
    from vrl.trainers.activation_checkpointing import selective_checkpoint_func

    torch.manual_seed(3)
    transformer = SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=8,
        out_channels=4,
        pos_embed_max_size=16,
    )
    compile_scope, checkpointing = variant
    if compile_scope == "regional":
        compile_repeated_blocks(transformer, mode="default")
    else:
        transformer = torch.compile(transformer, mode="default", fullgraph=False)
    getattr(transformer, "_orig_mod", transformer).enable_gradient_checkpointing(
        gradient_checkpointing_func=(
            selective_checkpoint_func if checkpointing == "selective" else None
        ),
    )
    return transformer.train()


def _loss(transformer: nn.Module, rows: slice) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(4, 4, 8, 8, generator=generator)
    text = torch.randn(4, 6, 16, generator=generator)
    pooled = torch.randn(4, 8, generator=generator)
    target = torch.randn(4, 4, 8, 8, generator=generator)
    timestep = torch.tensor([100.0, 400.0, 700.0, 900.0])
    prediction = transformer(
        hidden_states=hidden[rows],
        encoder_hidden_states=text[rows],
        pooled_projections=pooled[rows],
        timestep=timestep[rows],
        return_dict=False,
    )[0]
    return (prediction - target[rows]).square().mean()


def _named_parameters(transformer: nn.Module) -> list[tuple[str, nn.Parameter]]:
    module = getattr(transformer, "_orig_mod", transformer)
    return list(module.named_parameters())


class _Policy(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer}

    def set_module_root(self, name: str, module: nn.Module) -> None:
        assert name == "transformer"
        self.transformer = module


def _run_rank(rank: int, world_size: int, port: int, queue) -> None:
    from vrl.trainers.distributed import DistributedTrainingContext
    from vrl.trainers.strategy import FSDPStrategy

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # A spawned rank is a fresh interpreter outside conftest's CUDA pin. On a
    # CUDA host inductor would initialize CUDA inside the first checkpointed
    # forward, which torch.utils.checkpoint refuses.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        results = {}
        for variant in VARIANTS:
            policy = _Policy(_transformer(variant))
            strategy = FSDPStrategy(
                DistributedTrainingContext(
                    strategy="fsdp",
                    rank=rank,
                    world_size=world_size,
                    device=torch.device("cpu"),
                ),
                precision_policy="none",
                reshard_after_forward=True,
                cpu_offload=False,
            )
            strategy.prepare_model(policy)
            optimizer = torch.optim.AdamW(policy.transformer.parameters(), lr=1e-3)
            for _ in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                strategy.backward(_loss(policy.transformer, slice(2 * rank, 2 * rank + 2)))
                optimizer.step()
            results[variant] = {
                name: parameter.detach().full_tensor().numpy()
                for name, parameter in _named_parameters(policy.transformer)
            }
        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


def test_compiled_fsdp_replay_trains_like_the_unsharded_compiled_transformer() -> None:
    results = run_ranks(_run_rank, 2, timeout=600)

    for variant in VARIANTS:
        reference = _transformer(variant)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            _loss(reference, slice(0, 4)).backward()
            optimizer.step()
        expected = dict(_named_parameters(reference))
        for rank in range(2):
            sharded = results[rank][0][variant]
            assert sharded.keys() == expected.keys()
            for name, value in sharded.items():
                np.testing.assert_allclose(
                    value,
                    expected[name].detach().numpy(),
                    rtol=0,
                    atol=2e-6,
                    err_msg=f"{variant} {name}",
                )
