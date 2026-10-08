"""A diffusion family trains under the real DDP strategy.

``DDPStrategy.prepare_model`` replaces the policy's transformer with its DDP
wrapper, so a family forward that reads the transformer's own config must
read it through the wrapper. One gloo rank over the tiny SANA stack, wired as
the online recipe wires it, runs a real training step.
"""

from __future__ import annotations

import asyncio

import torch
import torch.distributed as dist

from tests.trainers.online._helpers import real_trainer
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import DDPStrategy


def test_sana_trains_one_step_under_ddp(monkeypatch, tmp_path) -> None:
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    try:
        strategy = DDPStrategy(
            DistributedTrainingContext(
                strategy="ddp", rank=0, world_size=1, device=torch.device("cpu")
            ),
            find_unused_parameters=False,
        )
        tb = real_trainer(
            monkeypatch,
            tmp_path,
            strategy=strategy,
            overrides=("actor.ppo_epochs=1", "actor.drop_zero_advantage=false"),
        )
        before = {name: p.detach().clone() for name, p in tb.trainable_parameters().items()}

        metrics = asyncio.run(tb.trainer.step(["a cat", "a dog"]))

        transformer = tb.bundle.model.trainable_modules["transformer"]
        assert isinstance(transformer, torch.nn.parallel.DistributedDataParallel)
        assert torch.isfinite(torch.tensor(metrics.loss))
        after = tb.trainable_parameters()
        assert any(not torch.equal(before[name], after[name].detach()) for name in before)
    finally:
        dist.destroy_process_group()
