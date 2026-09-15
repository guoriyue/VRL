"""Real four-rank accumulation with native BF16 base and FP32 adapters."""

import asyncio
import os
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor

from tests.trainers._strategy_policies import free_port
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _stamp_model_precision,
    _trajectory_signals,
)
from tests.trainers.online.test_step_split import _Algorithm, _build_trainer
from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.config import OnlineBatchPlan
from vrl.trainers.strategy import FSDPStrategy


class _MixedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.base.requires_grad_(False)
        self.adapter = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.base.weight.fill_(0.125)
            self.adapter.weight.fill_(0.25)

    def forward(self, values):
        return self.base(values.bfloat16()).float() + self.adapter(values)


class _Transformer(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["_MixedBlock"]

    def __init__(self):
        super().__init__()
        self.block = _MixedBlock()

    def forward(self, values):
        return self.block(values)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()
        _stamp_model_precision(self)

    @property
    def trainable_modules(self):
        return {"transformer": self.transformer}

    def set_module_root(self, name, module):
        setattr(self, name, module)

    def forward(self, rewards):
        return self.transformer(rewards[:, None].expand(-1, 4)).mean(dim=1)


class _GlobalAlgorithm(_Algorithm):
    class _Config(_Algorithm._Config):
        global_std = True

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        return group_relative_advantages(
            rewards, group_ids, eps=1e-8, adv_clip_max=0.5, global_std=True
        )

    def compute_loss(self, inputs):
        signals, advantages, _ = _algorithm_inputs(inputs)
        loss = -(signals.log_prob * advantages).mean() + 0.04 * signals.log_prob.square().mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


async def _run(output, rank=None, uneven=False):
    trainer = _build_trainer(output)
    trainer.model = _Policy()
    trainer.algorithm = _GlobalAlgorithm()
    trainer.config.drop_zero_advantage = True
    prompts = list(range(8)) if rank is None else list(range(2 * rank, 2 * rank + 2))
    plan = OnlineBatchPlan(
        prompts_per_batch=len(prompts),
        n_samples_per_prompt=4,
        prompts_per_collection=2 if rank is None else 1,
    )
    trainer.config.batch_plan = plan
    if rank is not None:
        strategy = FSDPStrategy(
            DistributedTrainingContext("fsdp", rank, 4, torch.device("cpu")),
            mesh_dims=["dp_shard"],
            precision_policy="none",
            reshard_after_forward=True,
            cpu_offload=False,
            shard_trainable_only=True,
        )
        strategy.prepare_model(trainer.model)
        trainer._strategy = strategy
        assert isinstance(trainer.model.transformer.block.adapter.weight, DTensor)
        assert not isinstance(trainer.model.transformer.block.base.weight, DTensor)
    assert trainer.model.transformer.block.base.weight.dtype == torch.bfloat16
    assert trainer.model.transformer.block.adapter.weight.dtype == torch.float32

    async def collect(prompts, **kwargs):
        batches = []
        for index, prompt in enumerate(prompts):
            rewards = torch.tensor([0.0, 1.0, 3.0, 7.0]) * (prompt + 1)
            if uneven and prompt == 0:
                rewards.fill_(2)
            batches.append(
                _diffusion_rollout_batch(
                    rewards=rewards,
                    group_ids=torch.full((4,), index, dtype=torch.long),
                    num_steps=2,
                )
            )
        return RolloutIteration(batches=batches)

    def evaluate(model, batch, timestep_idx, **kwargs):
        return _trajectory_signals(
            batch,
            model(batch.rewards),
            timestep_idx,
            old_log_prob=1.5 * batch.rewards,
        )

    trainer.rollout_schedule.next_iteration = collect
    trainer.evaluator.evaluate = evaluate
    norms = []
    for _ in range(2):
        metric = await _run_streaming_optimizer_update(trainer, prompts, batch_plan=plan)
        norms.append(metric.grad_norm)
        state = trainer._training_memory_state()
        trainer._strategy.park_training_state(state)
        trainer._strategy.restore_training_state(state)

    def full(value):
        return value.full_tensor() if isinstance(value, DTensor) else value

    weight = full(trainer.model.transformer.block.adapter.weight).detach().tolist()
    state = next(iter(trainer._optimizer.state.values()))
    moments = {name: full(value).tolist() for name, value in state.items()}
    assert trainer.state.global_step == 2
    assert not list(output.glob(".advantage-spool-*"))
    return norms, weight, moments


def _rank(rank, port, output, uneven, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    try:
        if uneven:
            with pytest.raises(ValueError, match="uneven filtering"):
                asyncio.run(_run(output / str(rank), rank, uneven=True))
            assert not list((output / str(rank)).glob(".advantage-spool-*"))
            queue.put((rank, True))
        else:
            queue.put((rank, asyncio.run(_run(output / str(rank), rank))))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("uneven", [False, True])
def test_four_rank_streaming_semantics(tmp_path, uneven):
    reference = None if uneven else asyncio.run(_run(tmp_path / "reference"))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    processes = [
        ctx.Process(target=_rank, args=(rank, port, tmp_path, uneven, queue)) for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        results = dict(queue.get(timeout=120) for _ in range(4))
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    for result in results.values():
        if uneven:
            assert result is True
            continue
        norms, weight, moments = result
        ref_norms, ref_weight, ref_moments = reference
        torch.testing.assert_close(
            torch.tensor(norms), torch.tensor(ref_norms), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            torch.tensor(weight), torch.tensor(ref_weight), rtol=1e-6, atol=1e-7
        )
        for key in moments:
            torch.testing.assert_close(
                torch.tensor(moments[key]), torch.tensor(ref_moments[key]), rtol=1e-5, atol=1e-6
            )
