"""An objective with an update-normalized weight sees the whole update first.

Flash-GRPO's rectification divides every sample's weight by the mean over the
optimizer update. The trainer therefore hands the algorithm, before the first
replay forward of an update, the recorded timestep of every (sample, trained
step) pair it is about to train on: per group the ``sde_window`` steps, or the
shared trained indices for the other selections.
"""

from __future__ import annotations

import asyncio

import torch
import torch.nn as nn

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _EvaluatorAlgorithmFake,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import OnlineTrainer

_GROUP = 3
_STEPS = 4
_TIMESTEPS = torch.tensor([999.0, 750.0, 500.0, 250.0])


class _Algorithm(_EvaluatorAlgorithmFake):
    required_signal_keys = ("log_prob",)
    required_data_keys: tuple[str, ...] = ()

    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def __init__(self) -> None:
        self.prepared: list[dict] = []

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def prepare_update(self, timesteps, *, scheduler, noise_level, sde_type):
        self.prepared.append(
            {
                "timesteps": timesteps.clone(),
                "scheduler": scheduler,
                "noise_level": noise_level,
                "sde_type": sde_type,
            }
        )

    def compute_loss(self, inputs):
        signals, _advantages, _old = _algorithm_inputs(inputs)
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector(CollectorControlFake):
    async def evaluate_rollout(self, pendings):
        return list(pendings)

    async def generate_rollout(self, request):
        group_size = int(request.options["group_size"])
        return _diffusion_rollout_batch(
            rewards=torch.arange(group_size, dtype=torch.float32),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            num_steps=_STEPS,
            timesteps=_TIMESTEPS.repeat(group_size, 1),
        )


class _Evaluator(Evaluator):
    scheduler = object()
    noise_level = 0.7
    sde_type = "flow_grpo"

    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


def _trainer(tmp_path, *, timestep_fraction: float) -> tuple[OnlineTrainer, _Algorithm]:
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    algorithm = _Algorithm()
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=2, n_samples_per_prompt=_GROUP),
            timestep_fraction=timestep_fraction,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )
    return trainer, algorithm


def test_prepare_update_receives_every_trained_transition_of_the_update(tmp_path) -> None:
    trainer, algorithm = _trainer(tmp_path, timestep_fraction=0.5)

    async def _run():
        batch = await trainer.collect_training_batch(["p1", "p2"])
        await trainer.train_on_rollout_batch(batch)
        return batch

    batch = asyncio.run(_run())

    # One preparation per optimizer update, handed the evaluator's SDE settings.
    assert len(algorithm.prepared) == 1
    prepared = algorithm.prepared[0]
    assert prepared["scheduler"] is _Evaluator.scheduler
    assert prepared["noise_level"] == 0.7
    assert prepared["sde_type"] == "flow_grpo"
    # Strided selection at fraction 0.5 trains steps {0, 2} of the 4-step grid;
    # the tensor enumerates (group, trained step, sample) with the trajectory's
    # own timestep values, for every collected group.
    trained = trainer._train_timestep_indices(_STEPS, 0.5, "strided")
    assert trained == [0, 2]
    expected = torch.cat(
        [
            _TIMESTEPS[index].repeat(int(collected.rewards.shape[0]))
            for collected in batch.batches
            for index in trained
        ]
    )
    assert len(batch.batches) >= 1
    assert torch.equal(prepared["timesteps"], expected)


def test_prepare_update_runs_once_per_ppo_epoch(tmp_path) -> None:
    trainer, algorithm = _trainer(tmp_path, timestep_fraction=1.0)
    trainer.config.ppo_epochs = 2

    asyncio.run(trainer.step(["p1", "p2"]))

    assert len(algorithm.prepared) == 2
