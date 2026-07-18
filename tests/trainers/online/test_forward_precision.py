"""Resolved forward-precision boundaries owned by ``OnlineTrainer``."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tests.trainers.online._helpers import (
    DEFAULT_FORWARD_PRECISION,
    _rollout_context,
    _trajectory_signals,
)
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.trainers.online import OnlineTrainer


def test_replay_forwards_contract_to_evaluator_before_algorithm_loss() -> None:
    model = nn.Linear(1, 1, bias=False)
    batch = RolloutBatch(
        observations=torch.zeros(2, 1, 1),
        actions=torch.zeros(2, 1, 1),
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.zeros(2, dtype=torch.long),
        context=_rollout_context(),
    )

    class _Evaluator:
        def evaluate(self, forward_model, replay_batch, timestep_index, **kwargs):
            assert kwargs["forward_precision"] is DEFAULT_FORWARD_PRECISION
            assert forward_model is model
            return _trajectory_signals(
                replay_batch,
                model.weight.reshape(1).expand(2),
                timestep_index,
            )

    class _AlgorithmAdapter:
        def compute_loss(self, algorithm, inputs):
            del algorithm
            loss = inputs.signals.primary.log_prob.mean()
            return loss, TrainStepMetrics(loss=float(loss.detach()))

    trainer = object.__new__(OnlineTrainer)
    trainer.algorithm = SimpleNamespace(
        uses_evaluator=True,
        config=SimpleNamespace(kl_coef=0.0),
    )
    trainer.evaluator = _Evaluator()
    trainer.model = model
    trainer.ref_model = None
    trainer.forward_precision = DEFAULT_FORWARD_PRECISION
    trainer.device = torch.device("cpu")

    loss, _metrics = trainer._compute_replay_loss(
        batch,
        torch.ones(2),
        0,
        algorithm_adapter=_AlgorithmAdapter(),
    )

    assert loss.requires_grad
