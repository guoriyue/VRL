"""Trajectory-granularity replay scheduling in the online trainer."""

from __future__ import annotations

import asyncio

import pytest
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
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.token.continuous_token_logprob import (
    ContinuousTokenLogProbEvaluator,
)
from vrl.rollouts.evaluators.token.multi_segment_token_logprob import (
    MultiSegmentTokenLogProbEvaluator,
)
from vrl.rollouts.evaluators.token.token_logprob import TokenLogProbEvaluator
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trajectory import (
    TrajectoryResolver,
    TrajectoryTensor,
    build_chunk_autoregressive_denoise_trajectory,
)


class _Algorithm(_EvaluatorAlgorithmFake):
    required_signal_keys = ("log_prob",)
    required_data_keys: tuple[str, ...] = ()

    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_loss(self, inputs):
        signals, advantages, old_log_probs = _algorithm_inputs(inputs)
        del advantages, old_log_probs
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


def _chunk_denoise_batch(batch_size: int = 2) -> RolloutBatch:
    request = GenerationRequest(
        request_id="trainer-batch-test",
        family="test-batch-diffusion",
        task="t2v",
        inputs=["trainer batch test prompt"],
        samples_per_prompt=batch_size,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            sample_id=f"trainer-batch-test:sample:{index}",
        )
        for index in range(batch_size)
    ]
    observations = torch.zeros(batch_size, 4, 3, 1)
    actions = torch.zeros_like(observations)
    transition_values = torch.zeros(batch_size, 4, 3)
    trajectory = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations,
        actions=actions,
        old_log_prob=transition_values,
        mask=torch.ones_like(transition_values),
        timesteps=transition_values,
        finalized_chunk_latents=torch.zeros(batch_size, 4, 1),
        replay_tensors={},
        context={},
    )
    return RolloutBatch(
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=torch.zeros(batch_size, dtype=torch.long),
        trajectory=trajectory,
    )


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        prompts = [getattr(item, "prompt", item) for item in prompts]
        group_size = int(kwargs["group_size"])
        batch_size = len(prompts) * group_size
        return _chunk_denoise_batch(batch_size)


class _TrajectoryEvaluator:
    replay_granularity = "trajectory"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def evaluate(self, model, batch, timestep_idx, **kwargs):
        del kwargs
        resolver = TrajectoryResolver.from_batch(batch)
        segment = resolver.primary_trainable_segment_name()
        observations = resolver.role_value(segment, "observation")
        assert tuple(observations.shape[1:3]) == (4, 3)
        self.calls.append(int(timestep_idx))
        log_prob = model.weight.reshape(()).expand(batch.rewards.shape[0])
        return _trajectory_signals(batch, log_prob, timestep_idx)


@pytest.mark.parametrize("streaming", [False, True])
def test_trajectory_evaluator_runs_once_for_chunk_transition_axes(streaming: bool) -> None:
    evaluator = _TrajectoryEvaluator()
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    with torch.no_grad():
        model.weight.zero_()
    batch_plan = OnlineBatchPlan(
        prompts_per_batch=1,
        n_samples_per_prompt=2,
        gradient_accumulation_steps=1 if streaming else 0,
        samples_per_replay_batch=0,
    )
    trainer = OnlineTrainer(
        algorithm=_Algorithm(),
        collector=_Collector(),
        evaluator=evaluator,
        model=model,
        config=TrainerConfig(
            batch_plan=batch_plan,
            timestep_fraction=0.25,
            drop_zero_advantage=False,
            output_dir="outputs/",
            optim=OptimConfig(lr=0.0),
            ema=EMAConfig(),
            debug=DebugConfig(),
            precision_drift_guard=PrecisionDriftGuardConfig(mode="off"),
        ),
        device="cpu",
    )

    if streaming:
        asyncio.run(
            _run_streaming_optimizer_update(
                trainer,
                ["prompt"],
                batch_plan=batch_plan,
            ),
        )
    else:
        asyncio.run(trainer.step(["prompt"]))

    assert evaluator.calls == [0]


def test_unknown_replay_granularity_fails_fast() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = type("Evaluator", (), {"replay_granularity": "batch"})()
    batch = _diffusion_rollout_batch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
        num_steps=4,
    )

    with pytest.raises(ValueError, match="replay_granularity"):
        trainer._train_replay_indices(batch, 1.0)


@pytest.mark.parametrize(
    "evaluator",
    [
        TokenLogProbEvaluator(),
        ContinuousTokenLogProbEvaluator(),
        MultiSegmentTokenLogProbEvaluator(enabled_segments=()),
    ],
)
def test_token_evaluators_replay_multi_token_trajectories_once(evaluator: object) -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = evaluator

    assert trainer._train_replay_indices(_chunk_denoise_batch(), 0.5) == [0]


def test_step_evaluator_uses_primary_action_axis_for_fractional_selection() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = object()
    batch = _diffusion_rollout_batch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
        num_steps=4,
    )
    assert batch.trajectory is not None
    batch.trajectory.segments["denoise"].tensors["observations"] = TrajectoryTensor(
        name="observations",
        value=torch.zeros(1, 99),
        axes=("sample",),
        role="observation",
    )

    assert trainer._train_replay_indices(batch, 0.5) == [0, 2]


def test_step_replay_rejects_multiple_primary_action_axes() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = object()

    with pytest.raises(ValueError, match="exactly one non-sample axis"):
        trainer._train_replay_indices(_chunk_denoise_batch(), 1.0)


def test_evaluator_less_diffusion_uses_primary_action_axis() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = None
    batch = _diffusion_rollout_batch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
        num_steps=4,
    )

    assert trainer._train_replay_indices(batch, 0.5) == [0, 2]
