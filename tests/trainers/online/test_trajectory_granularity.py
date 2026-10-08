"""Trajectory-granularity replay scheduling in the online trainer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from tests.rollouts.collector._helpers import real_collector
from tests.trainers.online._helpers import _diffusion_rollout_batch, bare_trainer
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.scripts.common.factory import AlgorithmEvaluatorPair
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer
from vrl.trajectory.builders import build_chunk_autoregressive_denoise_trajectory
from vrl.trajectory.reader import TrajectoryReader
from vrl.trajectory.types import TrajectoryTensor


def _indices(trainer, batch, fraction: float, selection: str) -> list[int]:
    """The trainer reads its selection off its config; set it per call."""

    trainer.config = SimpleNamespace(timestep_fraction=fraction, timestep_selection=selection)
    return trainer._train_replay_indices(batch)


def _trajectory_signals(
    batch,
    log_prob,
    timestep_idx: int = 0,
    *,
    old_log_prob=None,
):
    from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

    if old_log_prob is None:
        # An unchanged-policy replay unless the test supplies the behavior value.
        old_log_prob = log_prob.detach().clone()
    mask = torch.ones_like(log_prob)
    return TrajectorySignalBatch(
        segments={
            "default": SegmentSignal(
                name="default",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
            ),
        },
        group_ids=batch.group_ids,
        primary_segment="default",
    )


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


class _TrajectoryEvaluator:
    """A trajectory-granularity evaluator over the real policy's parameters.

    No family that produces chunk-autoregressive trajectories runs on CPU
    (CausVid needs flash attention and its pinned source checkout; MAGI-1 has
    no replayable likelihood), so the evaluator that consumes them is the one
    double here. It checks it was handed the whole ``[sample, chunk,
    transition]`` trajectory and returns a log-prob that depends on a real
    trainable parameter, so GRPO's backward reaches the policy.
    """

    replay_granularity = "trajectory"
    supports_deferred_replay_tensor_move = False

    def __init__(self) -> None:
        self.calls: list[int] = []

    def evaluate(self, model, batch, timestep_idx, **kwargs):
        del kwargs
        reader = TrajectoryReader.from_batch(batch)
        segment = reader.primary_trainable_segment_name()
        observations = reader.role_value(segment, "observation")
        assert tuple(observations.shape[1:3]) == (4, 3)
        self.calls.append(int(timestep_idx))
        parameter = next(p for p in model.parameters() if p.requires_grad)
        log_prob = (parameter.reshape(-1)[0] * 0.0).expand(batch.rewards.shape[0])
        return _trajectory_signals(batch, log_prob, timestep_idx)


def _trainer(
    monkeypatch, tmp_path, *, streaming: bool
) -> tuple[OnlineTrainer, _TrajectoryEvaluator]:
    """The recipe's trainer wiring on tiny SANA with the trajectory evaluator."""

    bench = real_collector(
        monkeypatch,
        tmp_path,
        overrides=(
            "algorithm.kl_coef=0.0",
            "actor.ppo_epochs=1",
            "actor.drop_zero_advantage=false",
            *(("actor.prompts_per_collection=1",) if streaming else ()),
        ),
    )
    stack = bench.stack
    built = stack.resolved.built
    bundle = stack.trainer_bundle()
    strategy = SingleProcessStrategy()
    algorithm = AlgorithmEvaluatorPair.from_configs(
        built,
        scheduler=getattr(bundle, "scheduler", None),
    ).algorithm
    evaluator = _TrajectoryEvaluator()
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=bench.collector,
        evaluator=evaluator,
        model=bundle.model,
        weight_syncer=RayRuntimeWeightSyncer(bench.runtime),
        sync_state_getter=lambda: strategy.export_rollout_state(bundle),
        config=built.trainer,
        strategy=strategy,
    )
    return trainer, evaluator


@pytest.mark.parametrize("streaming", [False, True])
def test_trajectory_evaluator_runs_once_for_chunk_transition_axes(
    monkeypatch, tmp_path, streaming: bool
) -> None:
    trainer, evaluator = _trainer(monkeypatch, tmp_path, streaming=streaming)
    # The chunk trajectory enters through the trainer's pre-collected iteration
    # seam, the one the global-std streaming recipe uses.
    iteration = RolloutIteration(batches=[_chunk_denoise_batch()])

    async def update():
        if streaming:
            await _run_streaming_optimizer_update(
                trainer,
                ["prompt"],
                _prepared=iter([(iteration, None, None)]),
            )
        else:
            batch = await trainer.collect_training_batch(["prompt"], _iteration=iteration)
            await trainer.train_on_rollout_batch(batch)
        await trainer.rollout_schedule.shutdown()

    asyncio.run(update())

    assert evaluator.calls == [0]


def test_unknown_replay_granularity_fails_fast() -> None:
    trainer = bare_trainer(evaluator=type("Evaluator", (), {"replay_granularity": "batch"})())
    batch = _diffusion_rollout_batch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
        num_steps=4,
    )

    with pytest.raises(ValueError, match="replay_granularity"):
        _indices(trainer, batch, 1.0, "strided")


def test_step_evaluator_uses_primary_action_axis_for_fractional_selection() -> None:
    trainer = bare_trainer(evaluator=None)
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

    assert _indices(trainer, batch, 0.5, "strided") == [0, 2]


def test_step_replay_rejects_multiple_primary_action_axes() -> None:
    trainer = bare_trainer(evaluator=None)

    with pytest.raises(ValueError, match="exactly one non-sample axis"):
        _indices(trainer, _chunk_denoise_batch(), 1.0, "strided")


def test_evaluator_less_diffusion_uses_primary_action_axis() -> None:
    trainer = bare_trainer(evaluator=None)
    batch = _diffusion_rollout_batch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
        num_steps=4,
    )

    assert _indices(trainer, batch, 0.5, "strided") == [0, 2]
