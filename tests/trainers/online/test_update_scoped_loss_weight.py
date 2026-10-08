"""An objective with an update-normalized weight sees the whole update first.

Flash-GRPO's rectification divides every sample's weight by the mean over the
optimizer update. The trainer therefore hands the algorithm, before the first
replay forward of an update, the recorded timestep of every (sample, trained
step) pair it is about to train on: per group the ``sde_window`` steps, or the
shared trained indices for the other selections.

The trainer is the real online wiring on tiny SANA with ``algorithm.kind=
flash_grpo``; ``prepare_update`` is the real FlashGRPO method, observed.
"""

from __future__ import annotations

import asyncio

import torch

from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.trajectory.reader import TrajectoryReader

_STEPS = 4


def _trainer(monkeypatch, tmp_path, *, timestep_fraction: float, ppo_epochs: int) -> TrainerBench:
    return real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            "algorithm.kind=flash_grpo",
            f"sampling.num_steps={_STEPS}",
            f"rollout.sde.window_range=[0,{_STEPS}]",
            "rollout.n_samples_per_prompt=3",
            "rollout.samples_per_generation_batch=3",
            "actor.training_microbatch_size=3",
            "actor.timestep_selection=strided",
            f"actor.timestep_fraction={timestep_fraction}",
            f"actor.ppo_epochs={ppo_epochs}",
            "actor.drop_zero_advantage=false",
        ),
    )


def _record_prepare(monkeypatch, tb: TrainerBench) -> list[torch.Tensor]:
    """Observe the real ``prepare_update``: the update timesteps it evaluates."""

    calls: list[torch.Tensor] = []
    real = tb.trainer.algorithm.prepare_update

    def record(update_timesteps):
        timesteps = update_timesteps()
        calls.append(timesteps.clone())
        return real(lambda: timesteps)

    monkeypatch.setattr(tb.trainer.algorithm, "prepare_update", record)
    return calls


def test_prepare_update_receives_every_trained_transition_of_the_update(
    monkeypatch, tmp_path
) -> None:
    tb = _trainer(monkeypatch, tmp_path, timestep_fraction=0.5, ppo_epochs=1)
    calls = _record_prepare(monkeypatch, tb)

    async def _run():
        batch = await tb.trainer.collect_training_batch(["p1", "p2"])
        await tb.trainer.train_on_rollout_batch(batch)
        return batch

    batch = asyncio.run(_run())

    # One preparation per optimizer update; the rectification is defined over
    # the same rollout SDE the replay evaluator integrates.
    assert len(calls) == 1
    (timesteps,) = calls
    algorithm, evaluator = tb.trainer.algorithm, tb.trainer.evaluator
    assert algorithm._scheduler is evaluator.scheduler
    assert (algorithm._noise_level, algorithm._sde_type) == (
        evaluator.noise_level,
        evaluator.sde_type,
    )
    # Strided selection at fraction 0.5 trains steps {0, 2} of the 4-step grid;
    # the tensor enumerates (group, trained step, sample) with the trajectory's
    # own timestep values, for every collected group.
    trained = tb.trainer._train_timestep_indices(_STEPS, 0.5, "strided")
    assert trained == [0, 2]
    recorded = [
        TrajectoryReader.from_batch(collected).replay_tensor_dict("denoise")["timesteps"]
        for collected in batch.batches
    ]
    assert len(recorded) == 2
    expected = torch.cat([grid[:, index] for grid in recorded for index in trained])
    assert torch.equal(timesteps, expected)
    # The recorded grid is the real scheduler's: later steps carry smaller timesteps.
    assert bool((recorded[0][:, 0] > recorded[0][:, 2]).all())


def test_prepare_update_runs_once_per_ppo_epoch(monkeypatch, tmp_path) -> None:
    tb = _trainer(monkeypatch, tmp_path, timestep_fraction=1.0, ppo_epochs=2)
    calls = _record_prepare(monkeypatch, tb)

    asyncio.run(tb.trainer.step(["p1", "p2"]))

    assert len(calls) == 2
