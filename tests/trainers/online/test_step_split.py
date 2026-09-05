"""P4 collect/train split: the decomposition must not change behavior.

``OnlineTrainer.step`` now runs ``collect_training_batch`` then
``train_on_rollout_batch``. These lock that the split reproduces the previous
single-method behavior: identical metrics and state, the collected batch carries
the data the train half needs, and rollout weight sync still happens in the train
half (not the collect half).
"""

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
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import TrainingBatch


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
        signals, _advantages, _old_log_probs = _algorithm_inputs(inputs)
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        group_size = int(kwargs["group_size"])
        return _diffusion_rollout_batch(
            rewards=torch.arange(group_size, dtype=torch.float32),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            num_steps=2,
        )


class _Evaluator(Evaluator):
    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


def _build_trainer(tmp_path) -> OnlineTrainer:
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return OnlineTrainer(
        algorithm=_Algorithm(),
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )


def _metric_fields(m: TrainStepMetrics) -> tuple:
    return (
        m.loss,
        m.policy_loss,
        m.reward_mean,
        m.reward_std,
        m.advantage_mean,
        m.grad_norm,
        m.group_size,
        m.trained_prompt_num,
        m.adv_zero_rate,
        m.adv_saturation,
    )


def test_step_equals_collect_then_train(tmp_path) -> None:
    """step() and an explicit collect+train produce identical metrics + state."""
    mono = _build_trainer(tmp_path / "mono")
    mono_metrics = asyncio.run(mono.step(["p"]))

    split = _build_trainer(tmp_path / "split")

    async def _run_split() -> TrainStepMetrics:
        batch = await split.collect_training_batch(["p"])
        return await split.train_on_rollout_batch(batch)

    split_metrics = asyncio.run(_run_split())

    assert _metric_fields(split_metrics) == _metric_fields(mono_metrics)
    assert split.state.step == mono.state.step
    assert split.state.global_step == mono.state.global_step
    assert torch.equal(split.model.weight, mono.model.weight)


def test_collect_returns_training_batch_with_data(tmp_path) -> None:
    trainer = _build_trainer(tmp_path)
    batch = asyncio.run(trainer.collect_training_batch(["p"]))

    assert isinstance(batch, TrainingBatch)
    assert batch.batches and batch.advantages
    assert len(batch.batches) == len(batch.advantages)
    # collect must not advance training state — that is the train half's job.
    assert trainer.state.step == 0
    assert trainer.state.global_step == 0


def test_weight_sync_happens_in_train_half_not_collect(tmp_path) -> None:
    """rollout_schedule.after_train_step fires in train, never during collect."""
    trainer = _build_trainer(tmp_path)
    calls: list[int] = []
    real_after = trainer.rollout_schedule.after_train_step

    async def _counting_after():
        calls.append(1)
        await real_after()
        stats = RolloutStats()
        stats.add_phase("continuous.weight_sync_pause_s", 0.25)
        return stats

    trainer.rollout_schedule.after_train_step = _counting_after  # type: ignore[method-assign]

    batch = asyncio.run(trainer.collect_training_batch(["p"]))
    assert calls == []  # collect half does not sync

    metrics = asyncio.run(trainer.train_on_rollout_batch(batch))
    assert calls == [1]  # train half syncs exactly once
    assert metrics.phase_times["continuous.weight_sync_pause_s"] == 0.25


def test_next_prompts_reaches_the_rollout_schedule(tmp_path) -> None:
    """The lookahead must survive step -> _step_impl -> collect -> schedule.

    Continuous rollout installs this as the producer's next prompt batch so
    generation overlaps training. A dropped forward costs that overlap without failing
    anything observable, and the owner-side tests all drive the schedule
    directly — this pins the trainer half of the hop. ``None`` must stay
    ``None`` rather than becoming ``[]``, which the owner rejects outright.
    """
    trainer = _build_trainer(tmp_path)
    seen: list[tuple[list, list | None]] = []
    real_next = trainer.rollout_schedule.next_iteration

    async def _recording_next(prompts, **kwargs):
        seen.append((list(prompts), kwargs.get("next_prompts")))
        return await real_next(prompts, **kwargs)

    trainer.rollout_schedule.next_iteration = _recording_next  # type: ignore[method-assign]

    asyncio.run(trainer.step(["p"], next_prompts=["q"]))
    asyncio.run(trainer.step(["q"]))

    assert seen == [(["p"], ["q"]), (["q"], None)]


def test_backward_uses_named_profile_range(tmp_path, monkeypatch) -> None:
    """The shared backward boundary must remain visible to external profilers."""
    import contextlib

    from vrl.utils import profiling

    trainer = _build_trainer(tmp_path)
    events: list[str] = []

    @contextlib.contextmanager
    def _record(name: str):
        events.append(f"enter:{name}")
        try:
            yield
        finally:
            events.append(f"exit:{name}")

    monkeypatch.setattr(profiling, "profile_range", _record)
    trainer._backward(trainer.model.weight.sum())

    assert events == ["enter:trainer.backward", "exit:trainer.backward"]


def test_streaming_all_filtered_update_does_not_advance_policy(tmp_path) -> None:
    """An all-zero streamed batch records a step without publishing new weights."""
    from vrl.scripts.common.online import _run_streaming_optimizer_update

    trainer = _build_trainer(tmp_path)
    trainer.config.drop_zero_advantage = True
    trainer.algorithm.compute_advantages_from_tensors = (  # type: ignore[method-assign]
        lambda rewards, group_ids: torch.zeros_like(rewards)
    )
    sync_calls: list[int] = []

    async def _unexpected_sync() -> RolloutStats:
        sync_calls.append(1)
        return RolloutStats()

    trainer.rollout_schedule.after_train_step = _unexpected_sync  # type: ignore[method-assign]
    initial_weight = trainer.model.weight.detach().clone()

    metrics = asyncio.run(
        _run_streaming_optimizer_update(
            trainer,
            ["p"],
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=1,
                n_samples_per_prompt=2,
                gradient_accumulation_steps=1,
            ),
        ),
    )

    assert trainer.state.step == 1
    assert trainer.state.global_step == 0
    assert sync_calls == []
    assert metrics.grad_norm == 0.0
    assert torch.equal(trainer.model.weight, initial_weight)
    assert trainer._precision_drift_guard_pending is True


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("drift", [0.0005, 0.1])
def test_corrected_replay_enforces_drift_guard_on_both_update_paths(
    tmp_path,
    streaming: bool,
    drift: float,
) -> None:
    """Correction bypasses exact parity, not the bounded first-update guard."""
    import json

    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.scripts.common.online import _run_streaming_optimizer_update
    from vrl.trainers.core.types import PrecisionDriftGuardConfig
    from vrl.trainers.online.precision_guard import PrecisionDriftError

    trainer = _build_trainer(tmp_path)
    trainer.algorithm.precision_correction = PrecisionCorrectionConfig(tis_mode="truncate")
    trainer.config.precision_drift_guard = PrecisionDriftGuardConfig(
        mode="fail",
        max_abs_log_ratio=0.001,
        max_ratio_abs_dev=0.001,
    )

    class DriftEvaluator(Evaluator):
        def evaluate(self, model, batch, timestep_idx, **kw):
            del kw
            fresh = model.weight.view(1).expand(batch.rewards.shape[0]) + drift
            return _trajectory_signals(
                batch,
                fresh,
                timestep_idx,
                old_log_prob=torch.ones_like(fresh),
            )

    trainer.evaluator = DriftEvaluator()
    initial_weight = trainer.model.weight.detach().clone()

    async def run_update():
        if streaming:
            return await _run_streaming_optimizer_update(
                trainer,
                ["p"],
                batch_plan=OnlineBatchPlan(
                    prompts_per_batch=1,
                    n_samples_per_prompt=2,
                    gradient_accumulation_steps=1,
                ),
            )
        return await trainer.step(["p"])

    if drift > trainer.config.precision_drift_guard.max_abs_log_ratio:
        with pytest.raises(PrecisionDriftError, match="precision drift guard"):
            asyncio.run(run_update())
        assert trainer.state.global_step == 0
        assert torch.equal(trainer.model.weight, initial_weight)
        assert trainer._precision_drift_guard_pending is True
    else:
        asyncio.run(run_update())
        assert trainer.state.global_step == 1
        assert not torch.equal(trainer.model.weight, initial_weight)
        assert trainer._precision_drift_guard_pending is False
        records = [
            json.loads(line)
            for line in (tmp_path / "training_debug.jsonl").read_text().splitlines()
        ]
        assert [record["event"] for record in records] == ["precision_drift_guard"]


def test_streaming_scaler_skipped_update_does_not_publish_weights(tmp_path) -> None:
    """A skipped optimizer attempt counts but cannot publish unchanged weights."""
    from vrl.scripts.common.online import _run_streaming_optimizer_update

    trainer = _build_trainer(tmp_path)
    trainer._clip_and_step = lambda optimizer: (0.0, False)  # type: ignore[method-assign]
    sync_calls: list[int] = []

    async def _unexpected_sync() -> RolloutStats:
        sync_calls.append(1)
        return RolloutStats()

    trainer.rollout_schedule.after_train_step = _unexpected_sync  # type: ignore[method-assign]

    asyncio.run(
        _run_streaming_optimizer_update(
            trainer,
            ["p"],
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=1,
                n_samples_per_prompt=2,
                gradient_accumulation_steps=1,
            ),
        ),
    )

    assert trainer.state.step == 1
    assert trainer.state.global_step == 1
    assert sync_calls == []


def test_phase_events_use_the_metric_step(tmp_path) -> None:
    """JSON phase events and the stats sink use the same zero-based step."""
    import json

    trainer = _build_trainer(tmp_path)
    trainer.config.profile = True

    asyncio.run(trainer.step(["p"]))

    event_path = tmp_path / "phase_events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert events
    assert {event["step"] for event in events} == {0}
    assert trainer.state.step == 1


def test_streaming_profiles_training_phases(tmp_path) -> None:
    """Streaming metrics and events cover replay, backward, and optimizer work."""
    import json

    from vrl.scripts.common.online import _run_streaming_optimizer_update

    trainer = _build_trainer(tmp_path)
    trainer.config.profile = True

    metrics = asyncio.run(
        _run_streaming_optimizer_update(
            trainer,
            ["p"],
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=1,
                n_samples_per_prompt=2,
                gradient_accumulation_steps=1,
            ),
        ),
    )

    for phase in ("evaluate", "backward", "optim_step"):
        assert metrics.phase_times[phase] > 0.0

    events = [
        json.loads(line) for line in (tmp_path / "phase_events.jsonl").read_text().splitlines()
    ]
    training_events = {
        event["phase"]: event["step"]
        for event in events
        if event["phase"] in {"evaluate", "backward", "optim_step"}
    }
    assert training_events == {"evaluate": 0, "backward": 0, "optim_step": 0}
