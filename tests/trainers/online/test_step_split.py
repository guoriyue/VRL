"""P4 collect/train split over the real trainer: the decomposition must not change behavior.

``OnlineTrainer.step`` runs ``collect_training_batch`` then
``train_on_rollout_batch``. These lock that the split reproduces the single
call on the real GRPO trainer (tiny SANA, ``real_trainer``): identical metrics
and weights, the collected batch carries the data the train half needs, and
rollout weight sync happens in the train half, never in the collect half.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from tests.rollouts.collector._helpers import IndexReward
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.algorithms.types import TrainStepMetrics
from vrl.rewards import RewardOutput, RewardSample
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.online.trainer import TrainingBatch

# Streaming accumulation: one prompt per collection batch, so the released
# microbatch cannot be replayed across PPO epochs.
_STREAMING = ("actor.prompts_per_collection=1", "actor.ppo_epochs=1")


class _ConstantReward(IndexReward):
    """Every sample scores the same, so every group's advantage is zero."""

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await super().score_batch(samples)
        return RewardOutput(scores=tuple(1.0 for _ in samples))


def _seed() -> None:
    # Request seeds come from Python's RNG, sampling noise from torch's.
    random.seed(0)
    torch.manual_seed(0)


def _snapshot(tb: TrainerBench) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in tb.trainable_parameters().items()}


def _unchanged(tb: TrainerBench, before: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(before[name], value) for name, value in tb.trainable_parameters().items()
    )


def _pushes(tb: TrainerBench) -> int:
    return tb.collector.trace.events.count("update_weights")


def _corrupt_weight_sync(monkeypatch: pytest.MonkeyPatch, tb: TrainerBench, scale: float) -> None:
    """Make the rollout serve weights that differ from the trainer's.

    Each push reaches the real runtime with seeded noise added, the shape of a
    broken weight transport: generation runs on theta + delta while replay
    runs on theta, so rollout and replay log-probs disagree for real.
    """

    runtime = tb.collector.runtime
    real = runtime.update_weights
    generator = torch.Generator().manual_seed(1)

    async def update_weights(trainable_state, policy_version):
        corrupted = {
            name: value + torch.randn(value.shape, generator=generator) * scale
            for name, value in trainable_state.items()
        }
        return await real(corrupted, policy_version)

    monkeypatch.setattr(runtime, "update_weights", update_weights)


def _debug_records(tb: TrainerBench) -> list[dict]:
    path = Path(tb.trainer.config.output_dir) / "training_debug.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _jsonl(tb: TrainerBench, name: str) -> list[dict]:
    path = Path(tb.trainer.config.output_dir) / name
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_step_equals_collect_then_train(monkeypatch, tmp_path) -> None:
    """step() and an explicit collect+train produce identical metrics and weights."""

    mono = real_trainer(monkeypatch, tmp_path / "mono")
    _seed()
    mono_metrics = asyncio.run(mono.trainer.step(["a cat"]))

    split = real_trainer(monkeypatch, tmp_path / "split")
    _seed()

    async def _run_split() -> TrainStepMetrics:
        batch = await split.trainer.collect_training_batch(["a cat"])
        return await split.trainer.train_on_rollout_batch(batch)

    split_metrics = asyncio.run(_run_split())

    assert _metric_fields(split_metrics) == _metric_fields(mono_metrics)
    assert split.trainer.state.step == mono.trainer.state.step == 1
    assert split.trainer.state.global_step == mono.trainer.state.global_step
    mono_params = mono.trainable_parameters()
    for name, value in split.trainable_parameters().items():
        assert torch.equal(value, mono_params[name]), name


def test_collect_returns_training_batch_with_data(monkeypatch, tmp_path) -> None:
    tb = real_trainer(monkeypatch, tmp_path)
    before = _snapshot(tb)

    batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))

    assert isinstance(batch, TrainingBatch)
    assert batch.batches and batch.advantages
    assert len(batch.batches) == len(batch.advantages)
    assert batch.batches[0].trajectory is not None
    # collect must not advance training state; that is the train half's job.
    assert tb.trainer.state.step == 0
    assert tb.trainer.state.global_step == 0
    assert _unchanged(tb, before)


def test_weight_sync_happens_in_train_half_not_collect(monkeypatch, tmp_path) -> None:
    """The post-train weight push fires in train, never during collect."""

    tb = real_trainer(monkeypatch, tmp_path)

    batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
    # Only the initial push the first rollout needs; no post-train sync yet.
    assert _pushes(tb) == 1
    assert tb.collector.runtime.current_policy_version == 1

    metrics = asyncio.run(tb.trainer.train_on_rollout_batch(batch))

    assert _pushes(tb) == 2
    assert tb.collector.runtime.current_policy_version == 2
    assert metrics.phase_times["rollout.weight_sync_s"] > 0.0


def test_next_prompts_reaches_the_rollout_schedule(monkeypatch, tmp_path) -> None:
    """The next-batch prefetch input must reach step -> collect -> schedule.

    Continuous rollout installs this as the producer's next prompt batch so
    generation overlaps training. A dropped forward costs that overlap without
    failing anything observable, so this pins the trainer half of the hop.
    ``None`` must stay ``None`` rather than becoming ``[]``, which the owner
    rejects outright.
    """

    tb = real_trainer(monkeypatch, tmp_path)
    schedule = tb.trainer.rollout_schedule
    seen: list[tuple[list, list | None]] = []
    real_next = schedule.next_iteration

    async def recording_next(prompts, **kwargs):
        seen.append((list(prompts), kwargs.get("next_prompts")))
        return await real_next(prompts, **kwargs)

    monkeypatch.setattr(schedule, "next_iteration", recording_next)

    asyncio.run(tb.trainer.step(["a cat"], next_prompts=["a dog"]))
    asyncio.run(tb.trainer.step(["a dog"]))

    assert seen == [(["a cat"], ["a dog"]), (["a dog"], None)]


def test_backward_uses_named_profile_range(monkeypatch, tmp_path) -> None:
    """The shared backward boundary must remain visible to external profilers."""

    import contextlib

    from vrl.utils import profiling

    tb = real_trainer(monkeypatch, tmp_path)
    events: list[str] = []

    @contextlib.contextmanager
    def record(name: str):
        events.append(f"enter:{name}")
        try:
            yield
        finally:
            events.append(f"exit:{name}")

    monkeypatch.setattr(profiling, "profile_range", record)
    parameter = next(iter(tb.trainable_parameters().values()))
    tb.trainer._backward(parameter.sum())

    assert events == ["enter:trainer.backward", "exit:trainer.backward"]
    assert parameter.grad is not None


def test_streaming_all_filtered_update_does_not_advance_policy(monkeypatch, tmp_path) -> None:
    """An all-zero-advantage streamed batch records a step without publishing weights."""

    tb = real_trainer(monkeypatch, tmp_path, reward=_ConstantReward(), overrides=_STREAMING)
    assert tb.trainer.config.drop_zero_advantage is True
    before = _snapshot(tb)

    metrics = asyncio.run(
        _run_streaming_optimizer_update(
            tb.trainer, ["a cat"], batch_plan=tb.trainer.config.batch_plan
        ),
    )

    assert tb.trainer.state.step == 1
    assert tb.trainer.state.global_step == 0
    # The initial push only: a filtered update has nothing to publish.
    assert _pushes(tb) == 1
    assert metrics.grad_norm == 0.0
    assert _unchanged(tb, before)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("scale", [0.05, 0.3], ids=["inside_bound", "beyond_bound"])
def test_corrected_replay_fails_parity_only_at_the_catastrophic_bound(
    monkeypatch,
    tmp_path,
    streaming: bool,
    scale: float,
) -> None:
    """Correction relaxes the recipe parity threshold, not the ln(10) bound."""

    from vrl.trainers.core.types import CORRECTED_REPLAY_MAX_ABS_LOG_RATIO

    tb = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            "trainer.precision_correction.tis_mode=truncate",
            # One optimizer step per update in both arms.
            "actor.ppo_epochs=1",
            *(_STREAMING if streaming else ()),
        ),
    )
    _corrupt_weight_sync(monkeypatch, tb, scale)
    before = _snapshot(tb)
    _seed()

    async def run_update():
        if streaming:
            return await _run_streaming_optimizer_update(
                tb.trainer, ["a cat"], batch_plan=tb.trainer.config.batch_plan
            )
        return await tb.trainer.step(["a cat"])

    try:
        asyncio.run(run_update())
        failure = None
    except RuntimeError as error:
        failure = error

    records = _debug_records(tb)
    assert [record["event"] for record in records] == ["replay_parity_gate"]
    gate = records[0]
    drift = gate["max_abs_diff"]
    beyond_bound = drift > CORRECTED_REPLAY_MAX_ABS_LOG_RATIO
    # Each scale lands on its intended side of the bound, and both are
    # beyond the recipe threshold that correction relaxes.
    assert drift > tb.trainer.config.replay_parity.max_abs_logprob_diff
    assert beyond_bound is (scale == 0.3)
    assert gate["passed"] is False
    assert gate["enforced"] is beyond_bound
    if beyond_bound:
        assert failure is not None and "replay parity failed" in str(failure)
        assert tb.trainer.state.global_step == 0
        assert _unchanged(tb, before)
    else:
        assert failure is None
        assert tb.trainer.state.global_step == 1
        assert not _unchanged(tb, before)


def test_streaming_scaler_skipped_update_does_not_publish_weights(monkeypatch, tmp_path) -> None:
    """A skipped optimizer attempt counts but cannot publish unchanged weights.

    Production creates the fp16 CUDA ``GradScaler`` from the training precision;
    the CPU lane has no CUDA, so it installs torch's CPU scaler of the same
    class. Its huge initial scale overflows the real gradients, so the scaler
    skips the step and backs off exactly as on a GPU.
    """

    tb = real_trainer(monkeypatch, tmp_path, overrides=_STREAMING)
    init_scale = torch.finfo(torch.float32).max
    scaler = torch.amp.GradScaler("cpu", init_scale=init_scale)
    tb.trainer._grad_scaler = scaler
    before = _snapshot(tb)
    _seed()

    asyncio.run(
        _run_streaming_optimizer_update(
            tb.trainer, ["a cat"], batch_plan=tb.trainer.config.batch_plan
        ),
    )

    # The scaler skipped the overflowed step and backed off.
    assert scaler.get_scale() < init_scale
    assert tb.trainer.state.step == 1
    assert tb.trainer.state.global_step == 1
    assert _pushes(tb) == 1
    assert _unchanged(tb, before)


def test_phase_events_use_the_metric_step(monkeypatch, tmp_path) -> None:
    """JSON phase events and rollout stats use the same zero-based step."""

    tb = real_trainer(monkeypatch, tmp_path, overrides=("trainer.profile=true",))

    asyncio.run(tb.trainer.step(["a cat"]))

    events = _jsonl(tb, "phase_events.jsonl")
    stats = _jsonl(tb, "rollout_stats.jsonl")
    assert events
    assert {event["step"] for event in events} == {0}
    assert [row["step"] for row in stats] == [0]
    assert tb.trainer.state.step == 1


def test_streaming_profiles_training_phases(monkeypatch, tmp_path) -> None:
    """Streaming metrics and events cover replay, backward, and optimizer work."""

    tb = real_trainer(monkeypatch, tmp_path, overrides=("trainer.profile=true", *_STREAMING))

    metrics = asyncio.run(
        _run_streaming_optimizer_update(
            tb.trainer, ["a cat"], batch_plan=tb.trainer.config.batch_plan
        ),
    )

    for phase in ("evaluate", "backward", "optim_step"):
        assert metrics.phase_times[phase] > 0.0

    stats = _jsonl(tb, "rollout_stats.jsonl")
    assert len(stats) == 1
    assert stats[0]["step"] == 0
    assert all(stats[0][phase] > 0.0 for phase in ("evaluate", "backward", "optim_step"))

    training_events = {
        event["phase"]: event["step"]
        for event in _jsonl(tb, "phase_events.jsonl")
        if event["phase"] in {"evaluate", "backward", "optim_step"}
    }
    assert training_events == {"evaluate": 0, "backward": 0, "optim_step": 0}
