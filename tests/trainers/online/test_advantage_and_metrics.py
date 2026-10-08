"""OnlineTrainer advantage normalization, metric aggregation, and zero-advantage gradients.

Every trainer is ``real_trainer`` (tiny SANA, real GRPO and SDE evaluator).
Rewards are real ``RewardFunction``s with chosen scores; what the algorithm
saw and returned is read off the traced real calls.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from tests.rollouts.collector._helpers import IndexReward
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.rewards import RewardOutput, RewardSample
from vrl.trainers.online.trainer import TrainingBatch


class _ScriptedReward(IndexReward):
    """Hands out the next scripted score per sample, plus an ``observer`` component."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self._scores = list(scores)
        self._cursor = 0

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await super().score_batch(samples)
        scores = tuple(self._scores[self._cursor : self._cursor + len(samples)])
        self._cursor += len(samples)
        return RewardOutput(
            scores=scores,
            components={"observer": tuple(score + 10.0 for score in scores)},
        )


def _trainer(monkeypatch, tmp_path, *overrides: str, reward=None) -> TrainerBench:
    tb = real_trainer(monkeypatch, tmp_path, reward=reward, overrides=tuple(overrides))
    tb.collector.trace.watch(tb.trainer.algorithm, "compute_loss", "compute_loss")
    return tb


def _loss_metrics(tb: TrainerBench) -> list:
    """The ``TrainStepMetrics`` every real ``compute_loss`` call returned."""

    return [metrics for _loss, metrics in tb.collector.trace.results.get("compute_loss", [])]


def _shift_rollout_log_prob(batch: TrainingBatch, deltas: torch.Tensor) -> None:
    """The rollout recorded log-probs the replay disagrees with by ``deltas`` per sample."""

    for rollout in batch.batches:
        old_log_prob = rollout.trajectory.segments["denoise"].tensors["old_log_prob"].value
        old_log_prob[:, 0] += deltas


class TestAdvantageAndMetrics:
    """Groups tests for advantage and metrics."""

    def test_admission_audit_failure_prevents_backward_on_local_and_peer_rank(
        self, monkeypatch, tmp_path
    ):
        local = _trainer(monkeypatch, tmp_path / "local")
        # The audit file cannot be created: a directory already holds its path.
        local.trainer.admission_ledger.path.mkdir(parents=True)
        with pytest.raises(OSError):
            asyncio.run(local.trainer.step(["a cat"]))
        assert "compute_loss" not in local.collector.trace.events

        peer = _trainer(monkeypatch, tmp_path / "peer")
        # Another rank reports a failed admission through the agreement
        # collective; a single-process strategy has no peer to ask, so its
        # answer is injected at that boundary.
        monkeypatch.setattr(peer.strategy.collectives, "all_true", lambda value: False)
        with pytest.raises(RuntimeError, match="another training rank"):
            asyncio.run(peer.trainer.step(["a cat"]))
        assert "compute_loss" not in peer.collector.trace.events

    def test_cea_step_advantages_independent_across_steps(self, monkeypatch, tmp_path) -> None:
        """Second-step advantages are normalized against the current group only,
        with no state leaking from previous steps."""

        tb = _trainer(
            monkeypatch,
            tmp_path,
            "actor.drop_zero_advantage=false",
            reward=_ScriptedReward([0.0, 0.0, 0.0, 1.0]),
        )
        tb.collector.trace.watch(
            tb.trainer.algorithm, "compute_advantages_from_components", "component_advantages"
        )

        asyncio.run(tb.trainer.step(["a cat"]))
        second_step = asyncio.run(tb.trainer.step(["a cat"]))

        # Advantages are computed purely from the current group: no stale history.
        assert second_step.advantage_mean == pytest.approx(0.0, abs=1e-3)
        # Component metrics belong to the consumed second batch only. The first
        # step's observations must not accumulate on a shared reward object.
        assert second_step.reward_components == {"observer": pytest.approx(10.5)}
        assert tb.collector.trace.events.count("component_advantages") == 2

    def test_cea_metrics_propagate_approx_kl(self, monkeypatch, tmp_path) -> None:
        """CEA aggregation must not silently drop approx_kl."""

        tb = _trainer(monkeypatch, tmp_path)

        metrics = asyncio.run(tb.trainer.step(["a cat"]))

        per_call = [m.update.approx_kl for m in _loss_metrics(tb)]
        # Four PPO epochs: the later ones replay a moved policy.
        assert len(per_call) == tb.trainer.config.ppo_epochs
        assert any(value != 0.0 for value in per_call)
        assert metrics.update.approx_kl == pytest.approx(statistics.fmean(per_call))

    def test_cea_metrics_capture_the_complete_first_optimizer_update(
        self, monkeypatch, tmp_path
    ) -> None:
        """Initial replay covers every replay unit in the first optimizer update."""

        tb = _trainer(
            monkeypatch,
            tmp_path,
            "actor.prompts_per_collection=1",
            "actor.ppo_epochs=1",
            # The test's own rollout/replay disagreement is the measured signal.
            "trainer.replay_parity.max_abs_logprob_diff=1.0",
        )
        batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
        _shift_rollout_log_prob(batch, torch.tensor([1e-4, 3e-4]))
        two_boundaries = replace(
            batch,
            batches=batch.batches * 2,
            advantages=batch.advantages * 2,
        )

        metrics = asyncio.run(tb.trainer.train_on_rollout_batch(two_boundaries))

        per_call = _loss_metrics(tb)
        assert len(per_call) == 2

        def mean(read):
            return pytest.approx(statistics.fmean(read(m) for m in per_call))

        assert metrics.update.clip_fraction == mean(lambda m: m.update.clip_fraction)
        assert metrics.update.clip_fraction > 0.0
        assert metrics.initial_replay.clip_fraction == mean(lambda m: m.update.clip_fraction)
        assert metrics.update.active_clip_fraction == mean(lambda m: m.update.active_clip_fraction)
        assert metrics.initial_replay.active_clip_fraction == mean(
            lambda m: m.update.active_clip_fraction
        )
        assert metrics.logprob_mismatch.logprob_abs_diff_max == pytest.approx(3e-4, rel=1e-3)
        assert metrics.initial_replay.logprob_abs_diff_max == pytest.approx(3e-4, rel=1e-3)
        assert metrics.initial_replay.finite is True
        assert metrics.weighted_kl_loss == mean(lambda m: m.weighted_kl_loss)
        assert metrics.update.tis_clip_fraction == mean(lambda m: m.update.tis_clip_fraction)
        assert metrics.update.rs_seq_masked_fraction == mean(
            lambda m: m.update.rs_seq_masked_fraction
        )

    def test_replay_metrics_follow_uneven_sample_chunk_weights(
        self, monkeypatch, tmp_path
    ) -> None:
        """An 8+2 split represents 80%+20% of the optimized group, not 50%+50%."""

        from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
        from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
        from vrl.trainers.online.trainer import _ReplayMetrics, _TrainingMicrobatch

        tb = real_trainer(
            monkeypatch,
            tmp_path,
            overrides=(
                "rollout.n_samples_per_prompt=10",
                "rollout.samples_per_generation_batch=10",
            ),
        )
        group = asyncio.run(tb.trainer.collect_training_batch(["a cat"])).batches[0]
        assert group.rewards.shape == (10,)

        batches = _TrainingMicrobatch.from_prompt_group(
            group, torch.ones(10), training_microbatch_size=8
        )
        assert [batch.loss_weight for batch in batches] == pytest.approx([0.8, 0.2])

        aggregate = _ReplayMetrics()
        for value, batch in zip((1.0, 9.0), batches, strict=True):
            aggregate.add(
                TrainStepMetrics(
                    loss=value,
                    policy_loss=value,
                    update=PolicyUpdateStats(clip_fraction=value),
                    logprob_mismatch=LogprobMismatchStats(
                        logprob_abs_diff_mean=value,
                        logprob_abs_diff_max=value,
                    ),
                ),
                weight=batch.loss_weight,
                capture_initial_replay=True,
            )

        initial_replay, initial_weight = aggregate.initial_replay_snapshot()
        metrics = aggregate.build(
            reward_mean=0.0,
            reward_std=0.0,
            reward_components={},
            advantage_mean=0.0,
            adv_saturation=0.0,
            adv_zero_rate=0.0,
            group_size=10.0,
            trained_prompt_num=1,
            phase_times={},
            initial_replay=initial_replay,
        )

        assert initial_weight == pytest.approx(1.0)
        assert metrics.loss == pytest.approx(2.6)
        assert metrics.update.clip_fraction == pytest.approx(2.6)
        assert metrics.logprob_mismatch.logprob_abs_diff_mean == pytest.approx(2.6)
        assert metrics.logprob_mismatch.logprob_abs_diff_max == pytest.approx(9.0)
        assert metrics.initial_replay.clip_fraction == pytest.approx(2.6)
        assert metrics.initial_replay.logprob_abs_diff_max == pytest.approx(9.0)

    def test_zero_advantage_samples_do_not_get_epsilon_gradient(
        self, monkeypatch, tmp_path
    ) -> None:
        """All-zero advantages skip backward instead of inventing gradients."""

        tb = _trainer(monkeypatch, tmp_path, reward=_ScriptedReward([1.0, 1.0]))
        assert tb.trainer.config.drop_zero_advantage is True
        before = {
            name: value.detach().clone() for name, value in tb.trainable_parameters().items()
        }

        metrics = asyncio.run(tb.trainer.step(["a cat"]))

        assert "compute_loss" not in tb.collector.trace.events
        assert tb.trainer.state.step == 1
        assert tb.trainer.state.global_step == 0
        assert metrics.grad_norm == 0.0
        assert metrics.group_size == 2.0
        assert metrics.trained_prompt_num == 1
        assert all(
            torch.equal(before[name], value) for name, value in tb.trainable_parameters().items()
        )
