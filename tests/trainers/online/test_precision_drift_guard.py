"""Tests for the precision drift guard (rollout-vs-replay logprob parity)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import _stamp_model_precision
from vrl.algorithms.logprob_mismatch import compute_logprob_mismatch_stats
from vrl.trainers.core.types import PrecisionDriftGuardConfig
from vrl.trainers.online.precision_guard import (
    PrecisionDriftError,
    measure_precision_drift,
    resolve_guard_mode,
    run_precision_drift_guard,
    select_guard_timesteps,
)


def _signals(delta: float):
    """A signals-like object: fresh = old + delta on every element."""

    return SimpleNamespace(
        primary=SimpleNamespace(
            log_prob=torch.full((4,), delta),
            old_log_prob=torch.zeros(4),
        )
    )


def _eval_with_drift(delta: float):
    return lambda _timestep: _signals(delta)


def _run(config, *, train, rollout, evaluate_fn, **kw):
    return run_precision_drift_guard(
        config,
        training_precision=train,
        rollout_precision=rollout,
        math_precision="fp32",
        timestep_indices=[0, 1, 2],
        evaluate_fn=evaluate_fn,
        **kw,
    )


# -- resolve_guard_mode ----------------------------------------------------


def test_auto_enables_fail_for_rollout_compute_mismatch() -> None:
    assert (
        resolve_guard_mode(
            "auto",
            training_precision="fp32",
            rollout_precision="bf16",
        )
        == "fail"
    )
    assert (
        resolve_guard_mode(
            "auto",
            training_precision="fp32",
            rollout_precision="fp16",
        )
        == "fail"
    )


def test_auto_is_off_for_same_dtype() -> None:
    assert (
        resolve_guard_mode(
            "auto",
            training_precision="fp32",
            rollout_precision="fp32",
        )
        == "off"
    )


def test_auto_detects_rollout_quantization() -> None:
    assert (
        resolve_guard_mode(
            "auto",
            training_precision="bf16",
            rollout_precision="bf16+fp8",
        )
        == "fail"
    )


def test_auto_normalizes_legacy_no_to_fp32() -> None:
    # Both public spellings resolve to one concrete no-autocast contract.
    assert (
        resolve_guard_mode(
            "auto",
            training_precision="no",
            rollout_precision="fp32",
        )
        == "off"
    )


def test_explicit_modes_pass_through_regardless_of_precision() -> None:
    for mode in ("off", "warn", "fail"):
        assert (
            resolve_guard_mode(
                mode,
                training_precision="fp32",
                rollout_precision="fp32",
            )
            == mode
        )


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="auto/off/warn/fail"):
        resolve_guard_mode(
            "loud",
            training_precision="fp32",
            rollout_precision="bf16",
        )


# -- run_precision_drift_guard ---------------------------------------------


def test_precision_drift_guard_passes_when_within_threshold() -> None:
    cfg = PrecisionDriftGuardConfig(mode="fail")
    record = _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_eval_with_drift(0.0))
    assert record is not None
    assert record["violated"] is False


def test_precision_drift_guard_checks_fp16_same_role_precision() -> None:
    cfg = PrecisionDriftGuardConfig(mode="fail")
    record = _run(
        cfg,
        train="fp16",
        rollout="fp16",
        evaluate_fn=_eval_with_drift(0.0),
        metadata={"trainer_transformer_dtype": "float16"},
    )
    assert record is not None
    assert record["mode"] == "fail"
    assert record["role_precision_match"] is True
    assert record["violated"] is False
    assert record["metadata"]["trainer_transformer_dtype"] == "float16"


def test_precision_drift_guard_fails_before_optimizer_when_ratio_drifts() -> None:
    cfg = PrecisionDriftGuardConfig(
        mode="fail",
        max_abs_log_ratio=1e-3,
        max_ratio_abs_dev=1e-3,
    )
    with pytest.raises(PrecisionDriftError, match="precision drift guard"):
        _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_eval_with_drift(0.05))


def test_precision_drift_guard_warns_without_failing(caplog) -> None:
    cfg = PrecisionDriftGuardConfig(mode="warn")
    with caplog.at_level(logging.WARNING):
        record = _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_eval_with_drift(0.05))
    assert record["violated"] is True
    assert any("precision drift guard" in r.getMessage() for r in caplog.records)


def test_precision_drift_guard_auto_enables_for_rollout_compute_mismatch() -> None:
    cfg = PrecisionDriftGuardConfig(mode="auto")
    record = _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_eval_with_drift(0.0))
    assert record is not None
    assert record["mode"] == "fail"
    assert record["violated"] is False


def test_precision_drift_guard_auto_fails_on_rollout_compute_mismatch_drift() -> None:
    cfg = PrecisionDriftGuardConfig(
        mode="auto",
        max_abs_log_ratio=1e-3,
        max_ratio_abs_dev=1e-3,
    )
    with pytest.raises(PrecisionDriftError, match="precision drift guard"):
        _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_eval_with_drift(0.05))


def test_precision_drift_guard_auto_is_off_for_same_dtype_without_running_eval() -> None:
    def _must_not_run(_timestep):
        raise AssertionError("evaluate_fn must not be called when the guard is off")

    cfg = PrecisionDriftGuardConfig(mode="auto")
    record = _run(cfg, train="fp32", rollout="fp32", evaluate_fn=_must_not_run)
    assert record is None


def test_precision_drift_guard_flags_nonfinite() -> None:
    cfg = PrecisionDriftGuardConfig(mode="fail", fail_on_nonfinite=True)

    def _inf(_timestep):
        return SimpleNamespace(
            primary=SimpleNamespace(
                log_prob=torch.tensor([0.0, float("inf")]),
                old_log_prob=torch.zeros(2),
            )
        )

    with pytest.raises(PrecisionDriftError):
        _run(cfg, train="fp32", rollout="bf16", evaluate_fn=_inf)


def test_measure_precision_drift_selects_one_whole_worst_rank_record(monkeypatch) -> None:
    """Distributed provenance must not splice maxima into another rank's record."""

    cfg = PrecisionDriftGuardConfig(
        mode="warn",
        max_abs_log_ratio=0.1,
        max_ratio_abs_dev=0.1,
    )
    remote_stats = compute_logprob_mismatch_stats(
        torch.full((4,), -5.0),
        torch.zeros(4),
    )

    dist = torch.distributed
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def _all_gather_object(gathered, local):
        remote = dict(local)
        remote.update(
            {
                "violated": True,
                "worst_rank": 1,
                "worst_timestep": 17,
                "worst_stats": asdict(remote_stats),
                "metadata": {"rank_marker": "remote"},
                "rank_marker": "remote",
            }
        )
        gathered[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", _all_gather_object)

    record = measure_precision_drift(
        cfg,
        training_precision="fp32",
        rollout_precision="bf16",
        math_precision="fp32",
        timestep_indices=[3],
        evaluate_fn=_eval_with_drift(1.0),
        metadata={"rank_marker": "local"},
    )

    assert record is not None
    assert record["world_size"] == 2
    assert record["worst_rank"] == 1
    assert record["worst_timestep"] == 17
    assert record["rank_marker"] == "remote"
    assert record["metadata"] == {"rank_marker": "remote"}
    assert record["worst_stats"] == asdict(remote_stats)
    # Rank 0 has the larger ratio max (exp(1)-1), while rank 1 has the much
    # larger abs-log drift. Returning rank 1's smaller ratio proves the record
    # was selected whole instead of receiving a field-wise global maximum.
    local_ratio_max = float(torch.exp(torch.tensor(1.0)) - 1.0)
    assert record["worst_stats"]["ratio_abs_dev_max"] < local_ratio_max


# -- select_guard_timesteps ------------------------------------------------


def test_select_guard_timesteps_picks_first_last_within_count() -> None:
    picks = select_guard_timesteps(list(range(10)), 3)
    assert len(picks) == 3
    assert picks[0] == 0
    assert picks[-1] == 9
    assert picks == sorted(set(picks))


def test_select_guard_timesteps_returns_all_when_fewer_than_max() -> None:
    assert select_guard_timesteps([0, 1], 3) == [0, 1]


def test_select_guard_timesteps_single() -> None:
    assert select_guard_timesteps([0, 3, 7], 1) == [0]


def test_online_trainer_precision_guard_fails_before_optimizer_when_ratio_drifts() -> None:
    """OnlineTrainer should fail before loss/backward/state update."""

    import asyncio

    from tests.trainers.online._helpers import _trajectory_signals
    from vrl.algorithms.types import TrainStepMetrics
    from vrl.rollouts.batch import RolloutBatch
    from vrl.rollouts.evaluators.base import Evaluator
    from vrl.trainers.core.types import EMAConfig, OptimConfig
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    class _Algorithm:
        class _Config:
            global_std = False
            eps = 1e-8
            adv_clip_max = 5.0
            kl_coef = 0.0

        config = _Config()

        def __init__(self) -> None:
            self.loss_calls = 0

        def compute_advantages_from_tensors(self, rewards, group_ids):
            del group_ids
            return rewards - rewards.mean()

        def compute_loss(self, inputs):
            del inputs
            self.loss_calls += 1
            loss = torch.tensor(0.0, requires_grad=True)
            return loss, TrainStepMetrics()

    class _Collector(CollectorControlFake):
        async def score_rollouts(self, pendings):
            return list(pendings)

        async def collect_unscored(self, prompts, **kwargs):
            group_size = int(kwargs["group_size"])
            return RolloutBatch(
                observations=torch.zeros(group_size, 2, 1),
                actions=torch.zeros(group_size, 2, 1),
                rewards=torch.arange(group_size, dtype=torch.float32),
                group_ids=torch.zeros(group_size, dtype=torch.long),
                context={},
            )

    class _Evaluator(Evaluator):
        def evaluate(self, model, batch, timestep_idx, **kw):
            del kw
            fresh = model.weight.view(1).expand(batch.rewards.shape[0])
            return _trajectory_signals(batch, fresh, timestep_idx)

    algorithm = _Algorithm()
    model = torch.nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    with torch.no_grad():
        model.weight.fill_(0.05)
    before = model.weight.detach().clone()
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            output_dir="outputs/",
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            train_precision="no",
            rollout_precision="bf16",
        ),
        device="cpu",
    )

    with pytest.raises(PrecisionDriftError, match="precision drift guard"):
        asyncio.run(trainer.step(["prompt-a"]))

    assert algorithm.loss_calls == 0
    assert trainer.state.step == 0
    assert trainer.state.global_step == 0
    torch.testing.assert_close(model.weight, before)
