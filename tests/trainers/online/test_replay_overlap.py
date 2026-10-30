"""Scheduling must preserve real GRPO gradients, optimizer state and updates."""

from __future__ import annotations

import asyncio

import pytest
import torch

from tests.rollouts.collector._helpers import IndexReward
from tests.trainers.online._helpers import real_trainer
from vrl.scripts.perf.replay_overlap import compare_updates, replay_schedule


def test_deferred_weight_gradients_match_the_real_grpo_update(monkeypatch, tmp_path):
    tb = real_trainer(
        monkeypatch,
        tmp_path,
        reward=IndexReward(),
        overrides=(
            "actor.training_microbatch_size=1",
            "actor.ppo_epochs=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
    monkeypatch.setattr(tb.trainer._rollout_runtime, "weight_syncer", None)
    before = {n: p.detach().clone() for n, p in tb.trainable_parameters().items()}
    report = compare_updates(
        tb.trainer,
        batch,
        modes=("serial", "deferred_wgrad"),
        warmup=0,
        repeats=1,
        atol=1e-6,
        rtol=1e-4,
    )
    assert report["passed"]
    assert report["arms"]["deferred_wgrad"]["schedule"]["queued_bytes_peak"] > 0
    for name, parameter in tb.trainable_parameters().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert "_clip_and_step" not in tb.trainer.__dict__


def test_failed_replay_restores_methods_without_applying_pending_gradients(monkeypatch, tmp_path):
    tb = real_trainer(
        monkeypatch, tmp_path, overrides=("actor.ema.enable=false", "algorithm.kl_coef=0.0")
    )
    linear = next(m for m in tb.model.transformer.modules() if type(m) is torch.nn.Linear)
    weight = linear.weight.detach().clone()
    with (
        pytest.raises(RuntimeError, match="injected failure"),
        replay_schedule(tb.trainer, mode="deferred_wgrad"),
    ):
        inputs = torch.randn(2, linear.in_features)
        linear(input=inputs).sum().backward()
        assert linear.weight.grad is None
        raise RuntimeError("injected failure")
    assert "forward" not in linear.__dict__
    assert "_clip_and_step" not in tb.trainer.__dict__
    torch.testing.assert_close(linear.weight, weight, rtol=0, atol=0)
    linear(torch.randn(2, linear.in_features)).sum().backward()
    assert linear.weight.grad is not None


def test_comparison_rejects_incorrect_delayed_gradients(monkeypatch, tmp_path):
    from vrl.scripts.perf import replay_overlap

    tb = real_trainer(
        monkeypatch,
        tmp_path,
        reward=IndexReward(),
        overrides=(
            "actor.training_microbatch_size=1",
            "actor.ppo_epochs=1",
            "actor.ema.enable=false",
            "algorithm.kl_coef=0.0",
        ),
    )
    batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
    monkeypatch.setattr(tb.trainer._rollout_runtime, "weight_syncer", None)
    actual = replay_overlap._DeferredLinear.backward

    def broken(ctx, grad_output):
        return actual(ctx, grad_output * 0)

    monkeypatch.setattr(replay_overlap._DeferredLinear, "backward", broken)
    with pytest.raises((AssertionError, RuntimeError)):
        compare_updates(
            tb.trainer,
            batch,
            modes=("serial", "deferred_wgrad"),
            warmup=0,
            repeats=1,
            atol=1e-6,
            rtol=1e-4,
        )
    assert "_clip_and_step" not in tb.trainer.__dict__


def test_unsupported_stream_arm_fails_before_warmup_mutates_weights(monkeypatch, tmp_path):
    tb = real_trainer(
        monkeypatch, tmp_path, overrides=("actor.ema.enable=false", "algorithm.kl_coef=0.0")
    )
    batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
    monkeypatch.setattr(tb.trainer._rollout_runtime, "weight_syncer", None)
    before = {n: p.detach().clone() for n, p in tb.trainable_parameters().items()}
    with pytest.raises(ValueError, match="requires CUDA"):
        compare_updates(
            tb.trainer,
            batch,
            modes=("serial", "streams"),
            warmup=2,
            repeats=1,
            atol=1e-6,
            rtol=1e-4,
        )
    for name, parameter in tb.trainable_parameters().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_autocast_delayed_dw_updates_fp32_masters_and_preserves_input_gradient(
    monkeypatch, tmp_path
):
    import copy

    tb = real_trainer(
        monkeypatch, tmp_path, overrides=("actor.ema.enable=false", "algorithm.kl_coef=0.0")
    )
    linear = next(
        m
        for m in tb.model.transformer.modules()
        if type(m) is torch.nn.Linear and m.bias is not None
    )
    reference = copy.deepcopy(linear)
    inputs = torch.randn(2, 3, linear.in_features, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference(reference_inputs)
    expected.float().square().mean().backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), tb.trainer.config.max_norm)
    reference_optimizer.step()
    with replay_schedule(tb.trainer, mode="deferred_wgrad"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = linear(input=inputs)
        actual.float().square().mean().backward()
        tb.trainer._clip_and_step(optimizer)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(linear.weight, reference.weight)
    torch.testing.assert_close(linear.bias, reference.bias)
