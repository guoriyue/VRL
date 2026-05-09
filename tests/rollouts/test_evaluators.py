"""Tests for rollout evaluator replay ownership."""

from __future__ import annotations

import contextlib
import types
from typing import Any

import pytest
import torch

import vrl.rollouts.evaluators.diffusion.flow_matching as fm
from vrl.algorithms.flow_matching import SDEStepResult
from vrl.rollouts.evaluators.types import SignalRequest


class _ReplayPolicy:
    def __init__(self, name: str = "policy") -> None:
        self.name = name
        self.disabled = False
        self.calls: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def replay_forward(self, batch: Any, timestep_idx: int) -> dict[str, torch.Tensor]:
        self.calls.append((self.name, self.disabled))
        return {"noise_pred": torch.zeros_like(batch.observations[:, timestep_idx])}


def _batch() -> Any:
    return types.SimpleNamespace(
        observations=torch.randn(2, 1, 4),
        actions=torch.randn(2, 1, 4),
        extras={"timesteps": torch.tensor([[0.8], [0.8]])},
    )


def _fake_sde_step_with_logprob(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    **kwargs: Any,
) -> SDEStepResult:
    del scheduler, timestep
    return SDEStepResult(
        prev_sample=sample,
        log_prob=torch.zeros(sample.shape[0]),
        prev_sample_mean=model_output,
        std_dev_t=torch.ones(sample.shape[0]),
        dt=torch.ones(sample.shape[0]) if kwargs.get("return_dt") else None,
    )


def test_flow_matching_evaluator_replays_through_policy_model(monkeypatch) -> None:
    monkeypatch.setattr(
        fm.flow_matching_math,
        "sde_step_with_logprob",
        _fake_sde_step_with_logprob,
    )
    policy = _ReplayPolicy()

    fm.FlowMatchingEvaluator(scheduler=None).evaluate(
        model=policy,
        batch=_batch(),
        timestep_idx=0,
        signal_request=SignalRequest(need_ref=False),
    )

    assert policy.calls == [("policy", False)]


def test_flow_matching_evaluator_requires_policy_owned_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        fm.flow_matching_math,
        "sde_step_with_logprob",
        _fake_sde_step_with_logprob,
    )

    with pytest.raises(AttributeError, match="replay_forward"):
        fm.FlowMatchingEvaluator(scheduler=None).evaluate(
            model=object(),
            batch=_batch(),
            timestep_idx=0,
            signal_request=SignalRequest(need_ref=False),
        )


def test_lora_reference_replay_disables_adapter_on_trainable_policy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fm.flow_matching_math,
        "sde_step_with_logprob",
        _fake_sde_step_with_logprob,
    )
    policy = _ReplayPolicy()

    signals = fm.FlowMatchingEvaluator(scheduler=None).evaluate(
        model=policy,
        batch=_batch(),
        timestep_idx=0,
        ref_model=policy,
        signal_request=SignalRequest(need_ref=True, need_kl_intermediates=True),
    )

    assert policy.calls == [("policy", False), ("policy", True)]
    assert signals.ref_log_prob is not None


def test_explicit_reference_policy_owns_its_own_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        fm.flow_matching_math,
        "sde_step_with_logprob",
        _fake_sde_step_with_logprob,
    )
    policy = _ReplayPolicy("policy")
    ref_policy = _ReplayPolicy("ref")

    signals = fm.FlowMatchingEvaluator(scheduler=None).evaluate(
        model=policy,
        batch=_batch(),
        timestep_idx=0,
        ref_model=ref_policy,
        signal_request=SignalRequest(need_ref=True, need_kl_intermediates=True),
    )

    assert policy.calls == [("policy", False)]
    assert ref_policy.calls == [("ref", False)]
    assert signals.ref_log_prob is not None
