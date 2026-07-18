"""Regression tests for the quantized-rollout drift probe's trainer semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vrl.scripts.perf.quantized_rollout_drift_probe as drift_probe
from vrl.scripts.perf.quantized_rollout_drift_probe import (
    _policy_grad_norm,
    _require_precision_guard,
    _step_logprob,
)
from vrl.trainers.online.precision_guard import PrecisionDriftError


def test_step_logprob_preserves_the_denoise_axis() -> None:
    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
            [[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]],
        ],
    )
    actions = torch.tensor([[0, 2], [1, 0]])

    actual = _step_logprob(logits, actions)
    expected = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, expected)


def test_duplicate_timesteps_do_not_multiply_the_grpo_gradient() -> None:
    """The trainer averages independent timestep losses instead of summing log-ratios."""

    torch.manual_seed(0)
    samples, hidden, vocab = 4, 3, 5
    weight = torch.randn(vocab, hidden)
    activations = torch.randn(samples, 1, hidden)
    actions = torch.randint(vocab, (samples, 1))
    advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    old_log_prob = _step_logprob(activations @ weight.t(), actions).detach()

    one_step_norm, _, _ = _policy_grad_norm(
        weight=weight,
        activations=activations,
        actions=actions,
        old_log_prob=old_log_prob,
        advantages=advantages,
    )
    two_step_norm, _, _ = _policy_grad_norm(
        weight=weight,
        activations=activations.expand(-1, 2, -1).clone(),
        actions=actions.expand(-1, 2).clone(),
        old_log_prob=old_log_prob.expand(-1, 2).clone(),
        advantages=advantages,
    )

    assert two_step_norm == pytest.approx(one_step_norm, rel=1e-6, abs=1e-7)


def test_precision_guard_failure_exits_nonzero(monkeypatch, capsys) -> None:
    """A catastrophic guard failure is a failed probe, not informational output."""

    def fail_guard(*_args, **_kwargs):
        raise PrecisionDriftError("ratio exceeded limit")

    monkeypatch.setattr(drift_probe, "run_precision_drift_guard", fail_guard)

    with pytest.raises(SystemExit) as exc_info:
        _require_precision_guard(
            object(),
            scheme="fp8",
            replay_logprob=torch.zeros(2),
            rollout_logprob=torch.zeros(2),
        )

    assert exc_info.value.code == 1
    assert "FAILED: ratio exceeded limit" in capsys.readouterr().out


def test_precision_guard_preserves_quantization_as_role_execution_label(
    monkeypatch,
    capsys,
) -> None:
    captured = {}

    def pass_guard(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "violated": False,
            "worst_stats": {"ratio_abs_dev_max": 0.0},
        }

    monkeypatch.setattr(drift_probe, "run_precision_drift_guard", pass_guard)

    _require_precision_guard(
        SimpleNamespace(max_ratio_abs_dev=1.0),
        scheme="nvfp4",
        replay_logprob=torch.zeros(2),
        rollout_logprob=torch.zeros(2),
    )

    assert captured["training_precision"] == "bf16"
    assert captured["rollout_precision"] == "bf16+nvfp4"
    assert captured["training_forward_precision"] == captured["rollout_forward_precision"]
    assert "PASSED" in capsys.readouterr().out


def test_drift_probe_rejects_legacy_fp4_scheme() -> None:
    with pytest.raises(ValueError, match="scheme must be fp8 or nvfp4"):
        drift_probe.main(scheme="fp4")


def test_explicit_nvfp4_probe_fails_when_hardware_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(drift_probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(drift_probe, "nvfp4_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        drift_probe.main(scheme="nvfp4")

    assert exc_info.value.code != 0
    assert "NVFP4-capable" in str(exc_info.value)
