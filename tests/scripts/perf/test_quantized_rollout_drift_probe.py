"""Regression tests for the quantized-rollout drift probe's trainer semantics."""

from __future__ import annotations

import pytest
import torch

import vrl.scripts.perf.quantized_rollout_drift_probe as drift_probe
from vrl.scripts.perf.quantized_rollout_drift_probe import (
    _policy_grad_norm,
    _step_logprob,
)


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


def test_drift_probe_rejects_legacy_fp4_scheme(monkeypatch, capsys) -> None:
    """argparse choices own scheme validation now that the CLI is the only entry."""

    monkeypatch.setattr("sys.argv", ["quantized_rollout_drift_probe", "--scheme", "fp4"])

    with pytest.raises(SystemExit) as exc_info:
        drift_probe.main()

    assert exc_info.value.code != 0
    assert "fp8" in capsys.readouterr().err


def test_explicit_nvfp4_probe_fails_when_hardware_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["quantized_rollout_drift_probe", "--scheme", "nvfp4"])
    monkeypatch.setattr(drift_probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(drift_probe, "nvfp4_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        drift_probe.main()

    assert exc_info.value.code != 0
    assert "NVFP4-capable" in str(exc_info.value)
