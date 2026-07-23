"""Tests for chunk-autoregressive Gaussian re-noise math."""

from __future__ import annotations

import math

import pytest
import torch

from vrl.math.denoise.renoise import renoise_step_with_logprob


def test_renoise_sample_and_replay_logprob_are_identical() -> None:
    x0 = torch.tensor([[[1.0, -2.0]], [[0.5, 3.0]]], dtype=torch.float32)
    noise = torch.tensor([[[0.25, -0.5]], [[1.0, -1.5]]], dtype=torch.float32)
    sigma = torch.tensor([0.2, 0.4])

    rollout = renoise_step_with_logprob(x0, sigma, noise=noise)
    replay = renoise_step_with_logprob(
        x0,
        sigma,
        next_sample=rollout.next_sample,
    )

    expected_mean = (1 - sigma[:, None, None]) * x0
    expected_action = expected_mean + sigma[:, None, None] * noise
    expected = (
        -((rollout.next_sample - expected_mean) ** 2) / (2 * sigma[:, None, None] ** 2)
        - torch.log(sigma[:, None, None])
        - 0.5 * math.log(2 * math.pi)
    ).mean(dim=(1, 2))
    assert torch.equal(rollout.next_sample, expected_action)
    assert torch.allclose(rollout.log_prob, expected)
    assert torch.equal(replay.log_prob, rollout.log_prob)


def test_renoise_replay_keeps_gradient_on_prediction_only() -> None:
    x0 = torch.tensor([[[0.2, -0.4]]], requires_grad=True)
    recorded = torch.tensor([[[0.1, 0.3]]], requires_grad=True)

    result = renoise_step_with_logprob(x0, 0.3, next_sample=recorded)
    result.log_prob.sum().backward()

    assert x0.grad is not None
    assert torch.count_nonzero(x0.grad) > 0
    assert recorded.grad is None


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_renoise_rejects_deterministic_or_reverse_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="next_sigma must be > 0"):
        renoise_step_with_logprob(torch.zeros(1, 2), sigma)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"noise": torch.zeros(1, 1)}, "noise shape"),
        ({"next_sample": torch.zeros(1, 1)}, "next_sample shape"),
    ],
)
def test_renoise_rejects_action_shape_mismatch(
    kwargs: dict[str, torch.Tensor],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        renoise_step_with_logprob(torch.zeros(2, 1), 0.2, **kwargs)


@pytest.mark.parametrize(
    ("sigma", "message"),
    [
        (torch.ones(2, 1), "scalar or sample-aligned rank-1"),
        (torch.ones(3), "length must be 1 or match"),
    ],
)
def test_renoise_rejects_invalid_sigma_shape(
    sigma: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        renoise_step_with_logprob(torch.zeros(2, 1), sigma)
