"""Mean-target GRPO: the loss is the squared distance to a fixed mean-space target.

Synthetic flow-SDE signals, no model; the replay density is built from a leaf
proposal mean as the evaluator builds it.
"""

from __future__ import annotations

import torch

from vrl.algorithms.grpo.continuous import (
    CenteredGRPO,
    CenteredGRPOConfig,
    MeanTargetGRPO,
    MeanTargetGRPOConfig,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.math.denoise.flow_matching import gaussian_step_log_prob
from vrl.rollouts.evaluators.types import RolloutProposalSignal, TrajectorySignalBatch

_SHAPE = (4, 1, 2, 2)
_STD, _SQRT_NEG_DT = 0.5, 0.4


def _replay(
    action: torch.Tensor, prev_sample_mean: torch.Tensor, rollout_mean: torch.Tensor
) -> TrajectorySignalBatch:
    std = torch.full((4, 1, 1, 1), _STD)
    sqrt_neg_dt = torch.full((4, 1, 1, 1), _SQRT_NEG_DT)
    log_prob = gaussian_step_log_prob(action, prev_sample_mean, std, sqrt_neg_dt=sqrt_neg_dt)
    return TrajectorySignalBatch(
        segments={
            "denoise": RolloutProposalSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=torch.zeros(4),
                mask=torch.ones(4),
                prev_sample_mean=prev_sample_mean,
                std_dev_t=std,
                dt=sqrt_neg_dt,
                sigma=torch.full_like(std, 0.5),
                old_prev_sample_mean=rollout_mean,
            ),
        },
        group_ids=torch.arange(4),
        primary_segment="denoise",
    )


def test_loss_gradient_points_from_the_mean_to_the_target() -> None:
    """d loss / d mu_theta = (mu_theta - mu*) / (s^2 d n) with mu* = mu_q + c (a - mu_q),
    at a replay mean that has already left the rollout mean."""

    torch.manual_seed(0)
    rollout_mean = torch.randn(_SHAPE)
    action = rollout_mean + 0.2 * torch.randn(_SHAPE)
    advantages = torch.tensor([1.0, -0.5, 0.25, -2.0])
    step_size = 0.5
    mean = (rollout_mean + 0.1 * torch.randn(_SHAPE)).requires_grad_(True)

    loss, _ = MeanTargetGRPO(MeanTargetGRPOConfig(step_size=step_size)).compute_loss(
        AlgorithmInput(advantages=advantages, signals=_replay(action, mean, rollout_mean))
    )
    loss.backward()

    coefficient = (step_size * advantages).view(4, 1, 1, 1)
    target = rollout_mean + coefficient * (action - rollout_mean)
    expected = (mean.detach() - target) / ((_STD * _SQRT_NEG_DT) ** 2 * 4) / 4
    torch.testing.assert_close(mean.grad, expected, rtol=1e-4, atol=1e-6)


def test_first_gradient_is_the_centered_score_scaled_by_the_step() -> None:
    torch.manual_seed(1)
    rollout_mean = torch.randn(_SHAPE)
    action = rollout_mean + 0.2 * torch.randn(_SHAPE)
    advantages = torch.tensor([1.0, -0.5, 0.25, -2.0])

    def grad_of(algorithm) -> torch.Tensor:
        mean = rollout_mean.clone().requires_grad_(True)
        loss, _ = algorithm.compute_loss(
            AlgorithmInput(advantages=advantages, signals=_replay(action, mean, rollout_mean))
        )
        loss.backward()
        return mean.grad

    torch.testing.assert_close(
        grad_of(MeanTargetGRPO(MeanTargetGRPOConfig(step_size=0.5))),
        0.5 * grad_of(CenteredGRPO(CenteredGRPOConfig())),
        rtol=1e-4,
        atol=1e-6,
    )
