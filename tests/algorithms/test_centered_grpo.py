"""Centered GRPO: the gradient is the score of the draw around the rollout mean.

Synthetic flow-SDE signals, no model. The replay density is built from a leaf
proposal mean exactly as the evaluator builds it (``gaussian_step_log_prob`` on
the stored action), so the gradient the loss puts on that mean is the theorem
under test.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from vrl.algorithms.grpo.continuous import (
    GRPO,
    CenteredGRPO,
    CenteredGRPOConfig,
    GRPOConfig,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.math.denoise.flow_matching import gaussian_step_log_prob
from vrl.rollouts.evaluators.types import (
    FlowSDESignal,
    RolloutProposalSignal,
    TrajectorySignalBatch,
)

_SHAPE = (4, 1, 2, 2)
_D = 4  # latent elements per sample


def _replay(
    *,
    action: torch.Tensor,
    prev_sample_mean: torch.Tensor,
    old_prev_sample_mean: torch.Tensor,
    old_log_prob: torch.Tensor,
    sde_type: str,
) -> TrajectorySignalBatch:
    std = torch.full((4, 1, 1, 1), 0.5)
    sqrt_neg_dt = torch.full((4, 1, 1, 1), 0.4)
    log_prob = gaussian_step_log_prob(
        action, prev_sample_mean, std, sqrt_neg_dt=sqrt_neg_dt, sde_type=sde_type
    )
    return TrajectorySignalBatch(
        segments={
            "denoise": RolloutProposalSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=torch.ones(4),
                prev_sample_mean=prev_sample_mean,
                std_dev_t=std,
                dt=sqrt_neg_dt,
                sigma=torch.full_like(std, 0.5),
                old_prev_sample_mean=old_prev_sample_mean,
            ),
        },
        group_ids=torch.arange(4),
        primary_segment="denoise",
    )


def _centered(sde_type: str) -> CenteredGRPO:
    config = CenteredGRPOConfig()
    config.sde_type = sde_type
    return CenteredGRPO(config)


@pytest.mark.parametrize(
    ("sde_type", "scale_sq"),
    [("flow_grpo", (0.5 * 0.4) ** 2), ("cps", 0.5)],
)
def test_gradient_is_the_score_of_the_draw_and_ignores_the_replay_offset(
    sde_type: str, scale_sq: float
) -> None:
    """d loss / d mu_theta = -A (a - mu_q) / (s^2 d) / n, whatever mu_theta - mu_q is.

    ``cps`` scores the unnormalized squared distance, so its "s^2" is 1/2.
    """

    torch.manual_seed(0)
    mu_q = torch.randn(_SHAPE)
    action = mu_q + 0.2 * torch.randn(_SHAPE)
    advantages = torch.tensor([1.0, -0.5, 0.25, -1.0])
    expected = -advantages.view(4, 1, 1, 1) * (action - mu_q) / (scale_sq * _D) / 4

    grads = []
    for offset in (0.0, 0.3):
        mu_theta = (mu_q + offset).requires_grad_(True)
        signals = _replay(
            action=action,
            prev_sample_mean=mu_theta,
            old_prev_sample_mean=mu_q,
            old_log_prob=torch.zeros(4),
            sde_type=sde_type,
        )
        loss, _ = _centered(sde_type).compute_loss(
            AlgorithmInput(advantages=advantages, signals=signals)
        )
        loss.backward()
        grads.append(mu_theta.grad.clone())

    torch.testing.assert_close(grads[0], expected)
    torch.testing.assert_close(grads[1], expected)


def test_parity_gradient_equals_plain_grpo_at_ratio_one() -> None:
    """With mu_theta == mu_q and the stored log-prob equal to the replay's, the
    centered score and GRPO's clipped surrogate put the same gradient on the mean."""

    torch.manual_seed(1)
    mu_q = torch.randn(_SHAPE)
    action = mu_q + 0.2 * torch.randn(_SHAPE)
    advantages = torch.tensor([1.0, -0.5, 0.25, -1.0])

    def grad_of(algorithm) -> torch.Tensor:
        mu_theta = mu_q.clone().requires_grad_(True)
        signals = _replay(
            action=action,
            prev_sample_mean=mu_theta,
            old_prev_sample_mean=mu_q,
            old_log_prob=torch.zeros(4),
            sde_type="flow_grpo",
        )
        signals.primary.old_log_prob = signals.primary.log_prob.detach()
        loss, _ = algorithm.compute_loss(AlgorithmInput(advantages=advantages, signals=signals))
        loss.backward()
        return mu_theta.grad

    torch.testing.assert_close(
        grad_of(_centered("flow_grpo")),
        grad_of(GRPO(GRPOConfig(clip_ratio=1e-4))),
    )


def test_proposal_gap_metric_is_the_mean_space_kl() -> None:
    """approx_kl reports |mu_theta - mu_q|^2 / (2 s^2) per element, without the draw's noise."""

    mu_q = torch.zeros(_SHAPE)
    mu_theta = torch.full(_SHAPE, 0.1)
    signals = _replay(
        action=mu_q,
        prev_sample_mean=mu_theta,
        old_prev_sample_mean=mu_q,
        old_log_prob=torch.zeros(4),
        sde_type="flow_grpo",
    )
    _, metrics = _centered("flow_grpo").compute_loss(
        AlgorithmInput(advantages=torch.ones(4), signals=signals)
    )
    assert metrics.update.approx_kl == pytest.approx(0.1**2 / (2 * (0.5 * 0.4) ** 2))
    assert metrics.update.clip_fraction == 0.0


def test_loss_refuses_a_replay_without_the_rollout_mean() -> None:
    signals = _replay(
        action=torch.zeros(_SHAPE),
        prev_sample_mean=torch.zeros(_SHAPE),
        old_prev_sample_mean=torch.zeros(_SHAPE),
        old_log_prob=torch.zeros(4),
        sde_type="flow_grpo",
    )
    without_mean = {f.name: getattr(signals.primary, f.name) for f in fields(FlowSDESignal)}
    signals.segments["denoise"] = FlowSDESignal(**without_mean)
    with pytest.raises(ValueError, match="return_prev_sample_mean"):
        _centered("flow_grpo").compute_loss(
            AlgorithmInput(advantages=torch.ones(4), signals=signals)
        )
