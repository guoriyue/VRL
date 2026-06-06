"""Tests for continuous GRPO advantages + loss."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

# ---------------------------------------------------------------------------
# Regression: single-sample GRPO advantage must NOT be NaN
# ---------------------------------------------------------------------------


class TestGRPOSingleSampleNaN:
    """Groups tests for grposingle sample na n."""
    def test_single_sample_returns_zero_not_nan(self) -> None:
        """Single sample per group → advantage = 0.0, NOT NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([5.0])
        group_ids = torch.tensor([0])
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any(), f"Got NaN advantages: {advantages}"
        assert advantages[0].item() == pytest.approx(0.0)

    def test_multiple_single_sample_groups(self) -> None:
        """Multiple groups each with 1 sample → all advantages = 0."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 5.0, 10.0])
        group_ids = torch.tensor([0, 1, 2])  # each prompt has 1 sample
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        assert torch.allclose(advantages, torch.zeros(3))

    def test_group_with_multiple_samples_works(self) -> None:
        """Group with multiple samples → proper normalization, no NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        group_ids = torch.tensor([0, 0, 0, 0])  # all same prompt
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        # Mean=4, should be negative for 1,3 and positive for 5,7
        assert advantages[0] < 0
        assert advantages[3] > 0

class TestGRPOFlowMatchingKL:
    """Groups tests for grpoflow matching KL."""
    def test_flow_kl_ignores_dt_by_default_to_match_flow_grpo(self) -> None:
        """Checks flow KL ignores dt by default to match flow GRPO."""
        grpo = GRPO(GRPOConfig(init_kl_coef=1.0))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
        )

        loss, metrics = grpo.compute_loss(
            AlgorithmInput(
                signals=signals,
                advantages=torch.zeros(2),
            ),
        )

        assert loss.item() == pytest.approx(0.5)
        assert metrics.kl_penalty == pytest.approx(0.5)

    def test_flow_kl_can_use_dt_when_explicitly_configured(self) -> None:
        """Checks flow KL can use dt when explicitly configured."""
        grpo = GRPO(GRPOConfig(init_kl_coef=1.0, flow_kl_use_dt=True))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
        )

        loss, metrics = grpo.compute_loss(
            AlgorithmInput(
                signals=signals,
                advantages=torch.zeros(2),
            ),
        )

        assert loss.item() == pytest.approx(50.0)
        assert metrics.kl_penalty == pytest.approx(50.0)


class TestGRPOClippedSurrogate:
    """Numeric checks on the PPO-clipped surrogate (continuous.py:102-141).

    These guard the classic sign trap in ``torch.maximum(unclipped, clipped)``
    on the negative-advantage side, plus the clip_fraction / approx_kl metrics.
    """

    def test_ratio_one_gives_negative_mean_advantage(self) -> None:
        """log_prob == old_log_prob → ratio 1 → loss == -mean(advantage)."""
        grpo = GRPO(GRPOConfig(init_kl_coef=0.0))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
        )
        advantages = torch.tensor([2.0, -4.0])
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=advantages),
        )
        assert loss.item() == pytest.approx(-advantages.mean().item())
        assert metrics.policy_loss == pytest.approx(-advantages.mean().item())
        # No drift from old policy → no clipping, zero approx-KL.
        assert metrics.clip_fraction == pytest.approx(0.0)
        assert metrics.approx_kl == pytest.approx(0.0)

    def test_positive_advantage_uses_clipped_ratio_when_ratio_high(self) -> None:
        """Positive advantage + ratio above 1+eps_clip → loss uses clipped ratio.

        ``maximum(-adv*ratio, -adv*clipped)`` with adv>0 picks the larger
        (less negative) term, i.e. the clipped one. So the surrogate is
        ``-adv*(1+eps_clip)``, not ``-adv*ratio``.
        """
        eps_clip = 0.2
        grpo = GRPO(GRPOConfig(init_kl_coef=0.0, eps_clip=eps_clip))
        log_ratio = 1.0  # ratio = e ≈ 2.718, well above 1+eps_clip
        signals = _flow_signals(
            log_prob=torch.full((1,), log_ratio),
            old_log_prob=torch.zeros(1),
        )
        adv = torch.tensor([3.0])
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=adv),
        )
        assert loss.item() == pytest.approx(-adv.item() * (1.0 + eps_clip))
        assert metrics.clip_fraction == pytest.approx(1.0)

    def test_negative_advantage_uses_unclipped_ratio_when_ratio_high(self) -> None:
        """Negative advantage + ratio above 1+eps_clip → unclipped term wins.

        With adv<0, ``-adv*ratio`` is positive and larger than
        ``-adv*clipped``; ``maximum`` selects the unclipped surrogate. This is
        the side where a naive ``min``/``max`` mix-up silently flips training.
        """
        eps_clip = 0.2
        grpo = GRPO(GRPOConfig(init_kl_coef=0.0, eps_clip=eps_clip))
        log_ratio = 1.0
        signals = _flow_signals(
            log_prob=torch.full((1,), log_ratio),
            old_log_prob=torch.zeros(1),
        )
        adv = torch.tensor([-3.0])
        loss, _ = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=adv),
        )
        ratio = torch.exp(torch.tensor(log_ratio)).item()
        assert loss.item() == pytest.approx(-adv.item() * ratio)

    def test_clip_fraction_and_approx_kl_match_formula(self) -> None:
        """clip_fraction = mean(|ratio-1|>eps_clip); approx_kl = 0.5*mean(d^2)."""
        eps_clip = 0.2
        grpo = GRPO(GRPOConfig(init_kl_coef=0.0, eps_clip=eps_clip))
        # One sample drifts hard (clipped), one stays put (not clipped).
        log_prob = torch.tensor([1.0, 0.0])
        old_log_prob = torch.zeros(2)
        signals = _flow_signals(log_prob=log_prob, old_log_prob=old_log_prob)
        _, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.ones(2)),
        )
        ratio = torch.exp(log_prob - old_log_prob)
        expected_clip = torch.mean((torch.abs(ratio - 1.0) > eps_clip).float()).item()
        expected_kl = 0.5 * torch.mean((log_prob - old_log_prob) ** 2).item()
        assert metrics.clip_fraction == pytest.approx(expected_clip)
        assert metrics.clip_fraction == pytest.approx(0.5)
        assert metrics.approx_kl == pytest.approx(expected_kl)


def _flow_signals(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    prev_sample_mean: torch.Tensor | None = None,
    ref_prev_sample_mean: torch.Tensor | None = None,
    std_dev_t: torch.Tensor | None = None,
    dt: torch.Tensor | None = None,
) -> TrajectorySignalBatch:
    return TrajectorySignalBatch(
        segments={
            "denoise": SegmentSignal(
                name="denoise",
                segment="denoise",
                axis="timestep",
                axes=("sample",),
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=torch.ones_like(log_prob),
                ref_log_prob=ref_log_prob,
                prev_sample_mean=prev_sample_mean,
                ref_prev_sample_mean=ref_prev_sample_mean,
                std_dev_t=std_dev_t,
                dt=dt,
            ),
        },
        group_ids=torch.arange(log_prob.shape[0], device=log_prob.device),
        primary_segment="denoise",
    )
