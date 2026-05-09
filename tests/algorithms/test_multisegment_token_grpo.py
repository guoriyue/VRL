"""Tests for multi-segment Token-GRPO loss aggregation."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.rollouts.evaluators.types import SignalBatch


def _segment_signal(new_lp: torch.Tensor, mask: torch.Tensor | None = None) -> SignalBatch:
    aux = {}
    if mask is not None:
        aux["token_mask"] = mask
    return SignalBatch(
        log_prob=new_lp,
        dist_family="categorical",
        aux=aux,
    )


def _multi_signal(
    segments: dict[str, SignalBatch],
    old_log_probs: dict[str, torch.Tensor],
) -> SignalBatch:
    first = next(iter(segments.values()))
    return SignalBatch(
        log_prob=first.log_prob,
        dist_family="multisegment_categorical",
        aux={
            "segments": segments,
            "old_log_probs": old_log_probs,
        },
    )


def test_weighted_mean_matches_token_grpo_per_segment() -> None:
    adv = torch.ones(2)
    old_initial = torch.zeros(2, 2)
    old_final = torch.zeros(2, 3)
    initial = _segment_signal(torch.full((2, 2), 0.1))
    final = _segment_signal(torch.full((2, 3), 0.3))
    signals = _multi_signal(
        {"initial_image": initial, "final_image": final},
        {"initial_image": old_initial, "final_image": old_final},
    )

    cfg = MultiSegmentTokenGRPOConfig(
        init_kl_coef=0.0,
        eps_clip=10.0,
        segment_weights={
            "initial_image": 1.0,
            "selfcheck_text": 0.0,
            "final_image": 3.0,
        },
    )
    algo = MultiSegmentTokenGRPO(cfg)
    loss, metrics = algo.compute_signal_loss(signals, adv, old_log_probs=torch.zeros(2, 1))

    base = TokenGRPO(TokenGRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    initial_loss, initial_metrics = base.compute_signal_loss(initial, adv, old_initial)
    final_loss, final_metrics = base.compute_signal_loss(final, adv, old_final)
    expected = (initial_loss + 3.0 * final_loss) / 4.0

    assert torch.allclose(loss, expected)
    assert metrics.policy_loss == pytest.approx(
        (initial_metrics.policy_loss + 3.0 * final_metrics.policy_loss) / 4.0,
    )


def test_default_selfcheck_weight_zero_does_not_affect_loss() -> None:
    adv = torch.ones(1)
    image_signal = _segment_signal(torch.zeros(1, 2))
    noisy_selfcheck = _segment_signal(torch.full((1, 2), 100.0))
    signals = _multi_signal(
        {
            "initial_image": image_signal,
            "selfcheck_text": noisy_selfcheck,
            "final_image": image_signal,
        },
        {
            "initial_image": torch.zeros(1, 2),
            "selfcheck_text": torch.full((1, 2), -100.0),
            "final_image": torch.zeros(1, 2),
        },
    )

    algo = MultiSegmentTokenGRPO(MultiSegmentTokenGRPOConfig(init_kl_coef=0.0))
    loss, _ = algo.compute_signal_loss(signals, adv, old_log_probs=torch.zeros(1, 1))

    assert loss.item() == pytest.approx(-1.0)
    assert "selfcheck_text" not in algo.last_segment_metrics


def test_nonzero_missing_segment_raises() -> None:
    signals = _multi_signal(
        {"initial_image": _segment_signal(torch.zeros(1, 2))},
        {"initial_image": torch.zeros(1, 2)},
    )
    algo = MultiSegmentTokenGRPO(
        MultiSegmentTokenGRPOConfig(
            segment_weights={"final_image": 1.0},
        ),
    )

    with pytest.raises(RuntimeError, match="final_image"):
        algo.compute_signal_loss(signals, torch.zeros(1), old_log_probs=torch.zeros(1, 1))
