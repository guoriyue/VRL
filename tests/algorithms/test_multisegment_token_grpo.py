"""Tests for multi-segment Token-GRPO loss aggregation."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch


def _segment_signal(
    name: str,
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> SegmentSignal:
    if mask is None:
        mask = torch.ones_like(new_lp)
    return SegmentSignal(
        name=name,
        distribution="categorical",
        log_prob=new_lp,
        old_log_prob=old_lp,
        mask=mask,
    )


def _inputs(
    segments: dict[str, SegmentSignal],
    advantages: torch.Tensor | dict[str, torch.Tensor],
) -> AlgorithmInput:
    first = next(iter(segments.values()))
    return AlgorithmInput(
        signals=TrajectorySignalBatch(
            segments=segments,
            group_ids=torch.arange(first.log_prob.shape[0], device=first.log_prob.device),
            primary_segment=next(iter(segments)),
        ),
        advantages=advantages,
    )


def test_weighted_mean_matches_token_grpo_per_segment() -> None:
    """The multi-segment loss is the weighted mean of per-segment TokenGRPO losses and metrics
    under ``segment_weights``, with zero-weight segments excluded from the denominator.
    """
    adv = torch.ones(2)
    old_initial = torch.full((2, 2), -2.0)
    old_final = torch.full((2, 3), -2.0)
    initial = _segment_signal("initial_image", old_initial + 0.1, old_initial)
    final = _segment_signal("final_image", old_final + 0.3, old_final)
    inputs = _inputs({"initial_image": initial, "final_image": final}, adv)

    cfg = MultiSegmentTokenGRPOConfig(
        kl_coef=0.0,
        clip_ratio=10.0,
        segment_weights={
            "initial_image": 1.0,
            "selfcheck_text": 0.0,
            "final_image": 3.0,
        },
    )
    algo = MultiSegmentTokenGRPO(cfg)
    loss, metrics = algo.compute_loss(inputs)

    base = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=10.0))
    initial_loss, initial_metrics = base.compute_loss(
        _inputs({"initial_image": initial}, adv),
    )
    final_loss, final_metrics = base.compute_loss(
        _inputs({"final_image": final}, adv),
    )
    expected = (initial_loss + 3.0 * final_loss) / 4.0

    assert torch.allclose(loss, expected)
    assert metrics.policy_loss == pytest.approx(
        (initial_metrics.policy_loss + 3.0 * final_metrics.policy_loss) / 4.0,
    )
    assert metrics.update.approx_kl == pytest.approx(
        (initial_metrics.update.approx_kl + 3.0 * final_metrics.update.approx_kl) / 4.0,
    )


def test_default_selfcheck_weight_zero_does_not_affect_loss() -> None:
    """An enabled selfcheck segment with zero weight cannot change the image loss."""
    adv = torch.ones(1)
    image_old = torch.full((1, 2), -2.0)
    image_signal = _segment_signal("initial_image", image_old.clone(), image_old)
    noisy_selfcheck = _segment_signal(
        "selfcheck_text",
        torch.full((1, 2), -1.0),
        torch.full((1, 2), -2.0),
    )
    final_signal = _segment_signal("final_image", image_old.clone(), image_old)
    inputs = _inputs(
        {
            "initial_image": image_signal,
            "selfcheck_text": noisy_selfcheck,
            "final_image": final_signal,
        },
        adv,
    )

    config = MultiSegmentTokenGRPOConfig(
        kl_coef=0.0,
        train_segments={"initial_image": True, "selfcheck_text": True, "final_image": True},
    )
    algo = MultiSegmentTokenGRPO(config)
    loss, _ = algo.compute_loss(inputs)

    assert loss.item() == pytest.approx(-1.0)

    config.segment_weights["selfcheck_text"] = 1.0
    weighted_loss, _ = algo.compute_loss(inputs)
    assert weighted_loss.item() < loss.item()


def test_nonzero_missing_segment_raises() -> None:
    inputs = _inputs(
        {
            "initial_image": _segment_signal(
                "initial_image",
                torch.zeros(1, 2),
                torch.zeros(1, 2),
            ),
        },
        torch.zeros(1),
    )
    algo = MultiSegmentTokenGRPO(
        MultiSegmentTokenGRPOConfig(
            segment_weights={"final_image": 1.0},
        ),
    )

    with pytest.raises(RuntimeError, match="final_image"):
        algo.compute_loss(inputs)
