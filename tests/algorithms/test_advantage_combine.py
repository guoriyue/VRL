"""Multi-objective advantage combination: per-component normalization must stop
a high-variance reward from dominating the combined advantage."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.advantages import GroupAdvantageEstimator

_KW = {"eps": 1e-4, "adv_clip_max": 5.0, "global_std": False}


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom > 1e-9 else 0.0


def test_normalized_sum_de_dominates_high_variance_component() -> None:
    """One group; a low-variance 'quality' reward and a 10x-variance 'nsfw'
    penalty that ranks samples in the OPPOSITE order, equal weights.

    weighted_sum_raw: the combined advantage tracks nsfw (it dominates by scale).
    normalized_sum: each component is z-scored first, so nsfw no longer
    dominates and quality gets equal say — here they cancel to ~0.
    """
    group_ids = torch.zeros(4, dtype=torch.long)
    quality = torch.tensor([1.0, 2.0, 3.0, 4.0])  # increasing, std ~1.1
    nsfw = torch.tensor([40.0, 30.0, 20.0, 10.0])  # decreasing, std ~11
    comps = {"quality": quality, "nsfw": nsfw}
    weights = {"quality": 1.0, "nsfw": 1.0}

    weighted_total = sum(weights[name] * reward for name, reward in comps.items())
    raw = GroupAdvantageEstimator(
        strategy="weighted_sum_raw",
        component_weights=weights,
        **_KW,
    ).compute(
        weighted_total,
        group_ids,
        component_rewards=comps,
    )
    norm = GroupAdvantageEstimator(
        strategy="normalized_sum",
        component_weights=weights,
        **_KW,
    ).compute(
        weighted_total,
        group_ids,
        component_rewards=comps,
    )

    # Raw path is captured by the high-variance nsfw reward (strong +corr).
    assert _corr(raw, nsfw) > 0.99
    assert _corr(raw, quality) < -0.99
    # Normalized path gives the two equal-weight components equal pull, so the
    # opposite rankings cancel to ~0: nsfw no longer dominates.
    assert norm.abs().max().item() < 0.1
    assert norm.abs().max().item() < 0.05 * raw.abs().max().item()


def test_normalized_sum_lets_low_variance_component_win_when_weighted() -> None:
    """With normalization, upweighting quality flips the ranking toward quality —
    impossible under raw summation where nsfw's scale swamps any sane weight."""
    group_ids = torch.zeros(4, dtype=torch.long)
    quality = torch.tensor([1.0, 2.0, 3.0, 4.0])
    nsfw = torch.tensor([40.0, 30.0, 20.0, 10.0])
    comps = {"quality": quality, "nsfw": nsfw}
    weights = {"quality": 2.0, "nsfw": 1.0}

    norm = GroupAdvantageEstimator(
        strategy="normalized_sum",
        component_weights=weights,
        **_KW,
    ).compute(
        weights["quality"] * quality + weights["nsfw"] * nsfw,
        group_ids,
        component_rewards=comps,
    )
    # Quality (increasing) now wins: highest-quality sample gets the top advantage.
    assert int(norm.argmax().item()) == 3
    assert _corr(norm, quality) > 0.9

    # Component values are not clipped before weighting; only the combined
    # advantage is clamped. Pre-clipping this outlier would incorrectly yield 1.25.
    sparse = torch.tensor([0.0] * 31 + [1.0])
    final_only = GroupAdvantageEstimator(
        eps=1e-4,
        adv_clip_max=2.5,
        global_std=False,
        strategy="normalized_sum",
        component_weights={"quality": 0.5},
    ).compute(
        0.5 * sparse,
        torch.zeros(32, dtype=torch.long),
        component_rewards={"quality": sparse},
    )
    assert final_only.max().item() == 2.5


def test_unknown_strategy_raises() -> None:
    try:
        GroupAdvantageEstimator(strategy="nope", component_weights={"q": 1.0}, **_KW)
    except ValueError as exc:
        assert "unknown advantage_combine strategy" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown strategy")


def test_nft_component_fusion_preserves_chain_credit_and_component_units() -> None:
    """NFT gives every turn the same fused credit, independent of reward units."""
    from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig

    config = DiffusionNFTConfig(advantage_combine="normalized_sum")
    algorithm = DiffusionNFT(
        config,
        advantage_estimator=config.build_estimator(
            component_weights={"instruction": 1.0, "preservation": 0.3}
        ),
    )
    # Three candidate trajectories, each repeated across three editing turns.
    instruction = torch.tensor([0.0, 1.0, 2.0]).repeat(3)
    preservation = torch.tensor([0.01, 0.03, 0.02]).repeat(3)
    groups = torch.zeros(9, dtype=torch.long)
    actual = algorithm.compute_advantages_from_components(
        instruction + 0.3 * preservation,
        {"instruction": instruction, "preservation": preservation},
        groups,
    )
    # Each component has standardized magnitude sqrt(3/2): IF ranks [0,1,2],
    # CC ranks [0,2,1]. Their weighted sum is [-1.3,0.3,1]*sqrt(3/2).
    expected = torch.tensor([-1.5921683, 0.3674235, 1.2247449]).repeat(3)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    rescaled = algorithm.compute_advantages_from_components(
        instruction + 300 * preservation,
        {"instruction": instruction, "preservation": preservation * 1000},
        groups,
    )
    torch.testing.assert_close(rescaled, expected, atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match="keys must match configured weights"):
        algorithm.compute_advantages_from_components(
            instruction, {"instruction": instruction}, groups
        )
    with pytest.raises(ValueError, match="unknown advantage_combine"):
        DiffusionNFTConfig(advantage_combine="invalid")
