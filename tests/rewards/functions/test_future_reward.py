"""CPU unit tests for the Future Reward discrimination probe (SPRINT_future_reward).

Locks the gate logic that must stay correct without a network/GPU: the probe's
candidate battery + per-family PASS/FAIL verdicts (the gate that killed pixel-L1).
DINOv2 / RAFT discrimination is integration-verified by running the probe itself.
"""

from __future__ import annotations

import torch

from vrl.scripts.eval.future_reward_discrimination_probe import (
    _build_candidates,
    _verdict,
)


def _agg(**means: float) -> dict[str, dict[str, float]]:
    """Build a probe aggregate from per-candidate means (std=0, n=8)."""
    return {name: {"mean": value, "std": 0.0, "n": 8} for name, value in means.items()}


def test_probe_builds_full_candidate_battery() -> None:
    frames = torch.rand(8, 16, 16, 3)
    other = torch.rand(8, 16, 16, 3)
    candidates = _build_candidates(frames, other, seed=0)
    assert {
        "exact",
        "perceptual_blur",
        "temporal_mean",
        "static_frozen",
        "frame_shuffle",
        "reverse",
        "random",
        "wrong_clip",
    } <= set(candidates)
    # static_frozen / temporal_mean are constant over time by construction.
    assert torch.allclose(candidates["static_frozen"][0], candidates["static_frozen"][-1])
    assert torch.allclose(candidates["temporal_mean"][0], candidates["temporal_mean"][-1])
    assert torch.equal(candidates["exact"], frames)


def test_verdict_fails_pixel_l1_like_scores() -> None:
    # Reproduces the pixel-L1 failure shape: hacks within ~1% of exact.
    agg = _agg(
        exact=0.994,
        perceptual_blur=0.944,
        temporal_mean=0.985,
        static_frozen=0.983,
        frame_shuffle=0.982,
        reverse=0.979,
        wrong_clip=0.776,
        random=0.756,
    )
    verdict = _verdict("target_video_similarity", agg)
    assert verdict["passed"] is False
    assert verdict["gap_ratio"] < 0.25


def test_verdict_passes_dino_like_scores() -> None:
    agg = _agg(
        exact=0.826,
        perceptual_blur=0.21,
        temporal_mean=0.573,
        static_frozen=0.577,
        frame_shuffle=0.586,
        reverse=0.565,
        wrong_clip=0.209,
        random=-0.008,
    )
    verdict = _verdict("target_dino_similarity", agg)
    assert verdict["passed"] is True


def test_verdict_motion_guard_requires_static_floor() -> None:
    passing = _agg(
        exact=0.36,
        perceptual_blur=0.036,
        temporal_mean=0.026,
        static_frozen=0.027,
        frame_shuffle=0.53,
        reverse=0.37,
        wrong_clip=0.36,
        random=1.0,
    )
    assert _verdict("motion_dynamics", passing)["passed"] is True
    # If static does not collapse to the floor, the guard must FAIL.
    leaky = dict(passing)
    leaky["static_frozen"] = {"mean": 0.30, "std": 0.0, "n": 8}
    assert _verdict("motion_dynamics", leaky)["passed"] is False
