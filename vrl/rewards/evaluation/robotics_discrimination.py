"""Domain policy for UnifiedReward robotics discrimination.

This module owns the deterministic adversarial candidate battery and the
fail-closed acceptance policy.  Transport, artifact materialization, and CLI
reporting stay in ``vrl.scripts.eval``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vrl.utils.media import align_frame_counts


@dataclass(frozen=True, slots=True)
class RoboticsRewardGatePolicy:
    """Acceptance policy on the public UnifiedReward 1-5 axis scale."""

    exact_axis_min: float = 3.0
    low_information_axis_max: float = 2.5
    exact_low_information_compound_gap: float = 1.0
    exact_adversary_compound_gap: float = 0.5
    wrong_clip_alignment_gap: float = 0.5
    frame_shuffle_physics_gap: float = 0.5
    anchor_pass_rate_min: float = 0.75


# These tuples define the persisted evaluation protocol, rather than duplicating
# another typed structure: adding or removing a candidate changes the gate itself.
_AXES = ("alignment", "physics", "alignment+physics")
_LOW_INFORMATION = (
    "perceptual_blur",
    "temporal_mean",
    "static_frozen",
    "random",
    "color_blob",
)
_REQUIRED_CANDIDATES = (
    "exact",
    *_LOW_INFORMATION,
    "frame_shuffle",
    "reverse",
    "wrong_clip",
)


def build_discrimination_candidates(
    frames: torch.Tensor,
    other: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build deterministic reward-hacking candidates from one real clip."""

    from torchvision.transforms.functional import gaussian_blur

    generator = torch.Generator().manual_seed(seed)
    count = frames.shape[0]
    height, width = frames.shape[1:3]
    mean_clip = frames.mean(dim=0, keepdim=True).repeat(count, 1, 1, 1)
    blurred_mean = (
        gaussian_blur(mean_clip.permute(0, 3, 1, 2), kernel_size=[31, 31], sigma=[9.0, 9.0])
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    blob_keyframes = torch.rand((2, 3, 4, 4), generator=generator)
    blob_keyframes = torch.nn.functional.interpolate(
        blob_keyframes,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    blend = torch.linspace(0.0, 1.0, count).reshape(count, 1, 1, 1)
    color_blob = (
        blob_keyframes[0].permute(1, 2, 0).unsqueeze(0) * (1.0 - blend)
        + blob_keyframes[1].permute(1, 2, 0).unsqueeze(0) * blend
    )
    candidates = {
        "exact": frames,
        "perceptual_blur": blurred_mean,
        "temporal_mean": mean_clip,
        "static_frozen": frames[0:1].repeat(count, 1, 1, 1),
        "frame_shuffle": frames.index_select(0, torch.randperm(count, generator=generator)),
        "reverse": frames.flip(0),
        "random": torch.rand(frames.shape, generator=generator),
        "color_blob": color_blob,
    }
    if other.numel():
        aligned, _ = align_frame_counts(other, frames)
        candidates["wrong_clip"] = aligned
    return candidates


def aggregate_axis_scores(
    anchors: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate each candidate/axis while retaining fail-closed extrema."""

    candidates = sorted({candidate for anchor in anchors for candidate in anchor})
    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for candidate in candidates:
        aggregate[candidate] = {}
        for axis in _AXES:
            values = [anchor[candidate][axis] for anchor in anchors if candidate in anchor]
            if not values:
                continue
            aggregate[candidate][axis] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
    return aggregate


def robotics_reward_verdict(
    anchors: list[dict[str, dict[str, float]]],
    aggregate: dict[str, dict[str, dict[str, float | int]]],
    *,
    policy: RoboticsRewardGatePolicy | None = None,
) -> dict[str, Any]:
    """Return a structured PASS/FAIL verdict for the robotics reward contract."""

    policy = policy or RoboticsRewardGatePolicy()
    missing = [name for name in _REQUIRED_CANDIDATES if name not in aggregate]
    if missing:
        raise ValueError(f"robotics reward gate missing candidates: {missing}")
    if not anchors:
        raise ValueError("robotics reward gate needs at least one scored anchor")

    def mean(candidate: str, axis: str) -> float:
        return float(aggregate[candidate][axis]["mean"])

    exact_alignment = mean("exact", "alignment")
    exact_physics = mean("exact", "physics")
    exact_compound = mean("exact", "alignment+physics")
    low_information_max = max(mean(name, "alignment+physics") for name in _LOW_INFORMATION)
    low_information_gap = exact_compound - low_information_max
    adversary_max = max(
        mean(name, "alignment+physics") for name in _REQUIRED_CANDIDATES if name != "exact"
    )
    adversary_gap = exact_compound - adversary_max
    wrong_clip_gap = exact_alignment - mean("wrong_clip", "alignment")
    frame_shuffle_gap = exact_physics - mean("frame_shuffle", "physics")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "exact_alignment_floor",
        exact_alignment >= policy.exact_axis_min,
        f"mean={exact_alignment:.3f}, need >= {policy.exact_axis_min:.3f}",
    )
    add_check(
        "exact_physics_floor",
        exact_physics >= policy.exact_axis_min,
        f"mean={exact_physics:.3f}, need >= {policy.exact_axis_min:.3f}",
    )
    add_check(
        "exact_low_information_compound_gap",
        low_information_gap >= policy.exact_low_information_compound_gap,
        f"gap={low_information_gap:.3f}, need >= {policy.exact_low_information_compound_gap:.3f}",
    )
    add_check(
        "exact_adversary_compound_gap",
        adversary_gap >= policy.exact_adversary_compound_gap,
        f"gap={adversary_gap:.3f}, need >= {policy.exact_adversary_compound_gap:.3f}",
    )
    add_check(
        "wrong_clip_alignment_gap",
        wrong_clip_gap >= policy.wrong_clip_alignment_gap,
        f"gap={wrong_clip_gap:.3f}, need >= {policy.wrong_clip_alignment_gap:.3f}",
    )
    add_check(
        "frame_shuffle_physics_gap",
        frame_shuffle_gap >= policy.frame_shuffle_physics_gap,
        f"gap={frame_shuffle_gap:.3f}, need >= {policy.frame_shuffle_physics_gap:.3f}",
    )

    high_axes: list[str] = []
    for candidate in _LOW_INFORMATION:
        for axis in ("alignment", "physics"):
            observed = float(aggregate[candidate][axis]["max"])
            if observed > policy.low_information_axis_max:
                high_axes.append(f"{candidate}.{axis}={observed:.3f}")
    add_check(
        "low_information_axes_ceiling",
        not high_axes,
        (
            f"all per-anchor scores <= {policy.low_information_axis_max:.3f}"
            if not high_axes
            else "too high: " + ", ".join(high_axes)
        ),
    )

    anchor_passes: list[bool] = []
    for scores in anchors:
        exact = scores["exact"]
        low_max = max(scores[name]["alignment+physics"] for name in _LOW_INFORMATION)
        anchor_adversary_max = max(
            scores[name]["alignment+physics"] for name in _REQUIRED_CANDIDATES if name != "exact"
        )
        low_axes_ok = all(
            scores[name][axis] <= policy.low_information_axis_max
            for name in _LOW_INFORMATION
            for axis in ("alignment", "physics")
        )
        anchor_passes.append(
            exact["alignment"] >= policy.exact_axis_min
            and exact["physics"] >= policy.exact_axis_min
            and exact["alignment+physics"] - low_max >= policy.exact_low_information_compound_gap
            and exact["alignment+physics"] - anchor_adversary_max
            >= policy.exact_adversary_compound_gap
            and exact["alignment"] - scores["wrong_clip"]["alignment"]
            >= policy.wrong_clip_alignment_gap
            and exact["physics"] - scores["frame_shuffle"]["physics"]
            >= policy.frame_shuffle_physics_gap
            and low_axes_ok
        )
    anchor_pass_rate = sum(anchor_passes) / len(anchor_passes)
    add_check(
        "anchor_pass_rate",
        anchor_pass_rate >= policy.anchor_pass_rate_min,
        f"rate={anchor_pass_rate:.3f}, need >= {policy.anchor_pass_rate_min:.3f}",
    )

    return {
        "passed": all(check["passed"] for check in checks),
        "policy": asdict(policy),
        "metrics": {
            "exact_alignment": exact_alignment,
            "exact_physics": exact_physics,
            "exact_alignment+physics": exact_compound,
            "exact_low_information_compound_gap": low_information_gap,
            "exact_adversary_compound_gap": adversary_gap,
            "wrong_clip_alignment_gap": wrong_clip_gap,
            "frame_shuffle_physics_gap": frame_shuffle_gap,
            "anchor_pass_rate": anchor_pass_rate,
        },
        "checks": checks,
    }


def require_robotics_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Validate public UnifiedReward axes and derive the training compound."""

    selected: dict[str, float] = {}
    for axis in ("alignment", "physics"):
        if axis not in scores:
            raise KeyError(f"UnifiedReward result missing {axis!r}: {sorted(scores)}")
        value = float(scores[axis])
        if not math.isfinite(value) or not 1.0 <= value <= 5.0:
            raise ValueError(f"UnifiedReward {axis} must be finite and in [1, 5], got {value}")
        selected[axis] = value
    selected["alignment+physics"] = selected["alignment"] + selected["physics"]
    return selected


__all__ = [
    "RoboticsRewardGatePolicy",
    "aggregate_axis_scores",
    "build_discrimination_candidates",
    "require_robotics_scores",
    "robotics_reward_verdict",
]
