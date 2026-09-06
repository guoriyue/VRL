"""Motion-dynamics reward: the anti-static-collapse floor.

The default lane drives ``_dynamic_degree`` with a banded flow field (top
quarter of rows moves by exactly 5 px, the rest is still) so each
``top_fraction`` picks a different number and the ``topk`` really selects.
The ``optional`` lane runs the genuine RAFT-small weights and asserts the
one property the banded stub cannot: a panning clip scores far above a static
one, so a wrong flow index, a lost ``[-1, 1]`` mapping, or swapped frame pairs
fail here and nowhere else.
"""

from __future__ import annotations

import math

import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.motion_dynamics import MotionDynamicsModel

_FLOW_SIZE = 8
_BAND_PIXELS = 5.0  # a (3, 4) flow vector on the top quarter of rows


class _BandFlow:
    """Stand-in for the RAFT module: top 25% of rows move (3, 4) px, the rest 0."""

    def __call__(self, first: torch.Tensor, second: torch.Tensor) -> list[torch.Tensor]:
        del second
        flow = torch.zeros(first.shape[0], 2, _FLOW_SIZE, _FLOW_SIZE)
        flow[:, 0, : _FLOW_SIZE // 4] = 3.0
        flow[:, 1, : _FLOW_SIZE // 4] = 4.0
        return [flow]


def _model(**config: object) -> MotionDynamicsModel:
    model = MotionDynamicsModel({"device": "cpu", "flow_size": _FLOW_SIZE, **config})
    model._module = _BandFlow()
    return model


def _clip(frames: int) -> RewardInferenceArtifact:
    """An in-memory ``[C, T, H, W]`` uint8 clip; ``__post_init__`` allows an empty path."""

    return RewardInferenceArtifact(
        artifact_id="motion",
        sample_id="motion-0",
        path="",
        media=torch.randint(0, 255, (3, frames, 16, 16), dtype=torch.uint8),
    )


def _diagonal_unit() -> float:
    """The banded magnitude in frame-diagonal units: 5 px over sqrt(2) * flow_size."""

    return _BAND_PIXELS / math.sqrt(2 * _FLOW_SIZE**2)


@pytest.mark.parametrize(
    ("top_fraction", "expected_fraction_of_unit"),
    [(0.25, 1.0), (0.5, 0.5), (1.0, 0.25)],
)
def test_dynamic_degree_is_the_top_fraction_mean_in_diagonal_units(
    top_fraction: float,
    expected_fraction_of_unit: float,
) -> None:
    """Exactly the top quarter moves, so the mean over top-k halves as k doubles."""

    raw = _model(top_fraction=top_fraction)._dynamic_degree(torch.rand(3, 16, 16, 3))

    assert raw == pytest.approx(_diagonal_unit() * expected_fraction_of_unit, rel=1e-6)


def test_magnitude_scale_saturates_the_reward_at_one() -> None:
    """The default scale (50) maps this motion past 1.0 and the reward clamps."""

    assert _model(top_fraction=0.25)(_clip(frames=4)) == {"motion_dynamics": 1.0}


def test_unit_magnitude_scale_is_linear_in_the_raw_degree() -> None:
    """With ``magnitude_scale=1`` the reward IS the raw degree: this pins the 50."""

    reward = _model(top_fraction=0.25, magnitude_scale=1.0)(_clip(frames=4))

    assert reward["motion_dynamics"] == pytest.approx(_diagonal_unit(), rel=1e-6)


def test_single_frame_clip_scores_zero_instead_of_crashing() -> None:
    """A one-frame clip has no flow pair; the guard must not kill the reward worker."""

    assert _model()(_clip(frames=1)) == {"motion_dynamics": 0.0}


def test_flow_size_must_be_divisible_by_eight() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        MotionDynamicsModel({"device": "cpu", "flow_size": 100})


@pytest.mark.optional
def test_real_raft_separates_a_static_clip_from_a_panning_clip() -> None:
    """Real RAFT-small: a still clip sits near the floor, a 6 px/frame pan saturates.

    Random-init RAFT would carry no motion signal; only the trained weights make
    this a discrimination test. Skips when the torchvision weights are not cached.
    """

    try:
        model = MotionDynamicsModel({"device": "cpu", "flow_size": 128})
        model._module_for_inference()
    except Exception as exc:  # pragma: no cover - offline machines without the weights
        pytest.skip(f"RAFT-small weights are unavailable: {exc}")

    torch.manual_seed(0)
    base = torch.rand(1, 96, 96, 3)

    def clip(shift: int) -> torch.Tensor:
        return torch.cat([torch.roll(base, shifts=shift * i, dims=2) for i in range(4)], dim=0)

    static = model._dynamic_degree(clip(shift=0))
    moving = model._dynamic_degree(clip(shift=6))

    assert static < 0.002
    assert moving > 20 * static
    assert min(1.0, static * model.magnitude_scale) < 0.1
    assert min(1.0, moving * model.magnitude_scale) > 0.9
