"""Explicit replay-axis selection and static payload preservation."""

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory import TrajectoryResolver, build_diffusion_trajectory


@pytest.fixture
def resolver() -> TrajectoryResolver:
    request = GenerationRequest(
        request_id="replay-axis",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.arange(12).reshape(2, 3, 2),
        actions=torch.zeros(2, 3, 2),
        old_log_prob=torch.zeros(2, 3),
        timesteps=torch.zeros(2, 3),
        kl=torch.zeros(2, 3),
        replay_tensors={"prompt_embeds": torch.ones(2, 4, 8)},
        context={},
    )
    return TrajectoryResolver(trajectory)


@pytest.mark.parametrize(
    "selection, message",
    [
        ({"axis": "denoise"}, "provided together"),
        ({"axis_index": 1}, "provided together"),
        ({"axis": "denosie", "axis_index": 1}, "unknown replay axis"),
        ({"axis": "denoise", "axis_index": -1}, "must be >= 0"),
        ({"axis": "denoise", "axis_index": True}, "must be an integer"),
        ({"axis": "denoise", "axis_index": 1.5}, "must be an integer"),
        ({"axis": "denoise", "axis_index": "1"}, "must be an integer"),
        ({"axis": "denoise", "axis_index": 3}, "out of range"),
    ],
)
def test_replay_rejects_invalid_axis_selection(resolver, selection, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolver.replay_tensor_dict("denoise", **selection)


def test_replay_slices_step_tensors_and_preserves_static_payload(resolver) -> None:
    complete = resolver.replay_tensor_dict("denoise")
    selected = resolver.replay_tensor_dict("denoise", axis="denoise", axis_index=1)

    torch.testing.assert_close(selected["observations"], complete["observations"][:, 1])
    assert selected["prompt_embeds"] is complete["prompt_embeds"]
    assert complete["observations"].shape == (2, 3, 2)
