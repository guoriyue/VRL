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


@pytest.mark.parametrize("container", [list, tuple])
def test_replay_slices_nested_sequences(resolver, container) -> None:
    observations = resolver.tensor("denoise", "observations")
    values = observations.value.tolist()
    observations.value = container(container(row) for row in values)
    # Revalidate the same representation accepted at the public boundary.
    resolver = TrajectoryResolver(resolver.trajectory)

    selected = resolver.replay_tensor_dict("denoise", axis="denoise", axis_index=1)

    assert selected["observations"] == [row[1] for row in values]


def test_replay_preserves_tensor_index_failure_without_sequence_retry(resolver) -> None:
    failure = RuntimeError("backend indexing failed")

    class BrokenTensor:
        shape = (2, 3, 2)

        def __getitem__(self, key):
            raise failure

        def __iter__(self):
            pytest.fail("a failed tensor operation must not fall back to iteration")

    resolver.tensor("denoise", "observations").value = BrokenTensor()

    with pytest.raises(ValueError, match="backend indexing failed") as caught:
        resolver.replay_tensor_dict("denoise", axis="denoise", axis_index=1)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize(
    "values, message",
    [
        ([[1, 2, 3]], "axis 'sample' has shape 1, expected 2"),
        ([[1, 2, 3], [4, 5]], "axis 'denoise' has shape 2, expected 3"),
        ([1, 2], "scalar before declared axis 'denoise'"),
        ([torch.ones(3, 2), torch.ones(2, 2)], "contains runtime-only state: Tensor"),
    ],
)
def test_replay_validates_declared_axes_of_sequence_payloads(resolver, container, values, message):
    resolver.tensor("denoise", "observations").value = container(values)
    with pytest.raises(ValueError, match=message):
        TrajectoryResolver(resolver.trajectory)


def test_replay_allows_ragged_dimensions_without_declared_axes(resolver):
    # Only sample is declared for prompt embeddings; inner lengths need not match.
    payload = ([1, 2], [3, 4, 5])
    resolver.tensor("denoise", "prompt_embeds").value = payload
    checked = TrajectoryResolver(resolver.trajectory)
    assert checked.replay_tensor_dict("denoise")["prompt_embeds"] is payload
