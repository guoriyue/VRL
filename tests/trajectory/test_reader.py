"""Explicit replay-axis selection and static payload preservation."""

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory import TrajectoryReader, build_diffusion_trajectory


@pytest.fixture
def reader() -> TrajectoryReader:
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
    return TrajectoryReader(trajectory)


@pytest.mark.parametrize("segment_name", ["", "missing"])
def test_replay_rejects_unknown_explicit_segment(reader, segment_name) -> None:
    with pytest.raises(ValueError, match="unknown trajectory segment"):
        reader.replay_tensor_dict(segment_name)


def test_replay_defaults_to_primary_segment_only_when_omitted(reader) -> None:
    explicit = reader.replay_tensor_dict("denoise")
    for selected in (reader.replay_tensor_dict(), reader.replay_tensor_dict(None)):
        assert selected.keys() == explicit.keys()
        assert all(selected[name] is value for name, value in explicit.items())


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
def test_replay_rejects_invalid_axis_selection(reader, selection, message) -> None:
    with pytest.raises(ValueError, match=message):
        reader.replay_tensor_dict("denoise", **selection)


def test_replay_slices_step_tensors_and_preserves_static_payload(reader) -> None:
    complete = reader.replay_tensor_dict("denoise")
    selected = reader.replay_tensor_dict("denoise", axis="denoise", axis_index=1)

    torch.testing.assert_close(selected["observations"], complete["observations"][:, 1])
    assert selected["prompt_embeds"] is complete["prompt_embeds"]
    assert complete["observations"].shape == (2, 3, 2)


@pytest.mark.parametrize("container", [list, tuple])
def test_replay_slices_nested_sequences(reader, container) -> None:
    observations = reader.tensor("denoise", "observations")
    values = observations.value.tolist()
    observations.value = container(container(row) for row in values)
    # Revalidate the same representation accepted at the public boundary.
    reader = TrajectoryReader(reader.trajectory)

    selected = reader.replay_tensor_dict("denoise", axis="denoise", axis_index=1)

    assert selected["observations"] == [row[1] for row in values]


def test_replay_preserves_tensor_index_failure_without_sequence_retry(reader) -> None:
    failure = RuntimeError("backend indexing failed")

    class BrokenTensor:
        shape = (2, 3, 2)

        def __getitem__(self, key):
            raise failure

        def __iter__(self):
            pytest.fail("a failed tensor operation must not fall back to iteration")

    reader.tensor("denoise", "observations").value = BrokenTensor()

    with pytest.raises(ValueError, match="backend indexing failed") as caught:
        reader.replay_tensor_dict("denoise", axis="denoise", axis_index=1)
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
def test_replay_validates_declared_axes_of_sequence_payloads(reader, container, values, message):
    reader.tensor("denoise", "observations").value = container(values)
    with pytest.raises(ValueError, match=message):
        TrajectoryReader(reader.trajectory)


def test_replay_allows_ragged_dimensions_without_declared_axes(reader):
    # Only sample is declared for prompt embeddings; inner lengths need not match.
    payload = ([1, 2], [3, 4, 5])
    reader.tensor("denoise", "prompt_embeds").value = payload
    checked = TrajectoryReader(reader.trajectory)
    assert checked.replay_tensor_dict("denoise")["prompt_embeds"] is payload
