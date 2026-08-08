"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

import pytest

from vrl.generation import GenerationInput
from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trajectory import TrajectoryStoragePolicy


def test_engine_request_builder_reads_resolved_request_sampling() -> None:
    """Checks engine request builder reads resolved request sampling."""
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(
            request_sampling={
                "alpha": 1,
                "window": (0, 2),
            },
        ),
    )

    collector_request = builder.build(
        [
            GenerationInput(
                prompt="prompt",
                metadata={"difficulty": "easy", "target_text": "HELLO"},
            ),
        ],
        3,
        request_overrides={"seed": 7},
        policy_version=11,
        metadata={"difficulty": "easy", "target_text": "HELLO"},
    )

    assert collector_request.request.family == "sd3_5"
    assert collector_request.request.task == "t2i"
    assert collector_request.request.samples_per_prompt == 3
    assert collector_request.request.policy_version == 11
    assert collector_request.request.sampling == {
        "alpha": 1,
        "window": [0, 2],
        "seed": 7,
    }
    assert collector_request.request.metadata == {}
    assert collector_request.request.inputs[0].task_type == "text_to_image"
    assert collector_request.request.inputs[0].metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
    }
    assert collector_request.metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
        "task_type": "text_to_image",
    }


def test_engine_request_builder_applies_request_overrides_last() -> None:
    """Checks engine request builder applies request overrides last."""
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(request_sampling={"alpha": 1}),
    )

    collector_request = builder.build(
        ["prompt"],
        1,
        request_overrides={"alpha": 2, "beta": 3},
    )

    assert collector_request.request.sampling == {"alpha": 2, "beta": 3}


@pytest.mark.parametrize(
    ("storage", "expected"),
    [
        (TrajectoryStoragePolicy(), None),
        (
            TrajectoryStoragePolicy(device="cpu", dtype="float16"),
            {"device": "cpu", "dtype": "float16"},
        ),
    ],
)
def test_engine_request_builder_derives_trajectory_storage_for_wire(
    storage: TrajectoryStoragePolicy,
    expected: dict[str, str] | None,
) -> None:
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(trajectory_storage=storage),
    )

    sampling = builder.build(["prompt"], 1).request.sampling

    if expected is None:
        assert "trajectory_storage" not in sampling
    else:
        assert sampling["trajectory_storage"] == expected
