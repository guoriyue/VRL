"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationInput
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder


def test_engine_request_builder_reads_resolved_request_sampling() -> None:
    """Checks engine request builder reads resolved request sampling."""
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(
            values={
                "alpha": 1,
                "window": (0, 2),
                "n_samples_per_prompt": 3,
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
        seed=7,
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
    assert collector_request.request.return_artifacts == {"output", "trajectory"}
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
        config=RolloutCollectorConfig(values={"alpha": 1}),
    )

    collector_request = builder.build(
        ["prompt"],
        1,
        request_overrides={"alpha": 2, "beta": 3},
    )

    assert collector_request.request.sampling == {"alpha": 2, "beta": 3}
