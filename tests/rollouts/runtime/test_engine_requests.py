"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

from vrl.rollouts.collector.config import RolloutConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder


def test_engine_request_builder_reads_resolved_request_sampling() -> None:
    """Checks engine request builder reads resolved request sampling."""
    builder = GenerationRequestBuilder(
        family="sd3_5",
        task="t2i",
        request_prefix="sd3_5",
        config=RolloutConfig(
            family="sd3_5",
            values={
                "alpha": 1,
                "window": (0, 2),
                "n_samples_per_prompt": 3,
            },
        ),
        return_artifacts=("output",),
        default_task_type="text_to_image",
    )

    collector_request = builder.build(
        ["prompt"],
        3,
        {
            "seed": 7,
            "policy_version": 11,
            "target_text": "HELLO",
            "sample_metadata": {"difficulty": "easy"},
        },
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
    assert collector_request.request.return_artifacts == {"output"}
    assert collector_request.request.metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
        "task_type": "text_to_image",
    }
    assert collector_request.metadata == collector_request.request.metadata


def test_engine_request_builder_applies_request_overrides_last() -> None:
    """Checks engine request builder applies request overrides last."""
    builder = GenerationRequestBuilder(
        family="sd3_5",
        task="t2i",
        request_prefix="sd3_5",
        config=RolloutConfig(family="sd3_5", values={"alpha": 1}),
        return_artifacts=("output",),
    )

    collector_request = builder.build(
        ["prompt"],
        1,
        {"request_overrides": {"alpha": 2, "beta": 3}},
    )

    assert collector_request.request.sampling == {"alpha": 2, "beta": 3}
