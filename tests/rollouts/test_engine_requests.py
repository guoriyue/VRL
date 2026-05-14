"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

from vrl.rollouts.collector.requests import RolloutEngineRequestBuilder
from vrl.rollouts.settings import RolloutSettings


def test_engine_request_builder_reads_resolved_request_sampling() -> None:
    builder = RolloutEngineRequestBuilder(
        family="fake",
        task="t2i",
        request_prefix="fake",
        config=RolloutSettings(
            family="fake",
            values={
                "alpha": 1,
                "window": (0, 2),
                "n_samples_per_prompt": 3,
            },
        ),
        return_artifacts=("output",),
        default_task_type="text_to_image",
    )

    plan = builder.build(
        ["prompt"],
        3,
        {
            "seed": 7,
            "policy_version": 11,
            "target_text": "HELLO",
            "sample_metadata": {"difficulty": "easy"},
        },
    )

    assert plan.request.family == "fake"
    assert plan.request.task == "t2i"
    assert plan.request.samples_per_prompt == 3
    assert plan.request.policy_version == 11
    assert plan.request.sampling == {"alpha": 1, "window": [0, 2], "seed": 7}
    assert plan.request.return_artifacts == {"output"}
    assert plan.request.metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
        "task_type": "text_to_image",
    }
    assert plan.reward_metadata == plan.request.metadata
    assert plan.pack_metadata == plan.request.metadata


def test_engine_request_builder_applies_request_overrides_last() -> None:
    builder = RolloutEngineRequestBuilder(
        family="fake",
        task="t2i",
        request_prefix="fake",
        config=RolloutSettings(family="fake", values={"alpha": 1}),
        return_artifacts=("output",),
    )

    plan = builder.build(
        ["prompt"],
        1,
        {"request_overrides": {"alpha": 2, "beta": 3}},
    )

    assert plan.request.sampling == {"alpha": 2, "beta": 3}
