"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

import random

import pytest

from vrl.generation import GenerationInput
from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trajectory.storage import TrajectoryStoragePolicy


def test_engine_request_builder_reads_resolved_request_sampling() -> None:
    """The builder takes family and task from the registry entry, sampling from the resolved
    config (tuples serialized as lists) plus overrides, threads policy version and task type,
    and copies caller metadata into the collector request with the task type added.
    """
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
            ),
        ],
        3,
        request_overrides={"negative_prompt": "blurry"},
        policy_version=11,
        metadata={"difficulty": "easy", "target_text": "HELLO"},
    )

    assert collector_request.request.family == "sd3_5"
    assert collector_request.request.task == "t2i"
    assert collector_request.request.samples_per_prompt == 3
    assert collector_request.request.policy_version == 11
    sampling = dict(collector_request.request.sampling)
    assert type(sampling.pop("seed")) is int
    assert sampling == {
        "alpha": 1,
        "window": [0, 2],
        "negative_prompt": "blurry",
    }
    assert collector_request.request.inputs[0].task_type == "text_to_image"
    assert collector_request.metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
        "task_type": "text_to_image",
    }


def test_engine_request_builder_applies_request_overrides_last() -> None:
    """Request overrides win over the configured request sampling for the same key."""
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    collector_request = builder.build(
        ["prompt"],
        1,
        request_overrides={"num_steps": 2, "guidance_scale": 3.0},
    )

    sampling = dict(collector_request.request.sampling)
    assert type(sampling.pop("seed")) is int
    assert sampling == {"num_steps": 2, "guidance_scale": 3.0}


def test_engine_request_builder_rejects_a_request_override_outside_the_family_vocabulary() -> None:
    """A typo in a manifest's request_overrides fails when the request is built."""
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(),
    )

    with pytest.raises(ValueError, match=r"unknown sampling\.typo_key"):
        builder.build(["prompt"], 1, request_overrides={"typo_key": 1})


def test_engine_request_builder_carries_the_engine_fields_off_the_sampling_dict() -> None:
    storage = TrajectoryStoragePolicy(device="cpu", dtype="float16")
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("sd3_5"),
        config=RolloutCollectorConfig(
            samples_per_generation_batch=2,
            trajectory_storage=storage,
        ),
    )

    request = builder.build(["prompt"], 1).request

    assert request.samples_per_generation_batch == 2
    assert request.trajectory_storage == storage
    assert set(request.sampling) == {"seed"}
    assert type(request.sampling["seed"]) is int


def test_rollout_seed_comes_from_the_checkpointed_driver_rng() -> None:
    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    previous = capture_rng_state()
    try:
        random.seed(42)
        builder = GenerationRequestBuilder(
            entry=get_model_family_entry("cosmos-predict2.5"),
            config=RolloutCollectorConfig(request_sampling={}),
        )
        builder.build(["first"], 2)
        checkpoint = capture_rng_state()
        uninterrupted = [builder.build(["next"], 2).request.sampling.get("seed") for _ in range(3)]
        random.seed(999)
        fresh_builder = GenerationRequestBuilder(
            entry=get_model_family_entry("cosmos-predict2.5"),
            config=RolloutCollectorConfig(request_sampling={}),
        )
        restore_rng_state(checkpoint)
        resumed = [fresh_builder.build(["next"], 2).request.sampling.get("seed") for _ in range(3)]
        assert all(type(seed) is int and 0 <= seed < 2**63 for seed in resumed)
        assert resumed == uninterrupted and len(set(resumed)) == 3
        assert builder.config.request_sampling == {}
    finally:
        restore_rng_state(previous)
