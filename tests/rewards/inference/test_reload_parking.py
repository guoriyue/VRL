"""Reload parking releases owners, preserves RNG, and retries failed cleanup."""

import gc
import weakref

import pytest
import torch

import vrl.rewards.runtime as runtime
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.launch_contract import RewardRuntimeLaunchContract


@pytest.mark.asyncio
async def test_reload_rebuilds_equal_scores_without_retaining_model(monkeypatch):
    owners = []
    trims = []

    class Model:
        def __init__(self, config):
            self.weight = torch.rand(1).item()
            owners.append(weakref.ref(self))

        def prepare_for_inference(self):
            torch.rand(4)

        def __call__(self, artifact):
            return {"overall": self.weight + sum(artifact.media)}

    monkeypatch.setattr(runtime, "import_from_path", lambda _: Model)
    monkeypatch.setattr(runtime, "_host_memory_trim", lambda: lambda: trims.append(True))
    scorer = runtime.InProcessRewardScorer(
        {"model_factory": "test:model", "sleep_offload": True, "memory_parking_mode": "reload"}
    )
    request = RewardInferenceRequest(
        request_id="reload",
        artifacts=(RewardInferenceArtifact("a", "sample", "", media=[1.0, 2.0]),),
    )
    before = torch.get_rng_state().clone()
    try:
        first = await scorer.score_batch(request)
        assert torch.equal(torch.get_rng_state(), before)
        await scorer.park_memory()
        gc.collect()
        assert owners[0]() is None
        await scorer.park_memory()
        assert len(owners) == 1
        second = await scorer.score_batch(request)
        assert first[0].scores == second[0].scores
        assert torch.equal(torch.get_rng_state(), before)
        assert len(owners) == 2
        assert scorer._pool is None
    finally:
        await scorer.shutdown()
    assert all(owner() is None for owner in owners)
    assert len(trims) == 5


@pytest.mark.asyncio
async def test_failed_reload_preparation_releases_partial_model_and_can_retry(monkeypatch):
    owners = []
    trims = []

    class Model:
        def __init__(self, config):
            owners.append(weakref.ref(self))
            torch.rand(3)

        def prepare_for_inference(self):
            if len(owners) == 1:
                raise RuntimeError("prepare failed")

    monkeypatch.setattr(runtime, "import_from_path", lambda _: Model)
    monkeypatch.setattr(runtime, "_host_memory_trim", lambda: lambda: trims.append(True))
    scorer = runtime.InProcessRewardScorer(
        {"model_factory": "test:model", "sleep_offload": True, "memory_parking_mode": "reload"}
    )
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="prepare failed"):
        await scorer.activate()
    assert owners[0]() is None
    assert torch.equal(torch.get_rng_state(), before)
    await scorer.activate()
    await scorer.shutdown()
    assert all(owner() is None for owner in owners)
    assert len(trims) == 4


@pytest.mark.asyncio
async def test_failed_host_release_is_not_reported_as_success(monkeypatch):
    calls = []

    def trim():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("host release failed")

    monkeypatch.setattr(runtime, "import_from_path", lambda _: lambda config: object())
    monkeypatch.setattr(runtime, "_host_memory_trim", lambda: trim)
    scorer = runtime.InProcessRewardScorer(
        {"model_factory": "test:model", "sleep_offload": True, "memory_parking_mode": "reload"}
    )
    await scorer.activate()
    with pytest.raises(RuntimeError, match="host release failed"):
        await scorer.park_memory()
    await scorer.park_memory()
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_reload_preload_release_failure_prevents_build_and_can_retry(monkeypatch):
    events = []

    def trim():
        events.append("trim")
        if len(events) == 1:
            raise RuntimeError("preload release failed")

    def factory(config):
        events.append("build")
        torch.rand(3)
        return object()

    monkeypatch.setattr(runtime, "import_from_path", lambda _: factory)
    monkeypatch.setattr(runtime, "_host_memory_trim", lambda: trim)
    scorer = runtime.InProcessRewardScorer(
        {"model_factory": "test:model", "sleep_offload": True, "memory_parking_mode": "reload"}
    )
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="preload release failed"):
        await scorer.activate()
    assert events == ["trim"]
    assert scorer._model is None
    assert torch.equal(torch.get_rng_state(), before)
    await scorer.activate()
    assert events == ["trim", "trim", "build"]
    await scorer.activate()
    assert events == ["trim", "trim", "build"]
    assert torch.equal(torch.get_rng_state(), before)
    await scorer.shutdown()


def test_reload_requires_explicit_parking_and_supported_mode():
    with pytest.raises(ValueError, match="requires sleep_offload"):
        RewardRuntimeLaunchContract.from_component_config({"memory_parking_mode": "reload"})
    with pytest.raises(ValueError, match="must be 'cumem' or 'reload'"):
        RewardRuntimeLaunchContract.from_component_config({"memory_parking_mode": "unknown"})
