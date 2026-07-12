"""Tests for the in-process LocalRewardRuntime transport."""

from __future__ import annotations

import pytest

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.runtime import LocalRewardRuntime


class _SumMediaModel:
    """Toy RewardModel: scores = sum of the in-memory media values."""

    def __call__(self, *, artifact, request):
        total = float(sum(artifact.as_media()))
        return {"overall": total, "extra": 1.0}


def _make_request(score_key: str = "overall") -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req-1",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a",
                path="",
                media_type="image",
                media=[1.0, 2.0],
            ),
            RewardInferenceArtifact(
                artifact_id="b",
                path="",
                media_type="image",
                media=[3.0],
            ),
        ),
        reward_name="fake",
        score_key=score_key,
    )


@pytest.mark.asyncio
async def test_local_runtime_scores_in_process_without_disk_or_ray() -> None:
    """Checks local runtime scores in process without disk or Ray."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request())

    assert [r.artifact_id for r in results] == ["a", "b"]  # original order preserved
    assert results[0].selected_score == pytest.approx(3.0)
    assert results[1].selected_score == pytest.approx(3.0)
    assert all(r.worker_id == "local" for r in results)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_local_runtime_composite_score_key_sums_components() -> None:
    """Checks local runtime composite score key sums components."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request(score_key="overall+extra"))

    assert results[0].selected_score == pytest.approx(4.0)  # 3.0 + 1.0
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_local_runtime_empty_request_returns_empty() -> None:
    """Checks local runtime empty request returns empty."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    request = RewardInferenceRequest(
        request_id="req-empty",
        artifacts=(),
        reward_name="fake",
        score_key="overall",
    )
    assert await runtime.score_batch(request) == []


class _FakeCumemAllocator:
    """Record the CuMem calls made by a parked local reward runtime."""

    def __init__(self) -> None:
        self.pool_tags: list[str] = []
        self.sleeps: list[tuple[str, ...]] = []
        self.wakes: list[list[str]] = []
        self.allocator_and_pools: dict[str, object] = {}

    def use_memory_pool(self, *, tag: str):
        import contextlib

        self.pool_tags.append(tag)
        self.allocator_and_pools[tag] = object()
        return contextlib.nullcontext()

    def sleep(self, *, offload_tags) -> None:
        self.sleeps.append(tuple(offload_tags))

    def wake_up(self, *, tags) -> None:
        self.wakes.append(list(tags))


def _immovable_factory(worker_config):
    """Build a reward model without ``to``; CuMem parking does not need it."""

    class _Immovable:
        def __call__(self, *, artifact, request):
            return {"overall": 2.0}

    return _Immovable()


def _parking_request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path="/tmp/a0.mp4",
                media_type="video",
                prompt="p",
            ),
        ),
        reward_name="fake",
        score_key="overall",
    )


@pytest.mark.asyncio
async def test_sleep_offload_uses_cumem_pool(monkeypatch) -> None:
    """Repeated scores reuse one construction pool and park after each request."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = LocalRewardRuntime(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    await runtime.score_batch(_parking_request())
    first_proof = await runtime.park_memory()
    await runtime.score_batch(_parking_request())
    await runtime.park_memory()

    assert len(allocator.pool_tags) == 1
    assert first_proof.request_id == "req"
    assert first_proof.released is True
    tag = allocator.pool_tags[0]
    assert allocator.sleeps == [(tag,), (tag,)]
    assert allocator.wakes == [[tag]]

    await runtime.shutdown()
    assert allocator.wakes[-1] == [tag]
    assert tag not in allocator.allocator_and_pools


@pytest.mark.asyncio
async def test_reward_parking_rejects_default_allocator_residual(monkeypatch) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = LocalRewardRuntime(
        {
            "device": "cuda:0",
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )
    readings = iter((100, 101))
    runtime._gpu_used_bytes = lambda: next(readings)  # type: ignore[method-assign]
    runtime._release_cuda_memory_for_parking = lambda: None  # type: ignore[method-assign]

    await runtime.score_batch(_parking_request())

    with pytest.raises(RuntimeError, match="incomplete reward memory parking"):
        await runtime.park_memory()


@pytest.mark.asyncio
async def test_reward_memory_parking_retries_after_sleep_failure(monkeypatch) -> None:
    """A failed allocator sleep does not publish proof or poison a retry."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    class _FlakyAllocator(_FakeCumemAllocator):
        def __init__(self) -> None:
            super().__init__()
            self.sleep_attempts = 0

        def sleep(self, *, offload_tags) -> None:
            self.sleep_attempts += 1
            if self.sleep_attempts == 1:
                raise RuntimeError("sleep failed")
            super().sleep(offload_tags=offload_tags)

    allocator = _FlakyAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = LocalRewardRuntime(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    await runtime.score_batch(_parking_request())
    with pytest.raises(RuntimeError, match="sleep failed"):
        await runtime.park_memory()
    assert runtime._pool is not None
    assert runtime._pool.asleep is False

    proof = await runtime.park_memory()

    assert allocator.sleep_attempts == 2
    assert proof.released is True
    assert runtime._pool.asleep is True


@pytest.mark.asyncio
async def test_dedicated_reward_runtime_stays_resident(monkeypatch) -> None:
    """A dedicated runtime never creates or sleeps a parking pool."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = LocalRewardRuntime(
        {"model_factory": f"{__name__}:_immovable_factory"},
    )

    await runtime.score_batch(_parking_request())

    assert runtime.requires_memory_parking is False
    assert runtime._pool is None
    assert allocator.pool_tags == []
    assert allocator.sleeps == []

    cleanup_calls = 0

    def release_device_cache() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    runtime._release_cuda_memory_for_parking = release_device_cache  # type: ignore[method-assign]
    await runtime.shutdown()

    assert cleanup_calls == 1
    assert runtime._model is None


@pytest.mark.asyncio
async def test_sleep_offload_requires_cumem(monkeypatch) -> None:
    """The shared-GPU path fails closed when CuMem is unavailable."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: None)
    runtime = LocalRewardRuntime(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    with pytest.raises(RuntimeError, match="CuMemAllocator"):
        await runtime.score_batch(_parking_request())


def test_sleep_offload_rejects_injected_model() -> None:
    """An already-built model cannot be retroactively placed in the CuMem pool."""
    with pytest.raises(ValueError, match="model_factory"):
        LocalRewardRuntime({"sleep_offload": True}, model=object())
