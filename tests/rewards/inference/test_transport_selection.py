"""Rewards score in-process; the removed pool transport fails loud."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.pickscore import PickScoreReward
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.runtime import LocalRewardRuntime, make_reward_runtime

_FACTORY = "vrl.rewards.models.pickscore:pickscore_reward_model"


def _request() -> RewardInferenceRequest:
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


def test_make_reward_runtime_inline() -> None:
    """Checks make reward runtime inline -> in-process transport."""
    runtime = make_reward_runtime(
        "inline",
        model_factory=_FACTORY,
        worker_config={"device": "cpu"},
    )
    assert isinstance(runtime, LocalRewardRuntime)


def test_make_reward_runtime_rejects_removed_pool() -> None:
    """Checks the removed pool transport fails loud with the migration hint."""
    with pytest.raises(ValueError, match="resource topology"):
        make_reward_runtime("pool", model_factory=_FACTORY)


def test_make_reward_runtime_rejects_unknown() -> None:
    """Checks make reward runtime rejects unknown."""
    with pytest.raises(ValueError, match="execution"):
        make_reward_runtime("bogus", model_factory=_FACTORY)


def test_reward_class_uses_inline_transport() -> None:
    """Checks reward class builds the in-process transport."""
    reward = PickScoreReward(device="cpu")
    assert isinstance(reward.runtime, LocalRewardRuntime)


class _FakeCumemAllocator:
    """Records pool/sleep/wake calls (the cumem contract, no CUDA needed).

    Kept as a fake on purpose: the real allocator needs vLLM installed plus a
    CUDA context; a memory-effect twin belongs in a vLLM-equipped GPU lane
    when one exists (the allocator-missing branch is already tested for real).
    """

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
    """Module-level factory: a reward model WITHOUT .to() (cumem needs none)."""

    class _Immovable:
        def __call__(self, *, artifact, request):
            return {"overall": 2.0}

    return _Immovable()


@pytest.mark.asyncio
async def test_sleep_offload_uses_cumem_pool(monkeypatch) -> None:
    """Model construction enters one pool; repeated scores only wake/sleep it."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)

    factory = f"{__name__}:_immovable_factory"
    a = LocalRewardRuntime({"sleep_offload": True, "model_factory": factory})

    await a.score_batch(_request())
    first_proof = await a.park_memory()
    await a.score_batch(_request())
    await a.park_memory()

    # vLLM creates a new torch MemPool on every context entry, so the model tag
    # is deliberately one-shot. Repeated execution is covered by the physical
    # residual proof instead of re-entering the allocator context.
    assert len(allocator.pool_tags) == 1
    assert first_proof.request_id == "req"
    assert first_proof.released is True
    a_tag = allocator.pool_tags[0]
    # score->sleep, wake->score->sleep.
    assert allocator.sleeps == [
        (a_tag,),
        (a_tag,),
    ]
    assert allocator.wakes == [[a_tag]]

    await a.shutdown()  # slept pool wakes before drop (frees pinned host copies)
    assert allocator.wakes[-1] == [a_tag]
    assert a_tag not in allocator.allocator_and_pools


def test_cumem_model_building_scope_is_one_shot(monkeypatch) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    pool = cuda_memory_mod.CumemPool.require("reward")

    with pool.building():
        pass
    with pytest.raises(RuntimeError, match="model-building scope is one-shot"):
        pool.building()

    assert allocator.pool_tags == ["reward"]


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

    await runtime.score_batch(_request())

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

    await runtime.score_batch(_request())
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

    await runtime.score_batch(_request())

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


def test_sleep_offload_requires_cumem(monkeypatch) -> None:
    """sleep_offload has no naive fallback: missing cumem fails loud at build."""
    import asyncio

    import vrl.utils.cuda_memory as cuda_memory_mod

    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: None)
    runtime = LocalRewardRuntime(
        {"sleep_offload": True, "model_factory": f"{__name__}:_immovable_factory"},
    )
    with pytest.raises(RuntimeError, match="CuMemAllocator"):
        asyncio.run(runtime.score_batch(_request()))


def test_sleep_offload_rejects_injected_model() -> None:
    """An already-built model cannot be pooled; sleep_offload rejects it."""
    with pytest.raises(ValueError, match="model_factory"):
        LocalRewardRuntime({"sleep_offload": True}, model=object())
