"""Tests for the in-process reward runtime transport."""

from __future__ import annotations

import math
import os
import weakref

import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardInferenceRuntime
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


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
async def test_in_process_runtime_scores_without_disk_or_ray() -> None:
    """Checks in-process scoring without disk or Ray."""
    runtime = InProcessRewardInferenceRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request())

    assert runtime.scoring_is_nonblocking is False
    assert runtime.external_accelerator_isolation_verified is True
    assert [r.artifact_id for r in results] == ["a", "b"]  # original order preserved
    assert results[0].selected_score == pytest.approx(3.0)
    assert results[1].selected_score == pytest.approx(3.0)
    assert all(r.worker_id == "local" for r in results)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_in_process_runtime_composite_score_key_sums_components() -> None:
    """Checks that in-process scoring sums composite score components."""
    runtime = InProcessRewardInferenceRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request(score_key="overall+extra"))

    assert results[0].selected_score == pytest.approx(4.0)  # 3.0 + 1.0
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_in_process_runtime_empty_request_returns_empty() -> None:
    """Checks that an empty in-process request returns no results."""
    runtime = InProcessRewardInferenceRuntime(model=_SumMediaModel())
    request = RewardInferenceRequest(
        request_id="req-empty",
        artifacts=(),
        reward_name="fake",
        score_key="overall",
    )
    assert await runtime.score_batch(request) == []


class _FakeCumemAllocator:
    """Record CuMem calls made by a parked in-process reward runtime."""

    def __init__(self) -> None:
        self.pool_tags: list[str] = []
        self.sleeps: list[tuple[str, ...]] = []
        self.wakes: list[list[str]] = []
        self.allocator_and_pools: dict[str, object] = {}
        self.building = False

    def use_memory_pool(self, *, tag: str):
        import contextlib

        self.pool_tags.append(tag)
        self.allocator_and_pools[tag] = object()

        @contextlib.contextmanager
        def scope():
            self.building = True
            try:
                yield
            finally:
                self.building = False

        return scope()

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


class _LazyTorchModel(TorchRewardModel):
    """Records whether its deferred state materialized inside a CuMem scope."""

    def __init__(self, worker_config):
        super().__init__(worker_config)
        self.load_scopes: list[bool] = []

    def _load_module(self) -> torch.nn.Module:
        import vrl.utils.cuda_memory as cuda_memory_mod

        allocator = cuda_memory_mod._cumem_allocator()
        self.load_scopes.append(bool(allocator and getattr(allocator, "building", False)))
        return torch.nn.Identity()

    def score_media(self, *, media, prompt, request):
        del prompt, request
        return {"overall": float(sum(media))}


def _lazy_torch_factory(worker_config):
    return _LazyTorchModel(worker_config)


class _PartialPrepareState:
    pass


_PARTIAL_PREPARE_REF: weakref.ReferenceType[_PartialPrepareState] | None = None


class _FailingPrepareModel(TorchRewardModel):
    def _load_module(self) -> torch.nn.Module:
        global _PARTIAL_PREPARE_REF
        self.partial_state = _PartialPrepareState()
        _PARTIAL_PREPARE_REF = weakref.ref(self.partial_state)
        raise RuntimeError("prepare failed")

    def score_media(self, *, media, prompt, request):  # pragma: no cover
        raise AssertionError("a failed preparation must never reach scoring")


def _failing_prepare_factory(worker_config):
    return _FailingPrepareModel(worker_config)


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
    runtime = InProcessRewardInferenceRuntime(
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
async def test_sleep_offload_materializes_lazy_model_inside_cumem_pool(monkeypatch) -> None:
    """A lazy wrapper cannot defer its real CUDA allocations until scoring."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = InProcessRewardInferenceRuntime(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_lazy_torch_factory",
        },
    )

    results = await runtime.score_batch(_make_request())

    assert [result.selected_score for result in results] == [3.0, 3.0]
    assert runtime._model.load_scopes == [True]
    await runtime.park_memory()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_dedicated_runtime_keeps_lazy_model_outside_cumem_pool(monkeypatch) -> None:
    """Without a shared-GPU lease, model loading remains first-score lazy."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = InProcessRewardInferenceRuntime(
        {"model_factory": f"{__name__}:_lazy_torch_factory"},
    )

    await runtime.score_batch(_make_request())

    assert runtime._model.load_scopes == [False]
    assert allocator.pool_tags == []
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_pooled_preparation_rolls_back_before_retry(monkeypatch) -> None:
    """A partial lazy load cannot become the runtime's committed model."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = InProcessRewardInferenceRuntime(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_failing_prepare_factory",
        },
    )
    runtime._release_cuda_memory_for_parking = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="prepare failed"):
        await runtime.score_batch(_make_request())

    assert _PARTIAL_PREPARE_REF is not None
    assert _PARTIAL_PREPARE_REF() is None
    assert runtime._model is None
    assert runtime._pool is None
    assert runtime._preload_gpu_used_bytes is None
    assert allocator.allocator_and_pools == {}

    runtime._worker_config["model_factory"] = f"{__name__}:_lazy_torch_factory"
    results = await runtime.score_batch(_make_request())
    assert [result.selected_score for result in results] == [3.0, 3.0]
    await runtime.park_memory()
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(("extra_residual_bytes", "should_pass"), ((0, True), (1, False)))
async def test_reward_parking_bounds_cuda_runtime_residual(
    monkeypatch,
    extra_residual_bytes: int,
    should_pass: bool,
) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = InProcessRewardInferenceRuntime(
        {
            "device": "cuda:0",
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
            "memory_parking_residual_bytes_limit": CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT,
        },
    )
    baseline = 1024
    residual = baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + extra_residual_bytes
    # load baseline, park proof, terminal shutdown proof
    readings = iter((baseline, residual, baseline))
    runtime._gpu_used_bytes = lambda: next(readings)  # type: ignore[method-assign]
    runtime._release_cuda_memory_for_parking = lambda: None  # type: ignore[method-assign]

    await runtime.score_batch(_parking_request())

    if should_pass:
        proof = await runtime.park_memory()
        assert proof.residual_gpu_used_bytes == residual
    else:
        with pytest.raises(RuntimeError, match="incomplete reward memory parking"):
            await runtime.park_memory()
    await runtime.shutdown()


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
    runtime = InProcessRewardInferenceRuntime(
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
    import vrl.rewards.runtime as reward_runtime_module
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)
    runtime = InProcessRewardInferenceRuntime(
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

    monkeypatch.setattr(
        reward_runtime_module,
        "release_cuda_memory_for_parking",
        lambda _device: release_device_cache(),
    )
    await runtime.shutdown()

    assert cleanup_calls == 1
    assert runtime._model is None


@pytest.mark.asyncio
async def test_sleep_offload_requires_cumem(monkeypatch) -> None:
    """Without vLLM a parking reward fails loud instead of faking a CPU park."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: None)
    runtime = InProcessRewardInferenceRuntime(
        {
            "device": "cuda:0",
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )
    runtime._gpu_used_bytes = lambda: 1024  # type: ignore[method-assign]
    runtime._release_cuda_memory_for_parking = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="CuMemAllocator is required"):
        await runtime.score_batch(_parking_request())

    assert runtime._model is None
    assert runtime._pool is None
    # The pool is claimed before the baseline sample, so a failed require leaves
    # no phantom baseline for shutdown's residual check to measure against.
    assert runtime._preload_gpu_used_bytes is None


def test_sleep_offload_rejects_injected_model() -> None:
    """An injected model cannot supply the required pre-load GPU baseline."""
    with pytest.raises(ValueError, match="model_factory"):
        InProcessRewardInferenceRuntime({"sleep_offload": True}, model=object())


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1",
    reason="set WM_RUN_REAL_MODEL_TESTS=1 for cached real-model gates",
)
@pytest.mark.asyncio
async def test_real_aesthetic_score_parks_stably_across_two_cycles() -> None:
    """A real CLIP forward may retain runtime code, never its 1.64 GiB model pool."""
    runtime = InProcessRewardInferenceRuntime(
        {
            "device": "cuda:0",
            "dtype": "float32",
            "model_name": "openai/clip-vit-large-patch14",
            "model_factory": "vrl.rewards.models.aesthetic:AestheticRewardModel",
            "sleep_offload": True,
            "memory_parking_residual_bytes_limit": CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT,
        },
    )
    request = RewardInferenceRequest(
        request_id="real-aesthetic-parking",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="image-0",
                path="",
                media_type="image",
                media=torch.zeros(3, 512, 512),
            ),
        ),
        reward_name="aesthetic",
        score_key="aesthetic",
    )
    try:
        first_result = await runtime.score_batch(request)
        first_proof = await runtime.park_memory()
        second_result = await runtime.score_batch(request)
        second_proof = await runtime.park_memory()

        assert math.isfinite(first_result[0].selected_score)
        assert second_result[0].selected_score == pytest.approx(
            first_result[0].selected_score,
        )
        first_delta = first_proof.residual_gpu_used_bytes - first_proof.baseline_gpu_used_bytes
        second_delta = second_proof.residual_gpu_used_bytes - second_proof.baseline_gpu_used_bytes
        assert 0 <= first_delta <= CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        # A second score may reuse the same process-lifetime CUDA code; it must
        # not accumulate another model-sized or steadily growing residual.
        assert second_delta <= first_delta + 4 * 1024 * 1024
    finally:
        await runtime.shutdown()
