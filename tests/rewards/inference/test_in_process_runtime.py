"""Tests for the in-process reward runtime transport."""

from __future__ import annotations

import math
import os
import random
import weakref

import numpy as np
import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT, gpu_process_used_bytes


class _SumMediaModel:
    """Toy RewardModel: scores = sum of the in-memory media values."""

    def __call__(self, artifact):
        total = float(sum(artifact.as_media()))
        return {"overall": total, "extra": 1.0}


def _make_request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req-1",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a",
                sample_id="sample-a",
                path="",
                media=[1.0, 2.0],
            ),
            RewardInferenceArtifact(
                artifact_id="b",
                sample_id="sample-b",
                path="",
                media=[3.0],
            ),
        ),
    )


@pytest.mark.asyncio
async def test_in_process_runtime_scores_without_disk_or_ray() -> None:
    """Checks in-process scoring without disk or Ray."""
    runtime = InProcessRewardScorer(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request())

    assert runtime.scoring_is_nonblocking is False
    assert runtime.external_accelerator_isolation_verified is True
    assert [r.artifact_id for r in results] == ["a", "b"]  # original order preserved
    assert results[0].scores == {"overall": 3.0, "extra": 1.0}
    assert results[1].scores == {"overall": 3.0, "extra": 1.0}
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_in_process_runtime_empty_request_returns_empty() -> None:
    """Checks that an empty in-process request returns no results."""
    runtime = InProcessRewardScorer(model=_SumMediaModel())
    request = RewardInferenceRequest(
        request_id="req-empty",
        artifacts=(),
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
        def __call__(self, artifact):
            return {"overall": 2.0}

    return _Immovable()


class _LazyTorchModel(TorchRewardModel):
    """Records whether its deferred state materialized inside a CuMem scope."""

    def __init__(self, worker_config):
        super().__init__(worker_config)
        self.load_scopes: list[bool] = []

    def _load_module(self) -> torch.nn.Module:
        import vrl.models.parking as parking_mod

        allocator = parking_mod.cumem_allocator()
        self.load_scopes.append(bool(allocator and getattr(allocator, "building", False)))
        return torch.nn.Identity()

    def score_media(self, *, media, prompt):
        del prompt
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

    def score_media(self, *, media, prompt):  # pragma: no cover
        raise AssertionError("a failed preparation must never reach scoring")


def _failing_prepare_factory(worker_config):
    return _FailingPrepareModel(worker_config)


_FLAKY_PREPARE_CALLS = {"count": 0}


def _flaky_prepare_factory(worker_config):
    """Fail the first build (partial prepare), succeed on the retry."""

    _FLAKY_PREPARE_CALLS["count"] += 1
    if _FLAKY_PREPARE_CALLS["count"] == 1:
        return _FailingPrepareModel(worker_config)
    return _lazy_torch_factory(worker_config)


def _parking_request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                sample_id="sample-0",
                path="/tmp/a0.mp4",
                prompt="p",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_sleep_offload_uses_cumem_pool(monkeypatch) -> None:
    """Repeated scores reuse one construction pool and park after each request."""
    import vrl.models.parking as parking_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    await runtime.score_batch(_parking_request())
    await runtime.park_memory()
    await runtime.score_batch(_parking_request())
    await runtime.park_memory()

    assert len(allocator.pool_tags) == 1
    tag = allocator.pool_tags[0]
    assert allocator.sleeps == [(tag,), (tag,)]
    assert allocator.wakes == [[tag]]

    await runtime.shutdown()
    assert allocator.wakes[-1] == [tag]
    assert tag not in allocator.allocator_and_pools


@pytest.mark.asyncio
async def test_sleep_offload_materializes_lazy_model_inside_cumem_pool(monkeypatch) -> None:
    """A lazy wrapper cannot defer its real CUDA allocations until scoring."""
    import vrl.models.parking as parking_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_lazy_torch_factory",
        },
    )

    results = await runtime.score_batch(_make_request())

    assert [result.scores["overall"] for result in results] == [3.0, 3.0]
    assert runtime._model.load_scopes == [True]
    await runtime.park_memory()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_dedicated_runtime_keeps_lazy_model_outside_cumem_pool(monkeypatch) -> None:
    """Without a shared-GPU lease, model loading remains first-score lazy."""
    import vrl.models.parking as parking_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {"model_factory": f"{__name__}:_lazy_torch_factory"},
    )

    await runtime.score_batch(_make_request())

    assert runtime._model.load_scopes == [False]
    assert allocator.pool_tags == []
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_pooled_preparation_rolls_back_before_retry(monkeypatch) -> None:
    """A partial lazy load cannot become the runtime's committed model."""
    import vrl.models.parking as parking_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    _FLAKY_PREPARE_CALLS["count"] = 0
    runtime = InProcessRewardScorer(
        {
            "sleep_offload": True,
            # First build fails mid-prepare (the transient cause), the retry
            # succeeds; the launch contract is frozen at construction, so the
            # retry goes through the same factory path.
            "model_factory": f"{__name__}:_flaky_prepare_factory",
        },
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        await runtime.score_batch(_make_request())

    assert _PARTIAL_PREPARE_REF is not None
    assert _PARTIAL_PREPARE_REF() is None
    assert runtime._model is None
    assert runtime._parking.pool is None
    assert allocator.allocator_and_pools == {}

    results = await runtime.score_batch(_make_request())
    assert [result.scores["overall"] for result in results] == [3.0, 3.0]
    await runtime.park_memory()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_reward_sleep_failure_quarantines_all_later_allocator_operations(
    monkeypatch,
) -> None:
    """A partial CuMem sleep cannot safely be retried, restored or closed."""
    import vrl.models.parking as parking_mod

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
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    await runtime.score_batch(_parking_request())
    with pytest.raises(RuntimeError, match="sleep failed"):
        await runtime.park_memory()
    assert runtime._parking.pool is not None
    assert runtime._parking.pool.asleep is False

    model = runtime._model
    for operation in (runtime.park_memory, runtime.activate, runtime.shutdown):
        with pytest.raises(parking_mod.CumemBroken, match="terminate the owning process"):
            await operation()
    with pytest.raises(parking_mod.CumemBroken, match="terminate the owning process"):
        await runtime.score_batch(_parking_request())
    assert allocator.sleep_attempts == 1
    assert runtime._model is model
    assert runtime._parking.pool.asleep is False


@pytest.mark.asyncio
async def test_reward_partial_wake_never_reaches_model_inference_again(monkeypatch) -> None:
    import vrl.models.parking as parking_mod

    class BrokenWake(_FakeCumemAllocator):
        def wake_up(self, *, tags):
            super().wake_up(tags=tags)
            raise RuntimeError("partial wake")

    allocator = BrokenWake()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        }
    )
    await runtime.score_batch(_parking_request())
    await runtime.park_memory()
    with pytest.raises(parking_mod.CumemBroken, match="partial wake"):
        await runtime.score_batch(_parking_request())
    with pytest.raises(parking_mod.CumemBroken, match="terminate the owning process"):
        await runtime.activate()
    assert len(allocator.wakes) == 1


@pytest.mark.asyncio
async def test_dedicated_reward_runtime_stays_resident(monkeypatch) -> None:
    """A dedicated runtime never creates or sleeps a parking pool."""
    import vrl.models.parking as parking_mod
    import vrl.rewards.runtime as reward_runtime_module

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer(
        {"model_factory": f"{__name__}:_immovable_factory"},
    )

    await runtime.score_batch(_parking_request())

    assert runtime.requires_memory_parking is False
    assert runtime._parking.pool is None
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
    import vrl.models.parking as parking_mod

    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: None)
    runtime = InProcessRewardScorer(
        {
            "device": "cuda:0",
            "sleep_offload": True,
            "model_factory": f"{__name__}:_immovable_factory",
        },
    )

    with pytest.raises(RuntimeError, match="CuMemAllocator is required"):
        await runtime.score_batch(_parking_request())

    assert runtime._model is None
    assert runtime._parking.pool is None


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1", reason="explicit CUDA pool gate"
)
async def test_reward_pool_captures_noncurrent_cuda_device(monkeypatch) -> None:
    """A reward pinned to cuda:1 must pool, park, and restore on cuda:1 while
    the driver's current device stays cuda:0 throughout."""

    import vrl.rewards.runtime as runtime_module

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    original = torch.cuda.current_device()
    torch.cuda.set_device(0)

    class Model:
        def __init__(self, config):
            self.value = torch.full((1024 * 1024,), 3.0, device=config["device"])

        def prepare_for_inference(self):
            self.lazy = torch.full_like(self.value, 7.0)

    monkeypatch.setattr(runtime_module, "import_from_path", lambda _: Model)
    runtime = InProcessRewardScorer(
        {
            "device": "cuda:1",
            "sleep_offload": True,
            "model_factory": "test:factory",
            "memory_parking_residual_bytes_limit": CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT,
        }
    )
    try:
        await runtime.activate()
        assert torch.cuda.current_device() == 0
        pool = runtime._parking.pool
        assert pool is not None
        owned = [data for data in pool._allocator.pointer_to_data.values() if data.tag == pool.tag]
        assert owned, "target-device model allocations escaped the CuMem pool"
        await runtime.park_memory()
        assert torch.cuda.current_device() == 0
        await runtime.activate()
        assert torch.equal(runtime._model.value.cpu(), torch.full((1024 * 1024,), 3.0))
        assert torch.equal(runtime._model.lazy.cpu(), torch.full((1024 * 1024,), 7.0))
    finally:
        await runtime.shutdown()
        assert torch.cuda.current_device() == 0
        torch.cuda.set_device(original)


def test_reward_parking_session_scopes_pool_operations_to_the_configured_device(
    monkeypatch,
) -> None:
    """A reward pinned to cuda:1 must pool, sleep and wake on cuda:1 while the
    driver's current device stays where it was; CPU and unset devices never
    touch torch.cuda.device."""
    from contextlib import nullcontext

    import vrl.models.parking as parking_mod
    from vrl.models.parking import ParkingSession

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    monkeypatch.setattr(parking_mod, "gpu_process_used_bytes", lambda device=None: 0)
    seen: list[object] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda target: seen.append(target) or nullcontext())

    session = ParkingSession("reward runtime", required=True, device="cuda:1")
    session.build(lambda: object(), cumem=True)
    session.park()
    session.restore()
    assert seen == [torch.device("cuda:1")] * 3

    for device in ("cpu", None):
        resident = ParkingSession("reward runtime", required=True, device=device)
        resident.build(lambda: object(), cumem=True)
        resident.park()
    assert len(seen) == 3


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1", reason="explicit CUDA RNG gate"
)
@pytest.mark.parametrize("raises", [False, True])
def test_reward_build_scope_preserves_all_initialized_cuda_rngs(raises) -> None:
    from vrl.rewards.runtime import _preserve_driver_rng_during_model_build

    assert torch.cuda.is_available()
    for device in range(torch.cuda.device_count()):
        torch.rand(8, device=f"cuda:{device}")
    before = torch.cuda.get_rng_state_all()
    try:
        with _preserve_driver_rng_during_model_build():
            for device in range(torch.cuda.device_count()):
                torch.rand(16, device=f"cuda:{device}")
            if raises:
                raise RuntimeError("construction failed")
    except RuntimeError as error:
        assert raises and str(error) == "construction failed"
    after = torch.cuda.get_rng_state_all()
    assert len(before) == len(after) == torch.cuda.device_count()
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pooled", "failure"),
    [(False, None), (False, "factory"), (True, None), (True, "factory"), (True, "prepare")],
)
async def test_reward_construction_preserves_driver_rng(monkeypatch, pooled, failure) -> None:
    import vrl.models.parking as parking_mod
    import vrl.rewards.runtime as runtime_mod

    def consume():
        random.random()
        np.random.random()
        torch.rand(3)

    class Model:
        def prepare_for_inference(self):
            consume()
            if failure == "prepare":
                raise RuntimeError("prepare failed")

    def factory(config):
        consume()
        if failure == "factory":
            raise RuntimeError("factory failed")
        return Model()

    monkeypatch.setattr(runtime_mod, "import_from_path", lambda _: factory)
    if pooled:
        allocator = _FakeCumemAllocator()
        monkeypatch.setattr(parking_mod, "cumem_allocator", lambda: allocator)
    runtime = InProcessRewardScorer({"model_factory": "test:factory", "sleep_offload": pooled})
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    try:
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                await runtime.activate()
        else:
            await runtime.activate()
            await runtime.activate()
        assert random.getstate() == before[0]
        after_np = np.random.get_state()
        assert after_np[0] == before[1][0] and after_np[2:] == before[1][2:]
        assert np.array_equal(after_np[1], before[1][1])
        assert torch.equal(torch.get_rng_state(), before[2])
    finally:
        await runtime.shutdown()
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


def test_sleep_offload_rejects_injected_model() -> None:
    """An injected model was not built inside the runtime-owned CuMem pool."""
    with pytest.raises(ValueError, match="model_factory"):
        InProcessRewardScorer({"sleep_offload": True}, model=object())


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1",
    reason="set WM_RUN_REAL_MODEL_TESTS=1 for cached real-model gates",
)
@pytest.mark.asyncio
async def test_real_aesthetic_score_parks_stably_across_two_cycles() -> None:
    """A real SigLIP forward may retain runtime code, never its model pool."""
    runtime = InProcessRewardScorer(
        {
            "device": "cuda:0",
            "dtype": "float32",
            "model_name": "google/siglip-so400m-patch14-384",
            "model_factory": "vrl.rewards.models.aesthetic:AestheticRewardModel",
            "sleep_offload": True,
        },
    )
    request = RewardInferenceRequest(
        request_id="real-aesthetic-parking",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="image-0",
                sample_id="sample-0",
                path="",
                media=torch.zeros(3, 512, 512),
            ),
        ),
    )
    baseline = gpu_process_used_bytes("cuda:0")
    try:
        first_result = await runtime.score_batch(request)
        await runtime.park_memory()
        first_residual = gpu_process_used_bytes("cuda:0")
        second_result = await runtime.score_batch(request)
        await runtime.park_memory()
        second_residual = gpu_process_used_bytes("cuda:0")

        assert math.isfinite(first_result[0].scores["aesthetic"])
        assert second_result[0].scores["aesthetic"] == pytest.approx(
            first_result[0].scores["aesthetic"],
        )
        first_delta = first_residual - baseline
        second_delta = second_residual - baseline
        assert 0 <= first_delta <= CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        # A second score may reuse the same process-lifetime CUDA code; it must
        # not accumulate another model-sized or steadily growing residual.
        assert second_delta <= first_delta + 4 * 1024 * 1024
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_structured_results_preserve_evidence_but_cannot_replace_identity_or_runtime_timing():
    from vrl.rewards.inference import RewardInferenceResult

    class Model:
        wrong_id = True
        version = "unit-v1"

        def score_results(self, artifacts):
            return [
                RewardInferenceResult(
                    artifact_id="wrong" if self.wrong_id else artifact.artifact_id,
                    scores={"overall": 0.25},
                    reward_model_version=self.version,
                    timing_ms={"inference_ms": 1e12, "detector_ms": 2},
                    diagnostics={"why": "missing:cat"},
                )
                for artifact in artifacts
            ]

        def score_batch(self, artifacts):
            raise AssertionError("numeric fallback must not run a second inference")

    model = Model()
    runtime = InProcessRewardScorer(
        {"device": "cpu", "reward_model_version": "unit-v1"}, model=model
    )
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        await runtime.score_batch(_make_request())
    model.wrong_id = False
    model.version = "other-version"
    with pytest.raises(ValueError, match="revision differs"):
        await runtime.score_batch(_make_request())
    model.version = None
    results = await runtime.score_batch(_make_request())
    assert [row.artifact_id for row in results] == ["a", "b"]
    assert results[0].reward_model_version == "unit-v1"
    assert results[0].timing_ms["inference_ms"] < 1e12
    assert results[0].timing_ms["detector_ms"] == 2
    assert results[0].diagnostics == {"why": "missing:cat"}
    await runtime.shutdown()
