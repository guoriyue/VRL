"""GenerationWorkerCore sleep/wake (SPRINT_frozen_component_preservation, defect A).

Level-1 offload-and-restore for on-demand activation: ``sleep`` parks
the loaded model on host RAM (transformer via ``nn.Module.to`` plus the frozen
VAE / text-encoders via ``move_frozen_components``) while keeping the executor
alive, and ``wake`` restores it onto the captured GPU without a cold reload. This
is the level-2 ``release_policy`` (discard + disk reload) counterpart that lets a
colocated trainer reclaim the GPU between collect and train.

The executor is injected directly so load_policy() short-circuits — no model build.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


@pytest.fixture(autouse=True)
def _isolate_cuda_parking_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU fakes independent of CUDA activity on the pytest host."""

    import vrl.generation.execution.worker as worker_module

    monkeypatch.setattr(worker_module, "gpu_used_bytes", lambda: 0)
    monkeypatch.setattr(worker_module, "release_cuda_memory_for_parking", lambda: None)


class _SleepModel:
    """RuntimeModel that records the device passed to ``to`` / frozen offload."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.to_calls: list[Any] = []
        self.frozen_calls: list[Any] = []

    def to(self, device: Any) -> _SleepModel:
        self.to_calls.append(device)
        return self

    def move_frozen_components(self, device: Any) -> None:
        self.frozen_calls.append(device)


class _Executor:
    family = "sd3_5"
    task = "t2i"

    def __init__(self, model: Any) -> None:
        self.model = model

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def gather_chunks(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _PipelinedExecutor(_Executor):
    def forward_plan_pipelined(self, request: Any, sample_rows: Any, engine_plan: Any) -> Any:
        return request, sample_rows, engine_plan


class _FakeCuMem:
    """Stand-in for vLLM's CuMemAllocator: records pool/sleep/wake calls.

    Kept as a fake on purpose: the real allocator needs vLLM installed plus a
    CUDA context (virtual-memory paging), neither available in the CPU lane.
    The allocator-missing branch is tested for real via
    test_sleep_offload_requires_cumem; a memory-effect twin belongs in a
    vLLM-equipped GPU lane when one exists.
    """

    def __init__(self) -> None:
        self.pool_tags: list[Any] = []
        self.sleep_calls: list[Any] = []
        self.wake_calls: list[Any] = []
        self.allocator_and_pools: dict[str, Any] = {}

    @contextlib.contextmanager
    def use_memory_pool(self, tag: Any = None):
        self.pool_tags.append(tag)
        self.allocator_and_pools[tag] = object()
        yield

    def sleep(self, offload_tags: Any = None) -> None:
        self.sleep_calls.append(offload_tags)

    def wake_up(self, tags: Any = None) -> None:
        self.wake_calls.append(tags)


def _core(
    model: Any | None,
    *,
    sleep_offload: bool = False,
    family: str = "sd3_5",
) -> GenerationWorkerCore:
    contract = GenerationRuntimeLaunchContract(
        family=family,
        model_build={},
        expected_model_identity={"schema": "test"},
        policy_version=1,
        sleep_offload=sleep_offload,
    )
    core = GenerationWorkerCore("rollout-0", contract)
    core.executor = _Executor(model) if model is not None else None
    core._preload_gpu_used_bytes = 0
    return core


def test_sleep_parks_model_and_frozen_on_cpu_keeping_executor() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)

    snapshot = core.sleep()

    # Transformer (nn.Module.to) AND the unregistered frozen components both leave
    # the GPU — the whole offload-and-restore point.
    assert model.to_calls == ["cpu"]
    assert model.frozen_calls == ["cpu"]
    # Executor is retained (level-1), unlike release_policy which nulls it.
    assert core.executor is not None
    assert snapshot.worker_id == "rollout-0"
    assert snapshot.backend == "cpu_offload"
    assert snapshot.residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
    assert snapshot.residual_gpu_used_bytes == snapshot.baseline_gpu_used_bytes


def test_wake_restores_model_to_captured_device() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)

    core.sleep()
    core.wake()

    # Restored onto the device the model lived on before sleeping, no disk reload.
    assert model.to_calls == ["cpu", "cuda:0"]
    assert model.frozen_calls == ["cpu", "cuda:0"]
    assert core.executor is not None


def test_cpu_sleep_and_wake_are_idempotent() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)

    core.sleep()
    core.sleep()
    core.wake()
    core.wake()

    assert model.to_calls == ["cpu", "cuda:0"]
    assert model.frozen_calls == ["cpu", "cuda:0"]


def test_sleep_without_loaded_model_is_a_safe_no_op() -> None:
    core = _core(None)
    snapshot = core.sleep()  # must not raise (nothing loaded)
    assert core.executor is None
    assert snapshot.backend == "cpu_only"
    assert snapshot.residual_bytes_limit == 0


def test_wake_after_eviction_rebuilds_via_load_policy() -> None:
    """If the executor was hard-released between sleep windows, wake reloads it."""
    core = _core(None)
    rebuilt: list[bool] = []
    core.load_policy = lambda: rebuilt.append(True)  # type: ignore[method-assign]

    core.wake()

    assert rebuilt == [True]


# -- cumem-backed offload -----------------------------------------------------


def test_cumem_sleep_wake_uses_allocator_not_module_moves() -> None:
    from vrl.utils.cuda_memory import CumemPool

    model = _SleepModel(device="cuda:0")
    core = _core(model)
    fake = _FakeCuMem()
    core._cumem = CumemPool(fake, "weights")  # as if load_policy pooled the model

    snapshot = core.sleep()
    core.wake()

    # The whole pooled model is released/restored through the allocator; the naive
    # per-module .to() round trip is bypassed entirely.
    assert fake.sleep_calls == [("weights",)]
    assert fake.wake_calls == [["weights"]]
    assert model.to_calls == []
    assert model.frozen_calls == []
    assert snapshot.residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


@pytest.mark.parametrize(
    ("extra_residual_bytes", "should_pass"),
    ((0, True), (1, False)),
)
def test_cumem_sleep_bounds_lazy_cuda_runtime_residual(
    monkeypatch: pytest.MonkeyPatch,
    extra_residual_bytes: int,
    should_pass: bool,
) -> None:
    """Runtime drift is bounded; one byte beyond the protocol limit still fails."""

    import vrl.generation.execution.worker as worker_module
    from vrl.utils.cuda_memory import CumemPool

    core = _core(_SleepModel(device="cuda:0"))
    core._cumem = CumemPool(_FakeCuMem(), "weights")
    baseline = 1024
    core._preload_gpu_used_bytes = baseline
    residual = baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + extra_residual_bytes
    readings = iter((10 * 1024**3, residual))
    monkeypatch.setattr(worker_module, "gpu_used_bytes", lambda: next(readings))

    if should_pass:
        snapshot = core.sleep()
        assert snapshot.residual_gpu_used_bytes == residual
        return
    with pytest.raises(RuntimeError, match="incomplete cumem memory parking"):
        core.sleep()


def test_cumem_sleep_and_wake_use_pool_state_for_idempotency() -> None:
    from vrl.utils.cuda_memory import CumemPool

    model = _SleepModel(device="cuda:0")
    core = _core(model)
    fake = _FakeCuMem()
    core._cumem = CumemPool(fake, "weights")

    core.sleep()
    core.sleep()
    core.wake()
    core.wake()

    assert fake.sleep_calls == [("weights",)]
    assert fake.wake_calls == [["weights"]]


def test_generation_execution_does_not_reenter_one_shot_cumem_scope() -> None:
    from vrl.utils.cuda_memory import CumemPool

    model = _SleepModel(device="cuda:0")
    core = _core(model)
    core.executor = _PipelinedExecutor(model)
    fake = _FakeCuMem()
    core._cumem = CumemPool(fake, "weights")
    with core._cumem.building():
        pass
    request = SimpleNamespace(policy_version=1)
    engine_plan = object()
    sample_rows: list[Any] = []

    result = core.execute_request_pipelined(request, engine_plan, sample_rows)

    assert result == (request, sample_rows, engine_plan)
    assert fake.pool_tags == ["weights"]


def test_load_policy_pools_model_when_sleep_offload_and_cumem_available(monkeypatch) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    fake = _FakeCuMem()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: fake)
    core = _core(None, sleep_offload=True)
    sentinel = object()
    core._build_executor = lambda: sentinel  # type: ignore[method-assign]

    built = core._build_executor_maybe_pooled()

    assert built is sentinel
    assert fake.pool_tags == ["weights"]  # model allocated inside the cumem pool
    assert core._cumem is not None  # sleep/wake will now route through cumem
    assert core._cumem._allocator is fake


def test_failed_pooled_build_closes_retained_pool(monkeypatch) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    fake = _FakeCuMem()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("build failed"),
    )

    with pytest.raises(RuntimeError, match="build failed"):
        core._build_executor_maybe_pooled()

    assert "weights" not in fake.allocator_and_pools
    assert core._cumem is None


def test_load_policy_falls_back_when_cumem_unavailable(monkeypatch) -> None:
    import vrl.utils.cuda_memory as cuda_memory_mod

    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: None)
    core = _core(None, sleep_offload=True)
    sentinel = object()
    core._build_executor = lambda: sentinel  # type: ignore[method-assign]

    built = core._build_executor_maybe_pooled()

    assert built is sentinel
    assert core._cumem is None  # naive sleep/wake path


def test_load_policy_does_not_pool_without_sleep_offload(monkeypatch) -> None:
    """A resident worker without phase parking never enters the cumem pool."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    called: list[bool] = []
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: called.append(True))
    core = _core(None, sleep_offload=False)
    core._build_executor = lambda: object()  # type: ignore[method-assign]

    core._build_executor_maybe_pooled()

    assert called == []  # allocator never consulted
    assert core._cumem is None


def test_ar_cpu_fallback_does_not_enter_cumem_pool(monkeypatch) -> None:
    """AR decode may call empty_cache, which is incompatible with the pool scope."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    called: list[bool] = []
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: called.append(True))
    core = _core(None, sleep_offload=True, family="janus_pro")
    sentinel = object()
    core._build_executor = lambda: sentinel  # type: ignore[method-assign]

    built = core._build_executor_maybe_pooled()

    assert built is sentinel
    assert called == []
    assert core._cumem is None


def test_cpu_fallback_rejects_executor_without_movable_model(monkeypatch) -> None:
    import vrl.generation.execution.worker as worker_module

    core = _core(None, sleep_offload=True)
    monkeypatch.setattr(worker_module.CumemPool, "try_create", lambda tag=None: None)
    core._build_executor = lambda: _Executor(object())  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"requires executor\.model\.to"):
        core.load_policy()

    assert core.executor is None
    assert core._preload_gpu_used_bytes is None


def test_diffusion_cpu_fallback_requires_frozen_component_parking(monkeypatch) -> None:
    import vrl.generation.execution.worker as worker_module

    class _MovableModel:
        def to(self, device: Any) -> _MovableModel:
            del device
            return self

    core = _core(None, sleep_offload=True)
    monkeypatch.setattr(worker_module.CumemPool, "try_create", lambda tag=None: None)
    core._build_executor = lambda: _Executor(_MovableModel())  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="requires move_frozen_components"):
        core.load_policy()

    assert core.executor is None
    assert core._preload_gpu_used_bytes is None


def test_executor_identity_failure_rolls_back_loaded_policy(monkeypatch) -> None:
    import vrl.generation.execution.worker as worker_module

    class _WrongExecutor(_Executor):
        family = "wrong"

    core = _core(None, sleep_offload=True)
    monkeypatch.setattr(worker_module.CumemPool, "try_create", lambda tag=None: None)
    core._build_executor = lambda: _WrongExecutor(_SleepModel())  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="executor identity does not match"):
        core.load_policy()

    assert core.executor is None
    assert core._preload_gpu_used_bytes is None


def test_sleep_move_failure_is_not_swallowed() -> None:
    class _FailingModel(_SleepModel):
        def to(self, device: Any) -> _SleepModel:
            raise RuntimeError(f"move to {device} failed")

    core = _core(_FailingModel())

    with pytest.raises(RuntimeError, match="parking and rollback both failed"):
        core.sleep()

    assert core._cpu_parked_device is None


def test_sleep_move_failure_does_not_commit_false_parked_state() -> None:
    class _FailOnceModel(_SleepModel):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def to(self, device: Any) -> _SleepModel:
            self.to_calls.append(device)
            if device == "cpu" and not self.failed:
                self.failed = True
                raise RuntimeError("first CPU move failed")
            return self

    model = _FailOnceModel()
    core = _core(model)

    with pytest.raises(RuntimeError, match="first CPU move failed"):
        core.sleep()

    assert core._cpu_parked_device is None
    snapshot = core.sleep()
    assert snapshot.backend == "cpu_offload"
    assert model.to_calls == ["cpu", "cuda:0", "cpu"]


def test_frozen_move_failure_rolls_back_before_retry() -> None:
    class _FailOnceFrozenModel(_SleepModel):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def move_frozen_components(self, device: Any) -> None:
            self.frozen_calls.append(device)
            if device == "cpu" and not self.failed:
                self.failed = True
                raise RuntimeError("first frozen CPU move failed")

    model = _FailOnceFrozenModel()
    core = _core(model)

    with pytest.raises(RuntimeError, match="first frozen CPU move failed"):
        core.sleep()

    assert core._cpu_parked_device is None
    snapshot = core.sleep()
    assert snapshot.backend == "cpu_offload"
    assert model.to_calls == ["cpu", "cuda:0", "cpu"]
    assert model.frozen_calls == ["cpu", "cuda:0", "cpu"]


def test_wake_failure_keeps_cpu_parked_state_for_retry() -> None:
    class _FailOnceFrozenWakeModel(_SleepModel):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def move_frozen_components(self, device: Any) -> None:
            self.frozen_calls.append(device)
            if device == "cuda:0" and not self.failed:
                self.failed = True
                raise RuntimeError("first frozen wake failed")

    model = _FailOnceFrozenWakeModel()
    core = _core(model)
    core.sleep()

    with pytest.raises(RuntimeError, match="first frozen wake failed"):
        core.wake()

    assert core._cpu_parked_device == "cuda:0"
    core.wake()
    assert core._cpu_parked_device is None
    assert model.to_calls == ["cpu", "cuda:0", "cuda:0"]
    assert model.frozen_calls == ["cpu", "cuda:0", "cuda:0"]


@pytest.mark.parametrize(
    ("extra_residual_bytes", "should_pass"),
    ((0, True), (1, False)),
)
def test_cpu_offload_bounds_lazy_cuda_runtime_residual(
    monkeypatch: pytest.MonkeyPatch,
    extra_residual_bytes: int,
    should_pass: bool,
) -> None:
    import vrl.generation.execution.worker as worker_module

    core = _core(_SleepModel())
    baseline = 1024
    core._preload_gpu_used_bytes = baseline
    residual = baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + extra_residual_bytes
    readings = iter((10 * 1024**3, residual))
    monkeypatch.setattr(worker_module, "gpu_used_bytes", lambda: next(readings))

    if should_pass:
        snapshot = core.sleep()
        assert snapshot.residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        assert snapshot.residual_gpu_used_bytes == residual
        return
    with pytest.raises(RuntimeError, match="incomplete cpu_offload memory parking"):
        core.sleep()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_cpu_fallback_returns_to_preload_process_baseline(monkeypatch) -> None:
    import vrl.generation.execution.worker as worker_module
    from vrl.utils.cuda_memory import release_cuda_memory_for_parking

    class _TinyCudaModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1024, 1024, device="cuda"))

        @property
        def device(self) -> torch.device:
            return self.weight.device

        def move_frozen_components(self, device: Any) -> None:
            del device

    core = _core(None)
    # The production proof intentionally measures the whole device and fails
    # closed if any process grows during handoff. This real-CUDA mechanism test
    # isolates the pytest process because desktop GPU clients can legitimately
    # change the device-wide reading between the two samples; the preceding
    # deterministic test covers strict device-wide residual rejection.
    monkeypatch.setattr(
        worker_module,
        "gpu_used_bytes",
        lambda: int(torch.cuda.memory_reserved()),
    )
    monkeypatch.setattr(
        worker_module,
        "release_cuda_memory_for_parking",
        release_cuda_memory_for_parking,
    )
    baseline = worker_module.gpu_used_bytes()
    model = _TinyCudaModel()
    core.executor = _Executor(model)
    core._preload_gpu_used_bytes = baseline

    snapshot = core.sleep()

    assert model.weight.device.type == "cpu"
    assert snapshot.residual_gpu_used_bytes <= snapshot.baseline_gpu_used_bytes
    core.wake()
    assert model.weight.device.type == "cuda"


def test_release_policy_wakes_and_closes_cumem_pool(monkeypatch) -> None:
    import vrl.generation.execution.worker as worker_module
    from vrl.utils.cuda_memory import CumemPool

    monkeypatch.setattr(worker_module, "release_cuda_memory", lambda **kwargs: None)
    core = _core(_SleepModel())
    fake = _FakeCuMem()
    pool = CumemPool(fake, "weights")
    with pool.building():
        pass
    core._cumem = pool
    core.sleep()

    core.release_policy()

    assert fake.wake_calls == [["weights"]]
    assert "weights" not in fake.allocator_and_pools
    assert core.executor is None
    assert core._cumem is None
    assert core._preload_gpu_used_bytes is None
    assert core._cpu_parked_device is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cumem_one_shot_scope_sleep_wake_in_subprocess() -> None:
    """One real scope round-trips; a second scope is blocked before C++ abort."""

    from vrl.utils.cuda_memory import CumemPool

    if CumemPool.try_create("vrl-one-shot-preflight") is None:
        pytest.skip("vLLM CuMemAllocator is unavailable")

    script = textwrap.dedent(
        """
        import gc
        import torch
        from vrl.utils.cuda_memory import CumemPool

        pool = CumemPool.require("vrl-one-shot-smoke")
        torch.cuda.synchronize()
        free_before, total = torch.cuda.mem_get_info()
        baseline = total - free_before
        with pool.building():
            value = torch.arange(4096, device="cuda", dtype=torch.float32)
        expected = value.cpu()
        pool.sleep()
        pool.wake()
        assert torch.equal(value.cpu(), expected)
        try:
            pool.building()
        except RuntimeError as error:
            assert "one-shot" in str(error)
        else:
            raise AssertionError("second CuMem building scope was not rejected")
        del value
        gc.collect()
        pool.close()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after, total = torch.cuda.mem_get_info()
        residual = total - free_after
        assert residual <= baseline + 4 * 1024 * 1024, (baseline, residual)
        """,
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
