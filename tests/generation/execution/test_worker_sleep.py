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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.models.diffusion.capabilities import diffusion_family_capability


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


def _unused_builder(spec: Any) -> Any:  # pragma: no cover - executor is injected
    raise RuntimeError("builder must not run; the test injects core.executor directly")


_MODULE = "tests.generation.execution.test_worker_sleep"


def _core(
    model: Any | None,
    *,
    sleep_offload: bool = False,
) -> GenerationWorkerCore:
    capability = diffusion_family_capability("sd3_5", "t2i")
    extra: dict[str, Any] = {"family_capability": capability.to_dict()}
    if sleep_offload:
        extra["sleep_offload"] = True
    contract = GenerationRuntimeLaunchContract(
        family="sd3_5",
        task="t2i",
        policy_version=1,
        runtime_builder=f"{_MODULE}:_unused_builder",
        executor_cls=f"{_MODULE}:_Executor",
        extra=extra,
    )
    core = GenerationWorkerCore("rollout-0", contract)
    core.executor = _Executor(model) if model is not None else None
    # CPU fake: never inspect or mutate a real GPU merely because this unit test
    # happens to run on a CUDA host. The isolated CUDA twin below restores the real
    # probes explicitly.
    core._gpu_used_bytes = lambda: 0  # type: ignore[method-assign]
    core._release_cuda_memory_for_parking = lambda: None  # type: ignore[method-assign]
    core._parking_baseline_gpu_used_bytes = 0
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

    core.sleep()
    core.wake()

    # The whole pooled model is released/restored through the allocator; the naive
    # per-module .to() round trip is bypassed entirely.
    assert fake.sleep_calls == [("weights",)]
    assert fake.wake_calls == [["weights"]]
    assert model.to_calls == []
    assert model.frozen_calls == []


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


def test_sleep_offload_requires_family_parking_declaration() -> None:
    core = _core(None, sleep_offload=True)
    core.capability = replace(
        core.capability,
        supports_complete_memory_parking=False,
    )

    with pytest.raises(RuntimeError, match="does not declare complete memory parking"):
        core.load_policy()


def test_ar_cpu_fallback_does_not_enter_cumem_pool(monkeypatch) -> None:
    """AR decode may call empty_cache, which is incompatible with the pool scope."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    called: list[bool] = []
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: called.append(True))
    core = _core(None, sleep_offload=True)
    core.capability = replace(core.capability, trajectory_kind="ar_discrete")
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


def test_sleep_move_failure_is_not_swallowed() -> None:
    class _FailingModel(_SleepModel):
        def to(self, device: Any) -> _SleepModel:
            raise RuntimeError(f"move to {device} failed")

    core = _core(_FailingModel())

    with pytest.raises(RuntimeError, match="move to cpu failed"):
        core.sleep()


def test_sleep_rejects_residual_above_preload_baseline(monkeypatch) -> None:
    core = _core(_SleepModel())
    core._parking_baseline_gpu_used_bytes = 0
    readings = iter((1024, 1))
    monkeypatch.setattr(core, "_gpu_used_bytes", lambda: next(readings))

    with pytest.raises(
        RuntimeError,
        match=r"incomplete cpu_offload memory parking: loaded=1024 residual=1",
    ):
        core.sleep()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_cpu_fallback_returns_to_preload_physical_baseline() -> None:
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
    core._gpu_used_bytes = lambda: GenerationWorkerCore._gpu_used_bytes()  # type: ignore[method-assign]
    core._release_cuda_memory_for_parking = (  # type: ignore[method-assign]
        lambda: GenerationWorkerCore._release_cuda_memory_for_parking()
    )
    baseline = core._gpu_used_bytes()
    model = _TinyCudaModel()
    core.executor = _Executor(model)
    core._parking_baseline_gpu_used_bytes = baseline

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
    assert core._parking_baseline_gpu_used_bytes is None
    assert core._parking_restore_device is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cumem_one_shot_scope_sleep_wake_in_subprocess() -> None:
    """One real scope round-trips; a second scope is blocked before C++ abort."""

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
