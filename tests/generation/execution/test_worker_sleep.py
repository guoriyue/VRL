"""GenerationWorkerCore sleep/wake (SPRINT_frozen_component_preservation, defect A).

Level-1 offload-and-restore for the release-after-collect lease: ``sleep`` parks
the loaded model on host RAM (transformer via ``nn.Module.to`` plus the frozen
VAE / text-encoders via ``move_frozen_components``) while keeping the executor
alive, and ``wake`` restores it onto the captured GPU without a cold reload. This
is the level-2 ``release_policy`` (discard + disk reload) counterpart that lets a
colocated trainer reclaim the GPU between collect and train.

The executor is injected directly so load_policy() short-circuits — no model build.
"""

from __future__ import annotations

import contextlib
from typing import Any

import vrl.generation.execution.worker as worker_mod
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


class _FakeCuMem:
    """Stand-in for vLLM's CuMemAllocator: records pool/sleep/wake calls."""

    def __init__(self) -> None:
        self.pool_tags: list[Any] = []
        self.sleep_calls: list[Any] = []
        self.wake_calls: list[Any] = []

    @contextlib.contextmanager
    def use_memory_pool(self, tag: Any = None):
        self.pool_tags.append(tag)
        yield

    def sleep(self, offload_tags: Any = None) -> None:
        self.sleep_calls.append(offload_tags)

    def wake_up(self, tags: Any = None) -> None:
        self.wake_calls.append(tags)


def _unused_builder(spec: Any) -> Any:  # pragma: no cover - executor is injected
    raise RuntimeError("builder must not run; the test injects core.executor directly")


_MODULE = "tests.generation.execution.test_worker_sleep"


def _core(model: Any | None, *, sleep_offload: bool = False) -> GenerationWorkerCore:
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
    return core


def test_sleep_parks_model_and_frozen_on_cpu_keeping_executor() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)

    core.sleep()

    # Transformer (nn.Module.to) AND the unregistered frozen components both leave
    # the GPU — the whole offload-and-restore point.
    assert model.to_calls == ["cpu"]
    assert model.frozen_calls == ["cpu"]
    # Executor is retained (level-1), unlike release_policy which nulls it.
    assert core.executor is not None


def test_wake_restores_model_to_captured_device() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)

    core.sleep()
    core.wake()

    # Restored onto the device the model lived on before sleeping, no disk reload.
    assert model.to_calls == ["cpu", "cuda:0"]
    assert model.frozen_calls == ["cpu", "cuda:0"]
    assert core.executor is not None


def test_sleep_without_loaded_model_is_a_safe_no_op() -> None:
    core = _core(None)
    core.sleep()  # must not raise (nothing loaded)
    assert core.executor is None


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
    """A teardown-lease / resident worker never enters the cumem pool."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    called: list[bool] = []
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: called.append(True))
    core = _core(None, sleep_offload=False)
    core._build_executor = lambda: object()  # type: ignore[method-assign]

    core._build_executor_maybe_pooled()

    assert called == []  # allocator never consulted
    assert core._cumem is None
