"""GenerationWorkerCore sleep/wake (SPRINT_frozen_component_preservation, defect A).

Level-1 offload-and-restore for on-demand activation: ``sleep`` delegates to the
worker's memory-parking owner. The backend follows from the family's declared
parking profile and its residency mode, never from what happens to be installed:
a CuMem family without a working allocator fails loud rather than degrading.
Wan's model-owned path resets its existing Accelerate hooks; model-profile
families move to CPU. ``wake`` restores active residency without a cold reload;
``release_policy`` remains the level-2 discard path.
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

    import vrl.generation.execution.memory_parking as parking_module

    monkeypatch.setattr(parking_module, "gpu_used_bytes", lambda: 0)
    monkeypatch.setattr(parking_module, "release_cuda_memory_for_parking", lambda: None)


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


class _PipelineOffloadModel(_SleepModel):
    """Meta-backed model whose Accelerate hooks already own CPU residency."""

    uses_pipeline_cpu_offload = True

    def __init__(self) -> None:
        super().__init__()
        self.pipeline_cpu_offload_healthy = True
        self.reset_calls = 0
        self.reset_error: BaseException | None = None

    def reset_pipeline_cpu_offload(self) -> None:
        self.reset_calls += 1
        if self.reset_error is not None:
            self.pipeline_cpu_offload_healthy = False
            raise self.reset_error

    def to(self, device: Any) -> _PipelineOffloadModel:
        raise AssertionError(f"pipeline-offloaded model must not move as a whole: {device}")

    def move_frozen_components(self, device: Any) -> None:
        raise AssertionError(f"pipeline-offloaded components must retain their hooks: {device}")


class _Executor:
    def __init__(
        self,
        model: Any,
        *,
        family: str = "sd3_5",
        task: str = "t2i",
    ) -> None:
        self.model = model
        self.family = family
        self.task = task

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
        self.sleep_error: BaseException | None = None
        self.wake_error: BaseException | None = None

    @contextlib.contextmanager
    def use_memory_pool(self, tag: Any = None):
        self.pool_tags.append(tag)
        self.allocator_and_pools[tag] = object()
        yield

    def sleep(self, offload_tags: Any = None) -> None:
        self.sleep_calls.append(offload_tags)
        if self.sleep_error is not None:
            raise self.sleep_error

    def wake_up(self, tags: Any = None) -> None:
        self.wake_calls.append(tags)
        if self.wake_error is not None:
            raise self.wake_error


def _core(
    model: Any | None,
    *,
    sleep_offload: bool = False,
    family: str = "sd3_5",
    pipeline_offload_mode: str | None = None,
) -> GenerationWorkerCore:
    if pipeline_offload_mode is None:
        pipeline_offload_mode = (
            "sequential" if bool(getattr(model, "uses_pipeline_cpu_offload", False)) else "none"
        )
    contract = GenerationRuntimeLaunchContract(
        family=family,
        model_build=(
            {"rollout": {"pipeline_offload_mode": pipeline_offload_mode}}
            if pipeline_offload_mode != "none"
            else {}
        ),
        expected_model_identity={"schema": "test"},
        policy_version=1,
        sleep_offload=sleep_offload,
    )
    core = GenerationWorkerCore("rollout-0", contract)
    core.executor = (
        _Executor(model, family=family, task=core.family_entry.task) if model is not None else None
    )
    return core


def _build_executor(core: GenerationWorkerCore, model: Any) -> _Executor:
    """Return a fake whose identity matches the worker launch contract."""

    return _Executor(
        model,
        family=core.family_entry.family,
        task=core.family_entry.task,
    )


def _install_cumem_pool(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeCuMem,
) -> None:
    """Route the parking owner's public CuMem factory through a CPU fake."""

    import vrl.generation.execution.memory_parking as parking_module

    monkeypatch.setattr(
        parking_module.CumemPool,
        "try_create",
        classmethod(
            lambda cls, tag=None: cls(fake, tag or "weights"),
        ),
    )


def _make_cumem_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a box without vLLM: ``require`` then raises through ``try_create``."""

    import vrl.generation.execution.memory_parking as parking_module

    monkeypatch.setattr(
        parking_module.CumemPool,
        "try_create",
        classmethod(lambda _cls, tag=None: None),
    )


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


def test_worker_keeps_backend_state_behind_one_parking_owner() -> None:
    core = _core(_SleepModel())

    assert "_memory_parking" in vars(core)
    assert {
        "_preload_gpu_used_bytes",
        "_cpu_parked_device",
        "_pipeline_offload_parked",
        "_pipeline_offload_error",
        "_cumem",
    }.isdisjoint(vars(core))
    assert {
        "_supports_cumem_pool",
        "_requires_frozen_component_parking",
        "_pipeline_offload_mode",
        "_pool",
        "_baseline_gpu_used_bytes",
        "_cpu_restore_device",
    }.isdisjoint(vars(core._memory_parking))


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


def test_parked_worker_rejects_execution_until_wake() -> None:
    model = _SleepModel(device="cuda:0")
    core = _core(model)
    core.executor = _PipelinedExecutor(
        model,
        family=core.family_entry.family,
        task=core.family_entry.task,
    )
    request = SimpleNamespace(policy_version=1)

    core.sleep()
    with pytest.raises(RuntimeError, match=r"parked.*refusing execute_request_pipelined"):
        core.execute_request_pipelined(request, object(), [])

    core.wake()
    assert core.execute_request_pipelined(request, "plan", ["rows"]) == (
        request,
        ["rows"],
        "plan",
    )


def test_failed_physical_proof_quarantines_already_moved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.execution.memory_parking as parking_module

    model = _SleepModel(device="cuda:0")
    core = _core(model)
    monkeypatch.setattr(
        parking_module,
        "release_cuda_memory_for_parking",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA synchronize failed")),
    )

    with pytest.raises(RuntimeError, match="CUDA synchronize failed"):
        core.sleep()
    assert model.to_calls == ["cpu"]
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()


def test_pipeline_offload_sleep_wake_preserves_accelerate_hooks() -> None:
    model = _PipelineOffloadModel()
    core = _core(
        None,
        sleep_offload=True,
        family="wan_2_1_i2v",
        pipeline_offload_mode="sequential",
    )
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    snapshot = core.sleep()
    core.sleep()
    core.wake()
    core.wake()

    assert snapshot.backend == "cpu_offload"
    assert snapshot.residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
    assert model.reset_calls == 1
    assert core.executor is not None


def test_pipeline_offload_never_consults_cumem_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.execution.memory_parking as parking_module

    attempts: list[str | None] = []
    monkeypatch.setattr(
        parking_module.CumemPool,
        "try_create",
        classmethod(lambda _cls, tag=None: attempts.append(tag)),
    )
    model = _PipelineOffloadModel()
    core = _core(
        None,
        sleep_offload=True,
        family="wan_2_1_i2v",
        pipeline_offload_mode="sequential",
    )
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]

    core.load_policy()
    core.sleep()

    assert attempts == []


def test_pipeline_offload_reset_failure_quarantines_worker() -> None:
    model = _PipelineOffloadModel()
    model.reset_error = RuntimeError("hook reinstall failed")
    core = _core(
        None,
        sleep_offload=True,
        family="wan_2_1_i2v",
        pipeline_offload_mode="sequential",
    )
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    with pytest.raises(RuntimeError, match="worker is quarantined"):
        core.sleep()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()

    assert model.reset_calls == 1


def test_pipeline_offload_residual_failure_does_not_commit_parked_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.execution.memory_parking as parking_module

    model = _PipelineOffloadModel()
    baseline = 1024
    readings = iter(
        (
            baseline,
            10 * 1024**3,
            baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + 1,
        ),
    )
    monkeypatch.setattr(parking_module, "gpu_used_bytes", lambda: next(readings))
    core = _core(
        None,
        sleep_offload=True,
        family="wan_2_1_i2v",
        pipeline_offload_mode="sequential",
    )
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    with pytest.raises(RuntimeError, match="failed physical GPU parking validation"):
        core.sleep()

    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()


def test_failed_forward_resets_pipeline_offload_before_returning_error() -> None:
    from vrl.generation.execution.chunks import SampleChunk
    from vrl.generation.execution.types import ChunkExecutionEnvelope
    from vrl.generation.types import GenerationRequest

    class _FailingExecutor(_Executor):
        def forward_chunk_plan(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("forward failed")

    model = _PipelineOffloadModel()
    model.device = "cpu"
    core = _core(model, family="wan_2_1_i2v")
    core.executor = _FailingExecutor(model)
    request = GenerationRequest(
        request_id="failed-forward",
        family="wan_2_1_i2v",
        task="i2v",
        inputs=["prompt"],
        samples_per_prompt=1,
        policy_version=1,
    )
    envelope = ChunkExecutionEnvelope(
        request=request,
        chunk=SampleChunk(
            prompt_index=0,
            prompt="prompt",
            sample_start=0,
            sample_count=1,
        ),
    )

    result = core.execute_chunk(envelope)

    assert result.error == "forward failed"
    assert model.reset_calls == 1


def test_failed_forward_and_hook_reset_quarantine_without_retry() -> None:
    from vrl.generation.execution.chunks import SampleChunk
    from vrl.generation.execution.types import ChunkExecutionEnvelope
    from vrl.generation.types import GenerationRequest

    class _FailingExecutor(_Executor):
        def forward_chunk_plan(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("forward failed")

    model = _PipelineOffloadModel()
    model.device = "cpu"
    model.reset_error = RuntimeError("hook removal failed")
    core = _core(model, family="wan_2_1_i2v")
    core.executor = _FailingExecutor(model)
    request = GenerationRequest(
        request_id="failed-recovery",
        family="wan_2_1_i2v",
        task="i2v",
        inputs=["prompt"],
        samples_per_prompt=1,
        policy_version=1,
    )
    envelope = ChunkExecutionEnvelope(
        request=request,
        chunk=SampleChunk(
            prompt_index=0,
            prompt="prompt",
            sample_start=0,
            sample_count=1,
        ),
    )

    with pytest.raises(RuntimeError, match="worker is quarantined"):
        core.execute_chunk(envelope)
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing execute_chunk"):
        core.execute_chunk(envelope)

    assert model.reset_calls == 1


def test_partial_slot_activation_failure_is_not_returned_as_retryable_result() -> None:
    from vrl.generation.execution.chunks import SampleChunk
    from vrl.generation.execution.types import ChunkExecutionEnvelope
    from vrl.generation.types import GenerationRequest

    class _BrokenActivationModel(_PipelineOffloadModel):
        supports_versioned_trainable_state = True

        def has_trainable_state(self, _version: int) -> bool:
            return True

        def activate_trainable_state(self, _version: int) -> None:
            self.pipeline_cpu_offload_healthy = False
            self.reset_error = RuntimeError("partial policy state")
            raise RuntimeError("weight copy failed")

    model = _BrokenActivationModel()
    model.device = "cpu"
    core = _core(model, family="wan_2_1_i2v")
    core._uses_versioned_slots = True
    request = GenerationRequest(
        request_id="failed-activation",
        family="wan_2_1_i2v",
        task="i2v",
        inputs=["prompt"],
        samples_per_prompt=1,
        policy_version=1,
    )
    envelope = ChunkExecutionEnvelope(
        request=request,
        chunk=SampleChunk(
            prompt_index=0,
            prompt="prompt",
            sample_start=0,
            sample_count=1,
        ),
    )

    with pytest.raises(RuntimeError, match="worker is quarantined"):
        core.execute_chunk(envelope)

    assert model.reset_calls == 1


def test_pipelined_oom_resets_pipeline_hooks_before_typed_retry() -> None:
    from vrl.generation.execution.types import PipelinedRequestOutOfMemory
    from vrl.generation.types import GenerationRequest

    class _OomPipelinedExecutor(_PipelinedExecutor):
        def forward_plan_pipelined(
            self,
            _request: Any,
            _sample_rows: Any,
            _engine_plan: Any,
        ) -> Any:
            raise RuntimeError("CUDA out of memory")

    model = _PipelineOffloadModel()
    model.device = "cpu"
    core = _core(model, family="wan_2_1_i2v")
    core.executor = _OomPipelinedExecutor(model)
    request = GenerationRequest(
        request_id="pipelined-oom",
        family="wan_2_1_i2v",
        task="i2v",
        inputs=["prompt"],
        samples_per_prompt=1,
        policy_version=1,
    )

    result = core.execute_request_pipelined(request, object(), [])

    assert isinstance(result, PipelinedRequestOutOfMemory)
    assert model.reset_calls == 1


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


def test_cumem_sleep_wake_uses_allocator_not_module_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SleepModel(device="cuda:0")
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    snapshot = core.sleep()
    core.wake()

    # The whole pooled model is released/restored through the allocator; the naive
    # per-module .to() round trip is bypassed entirely.
    tag = "vrl:generation:rollout-0:weights"
    assert fake.sleep_calls == [(tag,)]
    assert fake.wake_calls == [[tag]]
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

    import vrl.generation.execution.memory_parking as parking_module

    baseline = 1024
    residual = baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + extra_residual_bytes
    readings = iter((baseline, 10 * 1024**3, residual))
    monkeypatch.setattr(parking_module, "gpu_used_bytes", lambda: next(readings))
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    model = _SleepModel(device="cuda:0")
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    if should_pass:
        snapshot = core.sleep()
        assert snapshot.residual_gpu_used_bytes == residual
        return
    with pytest.raises(RuntimeError, match="failed physical GPU parking validation"):
        core.sleep()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()


def test_cumem_sleep_and_wake_use_pool_state_for_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SleepModel(device="cuda:0")
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    core.sleep()
    core.sleep()
    core.wake()
    core.wake()

    tag = "vrl:generation:rollout-0:weights"
    assert fake.sleep_calls == [(tag,)]
    assert fake.wake_calls == [[tag]]


def test_cumem_sleep_failure_requires_actor_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuMem()
    fake.sleep_error = RuntimeError("partial unmap")
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        _SleepModel(),
    )
    core.load_policy()

    with pytest.raises(RuntimeError, match="partial unmap"):
        core.sleep()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()
    with pytest.raises(RuntimeError, match="must terminate"):
        core.release_policy()
    assert core.executor is not None


def test_cumem_wake_failure_cannot_retry_or_close_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        _SleepModel(),
    )
    core.load_policy()
    core.sleep()
    fake.wake_error = RuntimeError("partial remap")

    with pytest.raises(RuntimeError, match="partial remap"):
        core.wake()
    fake.wake_error = None
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()
    with pytest.raises(RuntimeError, match="must terminate"):
        core.release_policy()
    assert core.executor is not None


def test_generation_execution_does_not_reenter_one_shot_cumem_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SleepModel(device="cuda:0")
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _PipelinedExecutor(  # type: ignore[method-assign]
        model,
        family=core.family_entry.family,
        task=core.family_entry.task,
    )
    core.load_policy()
    request = SimpleNamespace(policy_version=1)
    engine_plan = object()
    sample_rows: list[Any] = []

    result = core.execute_request_pipelined(request, engine_plan, sample_rows)

    assert result == (request, sample_rows, engine_plan)
    assert fake.pool_tags == ["vrl:generation:rollout-0:weights"]


def test_load_policy_pools_model_when_sleep_offload_and_cumem_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    model = _SleepModel()
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]

    core.load_policy()

    assert core.executor is not None
    assert core.executor.model is model
    assert fake.pool_tags == ["vrl:generation:rollout-0:weights"]


def test_failed_pooled_build_closes_retained_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("build failed"),
    )

    with pytest.raises(RuntimeError, match="build failed"):
        core.load_policy()

    assert "vrl:generation:rollout-0:weights" not in fake.allocator_and_pools
    assert core.executor is None


def test_failed_pooled_build_cleanup_makes_worker_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.execution.memory_parking as parking_module

    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    monkeypatch.setattr(
        parking_module.CumemPool,
        "close",
        lambda _pool: (_ for _ in ()).throw(RuntimeError("pool close failed")),
    )
    core = _core(None, sleep_offload=True)
    build_calls = 0

    def fail_build() -> Any:
        nonlocal build_calls
        build_calls += 1
        raise RuntimeError("build failed")

    core._build_executor = fail_build  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="policy load and cleanup both failed"):
        core.load_policy()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing policy build"):
        core.load_policy()
    assert build_calls == 1


def test_loaded_validation_drops_traceback_model_before_pool_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import vrl.generation.execution.memory_parking as parking_module

    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    original_close = parking_module.CumemPool.close
    model_ref: weakref.ReferenceType[Any] | None = None
    model_was_released: list[bool] = []

    def build_mismatched_backend() -> _Executor:
        nonlocal model_ref
        model = _PipelineOffloadModel()
        model_ref = weakref.ref(model)
        return _build_executor(core, model)

    def assert_model_released(pool: Any) -> None:
        gc.collect()
        assert model_ref is not None
        model_was_released.append(model_ref() is None)
        original_close(pool)

    core = _core(None, sleep_offload=True)
    core._build_executor = build_mismatched_backend  # type: ignore[method-assign]
    monkeypatch.setattr(parking_module.CumemPool, "close", assert_model_released)

    with pytest.raises(RuntimeError, match="residency mechanisms must be mutually exclusive"):
        core.load_policy()

    assert model_was_released == [True]


def test_sleep_offload_requires_cumem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CuMem-profile family refuses to load at all when the allocator is missing."""
    _make_cumem_unavailable(monkeypatch)
    core = _core(None, sleep_offload=True)  # sd3_5 declares GenerationParkingProfile.CUMEM
    built: list[bool] = []

    def build() -> Any:
        built.append(True)
        return _build_executor(core, _SleepModel())

    core._build_executor = build  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="CuMemAllocator is required"):
        core.load_policy()

    assert core.executor is None
    # The pool is claimed before the weights load, so a misconfigured box never
    # pays for a multi-GiB build it cannot park.
    assert built == []


def test_load_policy_does_not_pool_without_sleep_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resident worker without phase parking never enters the cumem pool."""
    import vrl.generation.execution.memory_parking as parking_module

    called: list[bool] = []
    monkeypatch.setattr(
        parking_module.CumemPool,
        "try_create",
        classmethod(lambda _cls, tag=None: called.append(True)),
    )
    core = _core(None, sleep_offload=False)
    model = _SleepModel()
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]

    core.load_policy()

    assert called == []


def test_ar_model_parking_does_not_enter_cumem_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AR decode may call empty_cache, which is incompatible with the pool scope."""
    import vrl.generation.execution.memory_parking as parking_module

    called: list[bool] = []
    monkeypatch.setattr(
        parking_module.CumemPool,
        "try_create",
        classmethod(lambda _cls, tag=None: called.append(True)),
    )
    core = _core(None, sleep_offload=True, family="janus_pro")
    model = _SleepModel()
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]

    core.load_policy()

    assert core.executor is not None
    assert core.executor.model is model
    assert called == []


def test_model_parking_rejects_executor_without_movable_model() -> None:
    core = _core(None, sleep_offload=True, family="janus_pro")  # declares MODEL profile
    core._build_executor = lambda: _build_executor(core, object())  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"requires executor\.model\.to"):
        core.load_policy()

    assert core.executor is None


def test_executor_identity_failure_rolls_back_loaded_policy() -> None:
    core = _core(None, sleep_offload=True, family="janus_pro")
    core._build_executor = lambda: _Executor(  # type: ignore[method-assign]
        _SleepModel(),
        family="wrong",
        task=core.family_entry.task,
    )

    with pytest.raises(ValueError, match="executor identity does not match"):
        core.load_policy()

    assert core.executor is None


def test_sleep_move_failure_is_not_swallowed() -> None:
    class _FailingModel(_SleepModel):
        def to(self, device: Any) -> _SleepModel:
            raise RuntimeError(f"move to {device} failed")

    core = _core(_FailingModel())

    with pytest.raises(RuntimeError, match="parking and rollback both failed"):
        core.sleep()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing sleep"):
        core.sleep()


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

    core.wake()
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
    import vrl.generation.execution.memory_parking as parking_module

    baseline = 1024
    residual = baseline + CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT + extra_residual_bytes
    readings = iter((baseline, 10 * 1024**3, residual))
    monkeypatch.setattr(parking_module, "gpu_used_bytes", lambda: next(readings))
    model = _SleepModel()
    core = _core(None, sleep_offload=True, family="janus_pro")  # declares MODEL profile
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()

    if should_pass:
        snapshot = core.sleep()
        assert snapshot.residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        assert snapshot.residual_gpu_used_bytes == residual
        return
    with pytest.raises(RuntimeError, match="failed physical GPU parking validation"):
        core.sleep()
    with pytest.raises(RuntimeError, match=r"quarantined.*refusing wake"):
        core.wake()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_model_parking_returns_to_preload_process_baseline(monkeypatch) -> None:
    import vrl.generation.execution.memory_parking as parking_module
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

    core = _core(None, sleep_offload=True, family="janus_pro")  # declares MODEL profile
    # The production proof intentionally measures the whole device and fails
    # closed if any process grows during handoff. This real-CUDA mechanism test
    # isolates the pytest process because desktop GPU clients can legitimately
    # change the device-wide reading between the two samples; the preceding
    # deterministic test covers strict device-wide residual rejection.
    monkeypatch.setattr(
        parking_module,
        "gpu_used_bytes",
        lambda: int(torch.cuda.memory_reserved()),
    )
    monkeypatch.setattr(
        parking_module,
        "release_cuda_memory_for_parking",
        release_cuda_memory_for_parking,
    )
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        _TinyCudaModel(),
    )
    core.load_policy()
    assert core.executor is not None
    model = core.executor.model

    snapshot = core.sleep()

    assert model.weight.device.type == "cpu"
    assert snapshot.residual_gpu_used_bytes <= snapshot.baseline_gpu_used_bytes
    core.wake()
    assert model.weight.device.type == "cuda"


def test_module_parking_release_then_reload_has_fresh_restore_state() -> None:
    first_model = _SleepModel()
    second_model = _SleepModel()
    core = _core(None, sleep_offload=True, family="janus_pro")
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        first_model,
    )
    core.load_policy()
    core.sleep()
    core.release_policy()

    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        second_model,
    )
    core.load_policy()
    core.sleep()
    core.wake()

    assert first_model.to_calls == ["cpu"]
    assert second_model.to_calls == ["cpu", "cuda:0"]


def test_pipeline_offload_release_then_reload_has_fresh_hook_state() -> None:
    first_model = _PipelineOffloadModel()
    second_model = _PipelineOffloadModel()
    core = _core(
        None,
        sleep_offload=True,
        family="wan_2_1_i2v",
        pipeline_offload_mode="sequential",
    )
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        first_model,
    )
    core.load_policy()
    core.sleep()
    core.release_policy()

    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        second_model,
    )
    core.load_policy()
    core.sleep()
    core.wake()

    assert first_model.reset_calls == 1
    assert second_model.reset_calls == 1


def test_cumem_parking_release_then_reload_claims_a_fresh_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    first_model = _SleepModel()
    second_model = _SleepModel()
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        first_model,
    )
    core.load_policy()
    core.sleep()
    core.release_policy()

    core._build_executor = lambda: _build_executor(  # type: ignore[method-assign]
        core,
        second_model,
    )
    core.load_policy()
    core.sleep()
    core.wake()

    tag = "vrl:generation:rollout-0:weights"
    assert fake.pool_tags == [tag, tag]
    assert fake.sleep_calls == [(tag,), (tag,)]
    assert fake.wake_calls == [[tag], [tag]]


def test_release_policy_wakes_and_closes_cumem_pool(monkeypatch) -> None:
    import vrl.generation.execution.memory_parking as parking_module

    monkeypatch.setattr(parking_module, "release_cuda_memory", lambda **kwargs: None)
    fake = _FakeCuMem()
    _install_cumem_pool(monkeypatch, fake)
    model = _SleepModel()
    core = _core(None, sleep_offload=True)
    core._build_executor = lambda: _build_executor(core, model)  # type: ignore[method-assign]
    core.load_policy()
    core.sleep()

    core.release_policy()

    tag = "vrl:generation:rollout-0:weights"
    assert fake.wake_calls == [[tag]]
    assert tag not in fake.allocator_and_pools
    assert core.executor is None


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
