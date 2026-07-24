"""Version-safety of GenerationWorkerCore.execute_request_pipelined (the
per-request stage-overlap path) — it must enforce the SAME slot / stale-slot /
version-mismatch guarantees as execute_chunk, at the request level, so a stale
request is never run + trained off-policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.execution.memory_parking import WorkerMemoryParking
from vrl.generation.execution.types import (
    PipelinedRequestOutOfMemory,
    StaleSlotDiscard,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract


class _Model:
    def __init__(self, available_versions: set[int]) -> None:
        self._available = available_versions
        self.activated: list[int] = []

    def has_trainable_state(self, version: int) -> bool:
        return version in self._available

    def activate_trainable_state(self, version: int) -> None:
        self.activated.append(version)


class _Executor:
    def __init__(self, model: _Model, *, error: RuntimeError | None = None) -> None:
        self.model = model
        self.error = error
        self.calls: list[tuple] = []

    def forward_plan_pipelined(self, request, sample_rows, plan):
        self.calls.append((request, sample_rows, plan))
        if self.error is not None:
            raise self.error
        return "GATHERED_OUTPUT"


def _core(*, executor, uses_slots: bool, policy_version: int | None):
    core = GenerationWorkerCore.__new__(GenerationWorkerCore)
    core.executor = executor
    core.worker_id = "w0"
    core._uses_versioned_slots = uses_slots
    core._policy_version = policy_version
    core._memory_parking = WorkerMemoryParking(
        "w0",
        GenerationRuntimeLaunchContract(
            family="sd3_5", model_build={}, expected_model_identity={"schema": "test"}
        ),
    )
    core.load_policy = lambda: None  # type: ignore[method-assign]
    return core


def _request(version):
    return SimpleNamespace(policy_version=version, request_id="r", metadata={})


def test_slot_mode_with_live_slot_activates_and_runs() -> None:
    model = _Model({7})
    ex = _Executor(model)
    core = _core(executor=ex, uses_slots=True, policy_version=9)

    out = GenerationWorkerCore.execute_request_pipelined(core, _request(7), "plan", "rows")

    assert out == "GATHERED_OUTPUT"
    assert model.activated == [7]  # served from the REQUEST's version slot, not 9
    assert len(ex.calls) == 1


def test_slot_mode_with_evicted_slot_raises_stale_discard_and_does_not_run() -> None:
    model = _Model(set())  # version 7 evicted
    ex = _Executor(model)
    core = _core(executor=ex, uses_slots=True, policy_version=9)

    with pytest.raises(StaleSlotDiscard):
        GenerationWorkerCore.execute_request_pipelined(core, _request(7), "plan", "rows")
    assert ex.calls == []  # never ran => never trained off-policy


def test_non_slot_version_mismatch_raises_and_does_not_run() -> None:
    ex = _Executor(_Model(set()))
    core = _core(executor=ex, uses_slots=False, policy_version=5)

    with pytest.raises(RuntimeError, match="policy_version mismatch"):
        GenerationWorkerCore.execute_request_pipelined(core, _request(6), "plan", "rows")
    assert ex.calls == []


def test_non_slot_matching_version_runs() -> None:
    ex = _Executor(_Model(set()))
    core = _core(executor=ex, uses_slots=False, policy_version=5)

    out = GenerationWorkerCore.execute_request_pipelined(core, _request(5), "plan", "rows")
    assert out == "GATHERED_OUTPUT"
    assert len(ex.calls) == 1


def test_no_expected_version_runs_unconditionally() -> None:
    ex = _Executor(_Model(set()))
    core = _core(executor=ex, uses_slots=False, policy_version=5)

    out = GenerationWorkerCore.execute_request_pipelined(core, _request(None), "plan", "rows")
    assert out == "GATHERED_OUTPUT"


def test_cuda_oom_clears_worker_state_and_returns_typed_retry(monkeypatch) -> None:
    cleanup_calls: list[dict[str, bool]] = []
    monkeypatch.setattr(
        "vrl.generation.execution.worker.release_cuda_memory",
        lambda **kwargs: cleanup_calls.append(kwargs),
    )
    ex = _Executor(
        _Model(set()),
        error=RuntimeError("CUDA out of memory while pipelining request"),
    )
    core = _core(executor=ex, uses_slots=False, policy_version=5)

    result = GenerationWorkerCore.execute_request_pipelined(
        core,
        _request(5),
        "plan",
        "rows",
    )

    assert result == PipelinedRequestOutOfMemory(
        request_id="r",
        worker_id="w0",
        error="CUDA out of memory while pipelining request",
    )
    assert cleanup_calls == [{}]


def test_non_oom_pipeline_error_propagates_without_cleanup(monkeypatch) -> None:
    cleanup_calls: list[dict[str, bool]] = []
    monkeypatch.setattr(
        "vrl.generation.execution.worker.release_cuda_memory",
        lambda **kwargs: cleanup_calls.append(kwargs),
    )
    ex = _Executor(_Model(set()), error=RuntimeError("scheduler state is invalid"))
    core = _core(executor=ex, uses_slots=False, policy_version=5)

    with pytest.raises(RuntimeError, match="scheduler state is invalid"):
        GenerationWorkerCore.execute_request_pipelined(
            core,
            _request(5),
            "plan",
            "rows",
        )

    assert cleanup_calls == []
