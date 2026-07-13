"""Versioned trainable-state slots in GenerationWorkerCore (non-draining sync).

These exercise the worker's slot branching directly: a model that supports
versioned slots makes update_weights INSTALL (not overwrite) and makes
execute_chunk serve each request from the slot for its OWN stamped version, so an
in-flight request generated under an older policy keeps working after a newer
weight sync (the prerequisite for skipping the drain bubble). A model without the
capability keeps the single in-place overwrite + global-version check.

The executor is injected directly so load_policy() short-circuits — no model build.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.types import ChunkExecutionEnvelope
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.types import GenerationRequest


class _SlotModel:
    """RuntimeModel that supports versioned slots and records protocol calls."""

    device = "cpu"
    supports_versioned_trainable_state = True

    def __init__(self) -> None:
        self.slots: dict[int, Any] = {}
        self.activated: list[int] = []
        self.load_calls: list[Any] = []

    # RuntimeModel contract (require_runtime_model)
    def replay_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: Any) -> None:
        self.load_calls.append(state_dict)

    # versioned-slot protocol
    def install_trainable_state(self, version: int, state_dict: Any) -> None:
        self.slots[int(version)] = state_dict

    def has_trainable_state(self, version: int) -> bool:
        return int(version) in self.slots

    def activate_trainable_state(self, version: int) -> None:
        self.activated.append(int(version))


class _PlainModel:
    """RuntimeModel without versioned-slot support (draining-barrier path)."""

    device = "cpu"

    def __init__(self) -> None:
        self.load_calls: list[Any] = []

    def replay_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: Any) -> None:
        self.load_calls.append(state_dict)


class _Executor:
    family = "sd3_5"
    task = "t2i"

    def __init__(self, model: Any) -> None:
        self.model = model

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"noise_pred": torch.zeros(1)}

    def gather_chunks(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _unused_builder(spec: Any) -> Any:  # pragma: no cover - executor is injected
    raise RuntimeError("builder must not run; the test injects core.executor directly")


_MODULE = "tests.generation.execution.test_worker_versioned_slots"


def _core(model: Any) -> GenerationWorkerCore:
    # runtime_builder/executor_cls are validated as non-None import-path strings at
    # construction but never imported here — _build_executor is skipped because we
    # inject core.executor below (load_policy() then short-circuits).
    contract = GenerationRuntimeLaunchContract(
        family="sd3_5",
        task="t2i",
        generation_kind="diffusion",
        policy_version=1,
        runtime_builder=f"{_MODULE}:_unused_builder",
        executor_cls=f"{_MODULE}:_Executor",
    )
    core = GenerationWorkerCore("rollout-0", contract)
    core.executor = _Executor(model)  # bypass load_policy() build
    return core


def _envelope(version: int) -> ChunkExecutionEnvelope:
    request = GenerationRequest(
        request_id=f"req-{version}",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        policy_version=version,
    )
    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=1)
    return ChunkExecutionEnvelope(
        request=request,
        chunk=chunk,
    )


# -- update_weights -----------------------------------------------------------


def test_update_weights_installs_versioned_slots_without_overwrite() -> None:
    model = _SlotModel()
    core = _core(model)

    assert core.update_weights({"transformer.w": "v1"}, 1) == 1
    assert core.update_weights({"transformer.w": "v2"}, 2) == 2

    assert core._uses_versioned_slots is True
    assert model.has_trainable_state(1) and model.has_trainable_state(2)
    # Slot mode installs; it must NOT overwrite the live model in place.
    assert model.load_calls == []
    # current_policy_version tracks the latest install (what the producer stamps onto NEW requests).
    assert core.current_policy_version() == 2


def test_update_weights_plain_model_loads_in_place() -> None:
    model = _PlainModel()
    core = _core(model)

    assert core.update_weights({"transformer.w": "v1"}, 1) == 1

    assert core._uses_versioned_slots is False
    assert model.load_calls == [{"transformer.w": "v1"}]
    assert core.current_policy_version() == 1


# -- execute_chunk ------------------------------------------------------------


def test_execute_chunk_missing_slot_returns_typed_stale_slot() -> None:
    model = _SlotModel()
    core = _core(model)
    core.update_weights({"transformer.w": "v1"}, 1)  # only slot 1 exists

    result = core.execute_chunk(_envelope(2))

    assert result.stale_slot is True
    assert result.output is None
    assert result.policy_version == 2
    assert "slot evicted" in (result.error or "")
    assert model.activated == []  # never forwarded


def test_execute_chunk_activates_request_version_slot() -> None:
    model = _SlotModel()
    core = _core(model)
    core.update_weights({"transformer.w": "v1"}, 1)
    core.update_weights({"transformer.w": "v2"}, 2)

    # An OLD v1 request still runs after v2 was installed.
    result = core.execute_chunk(_envelope(1))

    assert result.error is None
    assert result.stale_slot is False
    # Result carries the REQUEST's version so the executor's version assert passes.
    assert result.policy_version == 1
    assert model.activated == [1]


def test_plain_model_keeps_global_version_mismatch() -> None:
    model = _PlainModel()
    core = _core(model)
    core.update_weights({"transformer.w": "v1"}, 1)

    # Non-slot model: a request for a different version is the classic mismatch.
    result = core.execute_chunk(_envelope(2))

    assert result.stale_slot is False
    assert "policy_version mismatch" in (result.error or "")
    assert result.policy_version == 1
