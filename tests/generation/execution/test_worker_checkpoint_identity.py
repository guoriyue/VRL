"""Worker-side rollout checkpoint identity gates."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

import vrl.families.registry as registry
import vrl.models.checkpoint_identity as checkpoint_identity
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract


class _RuntimeModel:
    device = "cpu"

    def replay_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        self.loaded_state = dict(state_dict)


class _ChunkExecutor:
    family = "unit"
    task = "t2i"

    def __init__(self, model: _RuntimeModel) -> None:
        self.model = model

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def gather_chunks(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _FamilyEntry:
    family = "unit"
    task = "t2i"
    executor_cls = "tests.generation.execution.test_worker_checkpoint_identity:_ChunkExecutor"

    from vrl.families.registry import GenerationRuntimeCapabilities as _Caps

    runtime_capabilities = _Caps()

    def __init__(self) -> None:
        self.build_calls = 0

    def build_rollout(self, build: Any) -> Any:
        self.build_calls += 1
        return SimpleNamespace(model=_RuntimeModel())


def _contract(expected_model_identity: dict[str, Any]) -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="unit",
        model_build={
            "model_name_or_path": "unit-test",
            "revision": None,
            "device": "cpu",
            "parameter_dtype": "float32",
            "precision": {
                "dtype": "fp32",
                "float32_precision": "ieee",
                "quantization": None,
                "outer_autocast": False,
            },
            "model_config": {},
            "sampling_config": {},
            "rollout": {
                "prompt_encoder_dtype": "float32",
                "base_weight_sync": False,
            },
        },
        expected_model_identity=expected_model_identity,
    )


def _worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entry: _FamilyEntry,
    identities: list[dict[str, Any]],
) -> GenerationWorkerCore:
    identity_iter = iter(identities)
    monkeypatch.setattr(registry, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda build: next(identity_iter),
    )
    return GenerationWorkerCore("rollout-0", _contract(identities[0]))


def test_worker_accepts_matching_identity_before_and_after_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"schema": "test", "sources": {"base": "same"}}
    entry = _FamilyEntry()
    worker = _worker(
        monkeypatch,
        entry=entry,
        identities=[identity, identity],
    )

    worker.load_policy()

    assert isinstance(worker.executor, _ChunkExecutor)
    assert entry.build_calls == 1


def test_worker_rejects_cross_node_identity_mismatch_before_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver_identity = {"schema": "test", "sources": {"base": "driver"}}
    worker_identity = {"schema": "test", "sources": {"base": "worker"}}
    entry = _FamilyEntry()
    worker = _worker(
        monkeypatch,
        entry=entry,
        identities=[driver_identity],
    )
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda _build: worker_identity,
    )

    with pytest.raises(
        ValueError,
        match="model identity mismatch before model construction",
    ):
        worker.load_policy()

    assert entry.build_calls == 0
    assert worker.executor is None


def test_worker_rejects_source_change_during_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"schema": "test", "sources": {"base": "before"}}
    after = {"schema": "test", "sources": {"base": "after"}}
    entry = _FamilyEntry()
    worker = _worker(
        monkeypatch,
        entry=entry,
        identities=[before, after],
    )

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during bundle construction",
    ):
        worker.load_policy()

    assert entry.build_calls == 1
    assert worker.executor is None
