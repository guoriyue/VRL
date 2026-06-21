"""Ray generation worker resident-session tests."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.interfaces import ReplayResult, RuntimeBuildSpec, RuntimeBundle


class _TinyRuntimeModel:
    device = "cpu"

    def replay_forward(self, batch: Any, timestep_idx: int = 0, **kwargs: Any) -> ReplayResult:
        raise NotImplementedError("Ray worker idempotency test never replays")

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        self.loaded_state = dict(state_dict)


class _TinyChunkExecutor:
    build_count = 0
    family = "janus_pro"
    task = "ar_t2i"

    def __init__(self, model: _TinyRuntimeModel) -> None:
        type(self).build_count += 1
        self.model = model

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> ChunkResult:
        raise NotImplementedError("Ray worker idempotency test never executes chunks")

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def build_tiny_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    assert str(spec.device) == "cpu"
    return RuntimeBundle(
        model=_TinyRuntimeModel(),
        trainable_modules={},
        scheduler=None,
        raw_handle=None,
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    capability = ar_discrete_family_capability("janus_pro", "ar_t2i")
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        policy_version=1,
        model_build={
            "model_name_or_path": "unit-test",
            "device": "cpu",
            "dtype": "float32",
        },
        runtime_builder=(
            "tests.generation.ray.test_ray_resident_session:build_tiny_runtime_bundle"
        ),
        executor_cls="tests.generation.ray.test_ray_resident_session:_TinyChunkExecutor",
        extra={"family_capability": capability.to_dict()},
    )


def test_ray_generation_worker_load_policy_is_idempotent() -> None:
    """Checks Ray generation worker load policy is idempotent."""
    _TinyChunkExecutor.build_count = 0
    worker = RayGenerationWorker("rollout-0", _launch_contract())

    worker.load_policy()
    first_executor = worker.executor
    worker.load_policy()

    assert _TinyChunkExecutor.build_count == 1
    assert worker.executor is first_executor


def test_ray_generation_worker_rebuilds_executor_after_release() -> None:
    # The resident-session contract: a released worker tears down its executor and
    # rebuilds a fresh one on the next load_policy (not the cached idempotent path).
    """Checks Ray generation worker rebuilds executor after release."""
    _TinyChunkExecutor.build_count = 0
    worker = RayGenerationWorker("rollout-0", _launch_contract())

    worker.load_policy()
    first_executor = worker.executor

    worker.release_policy()
    assert worker.executor is None

    worker.load_policy()

    assert _TinyChunkExecutor.build_count == 2
    assert worker.executor is not None
    assert worker.executor is not first_executor
