"""Ray generation launcher integration tests."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.interfaces import ReplayResult, RuntimeBuildSpec, RuntimeBundle


class _TinyRuntimeModel:
    device = "cpu"

    def replay_forward(self, batch: Any, timestep_idx: int = 0, **kwargs: Any) -> ReplayResult:
        raise NotImplementedError("Ray launcher test never calls replay_forward")

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        self.loaded_state = dict(state_dict)


class _TinyChunkExecutor:
    family = "janus_pro"
    task = "ar_t2i"

    def __init__(self, model: _TinyRuntimeModel) -> None:
        self.model = model

    def workload_signature(self, request: GenerationRequest) -> Any:
        del request
        return None

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> ChunkResult:
        raise NotImplementedError("Ray launcher test only verifies worker construction")

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


class _Gatherer:
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
        backend_kind="test",
        backend_handle=None,
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    capability = ar_discrete_family_capability("janus_pro", "ar_t2i")
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        policy_version=7,
        model_build={
            "model_name_or_path": "unit-test",
            "device": "cpu",
            "dtype": "float32",
        },
        runtime_builder=(
            "tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"
        ),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
        extra={"family_capability": capability.to_dict()},
    )


def test_ray_generation_launcher_builds_worker_runtime_with_local_ray() -> None:
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    ray.shutdown()
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(
            init_ray=True,
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 2,
                "log_to_driver": False,
            },
        ).launch(
            RayGenerationConfig(
                num_workers=1,
                gpus_per_worker=0.0,
                cpus_per_worker=0.5,
                sync_trainable_state="disabled",
            ),
            _launch_contract(),
            _Gatherer(),
        )

        assert isinstance(runtime, RayGenerationRuntime)
        assert runtime.current_policy_version == 7
        assert runtime.weight_sync is None
        assert runtime._placement_group is not None

        workers = runtime.executor.workers
        assert [worker.worker_id for worker in workers] == ["rollout-0"]
        assert workers[0].actor is not None
        assert ray.get(workers[0].actor.current_policy_version.remote()) == 7
        metadata = ray.get(workers[0].actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert metadata["policy_version"] == 7
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()
