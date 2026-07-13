"""Ray generation worker resident-session tests."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import (
    ModelBuild,
    ReplayResult,
    RolloutBuildOptions,
    RuntimeBundle,
)


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


def build_tiny_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    assert str(build.device) == "cpu"
    assert build.parameter_dtype is torch.float16
    assert isinstance(build.rollout, RolloutBuildOptions)
    assert build.rollout.autocast_dtype is torch.bfloat16
    assert build.rollout.prompt_encoder_dtype is torch.float32
    return RuntimeBundle(
        model=_TinyRuntimeModel(),
        trainable_modules={},
        scheduler=None,
        raw_handle=None,
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        generation_kind="ar",
        policy_version=1,
        model_build={
            "model_name_or_path": "unit-test",
            "device": "cpu",
            "parameter_dtype": "float16",
            "rollout": {
                "autocast_dtype": "bfloat16",
                "prompt_encoder_dtype": "float32",
                "quantization_format": None,
                "quantization_recipe": None,
                "base_weight_sync": False,
            },
        },
        runtime_builder=(
            "tests.generation.ray.test_ray_resident_session:build_tiny_runtime_bundle"
        ),
        executor_cls="tests.generation.ray.test_ray_resident_session:_TinyChunkExecutor",
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
