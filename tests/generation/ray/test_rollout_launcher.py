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

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test


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


def test_ray_generation_launcher_builds_worker_runtime_with_embedded_ray() -> None:
    """Launcher builds real workers into the owner's placement group."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_rollout_owner(ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                num_workers=1,
                gpus_per_worker=0.0,
                cpus_per_worker=0.5,
                sync_trainable_state="disabled",
                chunk_placement_strategy="dynamic",
            ),
            _launch_contract(),
            _Gatherer(),
            placement=owner.rollout_placement,
        )

        assert isinstance(runtime, RayGenerationRuntime)
        assert runtime.current_policy_version == 7
        assert runtime.weight_sync is None
        # Launcher uses the owner's group; it does not own/remove it.
        assert runtime._placement_group is None
        # Config-selected placement strategy must reach the live planner.
        assert runtime.executor.planner.policy.strategy == "dynamic"

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
        owner.shutdown()
        ray.shutdown()


def _cpu_rollout_owner(ray: Any) -> Any:
    """Build a GlobalRayPlacementOwner with a single CPU rollout bundle."""
    from omegaconf import OmegaConf

    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import resolve_distributed_resources

    resolved = resolve_distributed_resources(
        OmegaConf.create(
            {
                "distributed": {
                    "resources": {
                        "visible_devices": [],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
                    },
                    "rollout": {},
                    "reward": {},
                },
            },
        ),
    )
    owner = GlobalRayPlacementOwner(resolved, rollout_cpus_per_worker=0.5)
    owner.create()
    return owner


def test_owner_placement_runtime_does_not_own_placement_group() -> None:
    """Persistent runtime built on owner placement must not own/remove the PG."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_rollout_owner(ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                num_workers=1,
                gpus_per_worker=0.0,
                cpus_per_worker=0.5,
                sync_trainable_state="disabled",
            ),
            _launch_contract(),
            _Gatherer(),
            placement=owner.rollout_placement,
        )

        # Runtime owns its workers but not the owner-managed placement group.
        assert runtime._placement_group is None
        assert runtime._owned_actors == []
        assert [w.worker_id for w in runtime.executor.workers] == ["rollout-0"]

        # Tearing down the runtime kills workers but leaves the owner's PG alive.
        asyncio.run(runtime.shutdown())
        runtime = None
        assert owner._placement_group is not None
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()


def test_launcher_marks_explicit_colocated_persistent_runtime() -> None:
    """Launcher preserves the colocated bit for single-GPU continuous debug."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_rollout_owner(ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                num_workers=1,
                gpus_per_worker=0.0,
                cpus_per_worker=0.5,
                allow_driver_gpu_overlap=True,
                persistent_colocated_workers=True,
                sync_trainable_state="disabled",
            ),
            _launch_contract(),
            _Gatherer(),
            placement=owner.rollout_placement,
        )

        assert runtime.is_colocated() is True
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()


def test_release_after_collect_owner_placement_survives_release() -> None:
    """Release-after-collect runtime drops workers but never the owner-managed PG."""
    ray = pytest.importorskip("ray")

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_rollout_owner(ray)
    runtime = RayGenerationRuntime.with_release_after_collect(
        RayGenerationConfig(
            num_workers=1,
            gpus_per_worker=0.0,
            cpus_per_worker=0.5,
            sync_trainable_state="disabled",
            release_after_collect=True,
        ),
        _launch_contract(),
        _Gatherer(),
        placement=owner.rollout_placement,
    )
    try:
        # Acquire workers, then release: the underlying runtime is dropped...
        inner = asyncio.run(runtime._ensure_runtime())
        assert inner._placement_group is None  # inner never owned the PG
        asyncio.run(runtime.release())
        assert runtime._release_after_collect is not None
        assert runtime._release_after_collect.runtime is None
        # ...but the owner's placement group is untouched and reacquire works.
        assert owner._placement_group is not None
        reacquired = asyncio.run(runtime._ensure_runtime())
        assert [w.worker_id for w in reacquired.executor.workers] == ["rollout-0"]
    finally:
        asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()
