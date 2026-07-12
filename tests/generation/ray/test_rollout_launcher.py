"""Ray generation launcher integration tests."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

try:
    import pytest
except ModuleNotFoundError:  # Ray workers import this module for tiny builders.
    pytest = None

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.interfaces import ModelBuild, ReplayResult, RuntimeBundle

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test if pytest is not None else ()


class _TinyRuntimeModel:
    device = "cpu"

    def to(self, device: Any) -> _TinyRuntimeModel:
        self.device = str(device)
        return self

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


def build_tiny_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    assert str(build.device) == "cpu"
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
        policy_version=7,
        model_build={
            "model_name_or_path": "unit-test",
            "device": "cpu",
            "parameter_dtype": "float32",
        },
        runtime_builder=("tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"),
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
                sync_trainable_state=False,
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
        assert not hasattr(runtime, "_placement_group")
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
                sync_trainable_state=False,
            ),
            _launch_contract(),
            _Gatherer(),
            placement=owner.rollout_placement,
        )

        # Runtime owns its workers but not the owner-managed placement group.
        assert not hasattr(runtime, "_placement_group")
        assert not hasattr(runtime, "_owned_actors")
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


def test_launcher_preserves_explicit_colocation_protocol_signal() -> None:
    """Direct runtime construction preserves the is_colocated safety signal."""
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
                sync_trainable_state=False,
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


def test_phase_handoff_keeps_actor_and_owner_placement() -> None:
    """A shared-GPU handoff parks its actor without dropping the owner PG."""
    ray = pytest.importorskip("ray")

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_rollout_owner(ray)
    runtime = RayGenerationRuntime.with_on_demand_activation(
        RayGenerationConfig(
            num_workers=1,
            gpus_per_worker=0.0,
            cpus_per_worker=0.5,
            sync_trainable_state=False,
            resources=SimpleNamespace(
                rollout_gpus_per_worker=0.0,
                lifecycle=SimpleNamespace(
                    rollout=SimpleNamespace(mode="on_demand"),
                ),
            ),
        ),
        RayGenerationLaunchInputs(
            launch_contract=_launch_contract(),
            gatherer=_Gatherer(),
        ),
        placement=owner.rollout_placement,
    )
    try:
        # Explicit activation launches workers; offload parks them in place.
        asyncio.run(runtime.activate())
        assert runtime._on_demand is not None
        inner = runtime._on_demand.inner_runtime
        assert inner is not None
        assert not hasattr(inner, "_placement_group")  # inner never owns the PG
        first_actor = inner.executor.workers[0].actor
        asyncio.run(runtime.offload())
        assert runtime._on_demand is not None
        assert runtime._on_demand.inner_runtime is inner
        assert runtime._on_demand.workers_offloaded is True
        # The owner's placement group is untouched and activation wakes in place.
        assert owner._placement_group is not None
        asyncio.run(runtime.activate())
        reacquired = runtime._on_demand.inner_runtime
        assert reacquired is not None
        assert [w.worker_id for w in reacquired.executor.workers] == ["rollout-0"]
        assert reacquired is inner
        assert reacquired.executor.workers[0].actor is first_actor
    finally:
        asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()
