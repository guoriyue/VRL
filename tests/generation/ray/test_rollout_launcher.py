"""Ray generation launcher integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import pytest
except ModuleNotFoundError:  # Ray workers import this module for tiny builders.
    pytest = None

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.config import RayGenerationConfig, RolloutWorkerConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test if pytest is not None else ()


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
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _worker_setup_hook(repo_root: str) -> Any:
    """Return a by-value hook that installs a tiny canonical family in workers."""

    def install() -> None:
        import contextlib
        import sys
        from dataclasses import replace

        # Ray workers do not inherit pytest's cwd-based import path. Put this
        # checkout first so the actor cannot accidentally load another editable
        # VRL checkout from the developer environment.
        sys.path.insert(0, repo_root)

        import vrl.families.registry as registry
        import vrl.models.checkpoint_identity as checkpoint_identity
        from vrl.models.interfaces import RuntimeBundle

        class TinyRuntimeModel:
            device = "cpu"

            def to(self, device: Any) -> TinyRuntimeModel:
                self.device = str(device)
                return self

            def replay_forward(self, batch: Any, timestep_idx: int = 0, **kwargs: Any) -> Any:
                raise NotImplementedError("Ray launcher test never calls replay_forward")

            def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
                return contextlib.nullcontext()

            def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
                self.loaded_state = dict(state_dict)

        class TinyChunkExecutor:
            family = "janus_pro"
            task = "ar_t2i"

            def __init__(self, model: TinyRuntimeModel) -> None:
                self.model = model

            def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Ray launcher test only verifies worker construction",
                )

            def gather_chunks(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Ray launcher test only verifies worker construction",
                )

        def build_tiny_rollout(_entry: Any, build: Any) -> RuntimeBundle:
            assert str(build.device) == "cpu"
            return RuntimeBundle(
                model=TinyRuntimeModel(),
                trainable_modules={},
                scheduler=None,
                raw_handle=None,
                precision=build.precision,
                loads_full_generation_modules=True,
            )

        # The hook executes before worker imports the launch contract. Publishing
        # the test executor on an importable production module lets canonical
        # registry dispatch stay intact without any compatibility fields.
        registry._RayLauncherTestExecutor = TinyChunkExecutor
        entry = registry.FAMILY_REGISTRY["janus_pro"]
        registry.FAMILY_REGISTRY["janus_pro"] = replace(
            entry,
            executor_cls="vrl.families.registry:_RayLauncherTestExecutor",
        )
        registry.ModelFamilyEntry.build_rollout = build_tiny_rollout
        checkpoint_identity.resolve_checkpoint_model_identity = lambda _build: {"schema": "test"}

    return install


def _init_ray(ray: Any) -> None:
    from ray._private import ray_constants

    ray.shutdown()
    repo_root = str(Path(__file__).resolve().parents[3])
    # Same local-cluster contract as the shared conftest fixture: Ray's uv hook
    # would package the whole checkout and re-resolve a project environment
    # without the driver's dev dependencies, so workers could not unpickle the
    # pytest-module-defined setup hook (ModuleNotFoundError: _pytest).
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=2,
        log_to_driver=False,
        runtime_env={"worker_process_setup_hook": _worker_setup_hook(repo_root)},
        _skip_env_hook=True,
    )


def _launch_inputs() -> RayGenerationLaunchInputs:
    return RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family="janus_pro",
            model_build={
                "model_name_or_path": "unit-test",
                "revision": None,
                "device": "cpu",
                "parameter_dtype": "float32",
                "precision": {
                    "dtype": "fp32",
                    "float32_precision": "tf32",
                    "quantization": None,
                    "outer_autocast": False,
                },
            },
            expected_model_identity={"schema": "test"},
            policy_version=7,
        ),
        gatherer=_Gatherer(),
    )


def _worker_config(**overrides: Any) -> RolloutWorkerConfig:
    values = {
        "cpus_per_worker": 0.5,
        "max_inflight_chunks_per_worker": 1,
        "health_check_interval_s": 30.0,
        "health_check_timeout_s": 30.0,
        "health_check_first_wait_s": 0.0,
        "pipelined": False,
        "chunk_placement_strategy": "round_robin",
        "sync_trainable_state": False,
    }
    values.update(overrides)
    return RolloutWorkerConfig(**values)


def test_ray_generation_launcher_builds_worker_runtime_with_embedded_ray() -> None:
    """Launcher builds real workers into the owner's placement group."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    _init_ray(ray)
    worker = _worker_config(chunk_placement_strategy="dynamic")
    owner = _cpu_rollout_owner(ray, worker=worker)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=worker,
            ),
            _launch_inputs(),
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
        metadata = ray.get(workers[0].actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert metadata["policy_version"] == 7
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()


def _cpu_rollout_owner(
    ray: Any,
    *,
    worker: RolloutWorkerConfig | None = None,
) -> Any:
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
    owner = GlobalRayPlacementOwner(resolved, worker or _worker_config())
    owner.create()
    return owner


def test_owner_placement_runtime_does_not_own_placement_group() -> None:
    """Persistent runtime built on owner placement must not own/remove the PG."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    _init_ray(ray)
    owner = _cpu_rollout_owner(ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
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


def test_launcher_uses_resolved_colocation_protocol_signal() -> None:
    """Launcher derives the runtime colocation signal from resolved topology."""
    ray = pytest.importorskip("ray")
    import vrl.generation.ray.launcher as launcher_mod

    _init_ray(ray)
    owner = _cpu_rollout_owner(ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        assert runtime.is_colocated() is False
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()


def test_phase_handoff_keeps_actor_and_owner_placement() -> None:
    """A shared-GPU handoff parks its actor without dropping the owner PG."""
    ray = pytest.importorskip("ray")

    _init_ray(ray)
    owner = _cpu_rollout_owner(ray)
    on_demand_resources = replace(
        owner.resources,
        lifecycle=replace(
            owner.resources.lifecycle,
            rollout=replace(
                owner.resources.lifecycle.rollout,
                mode="on_demand",
            ),
        ),
    )
    runtime = RayGenerationRuntime.with_on_demand_activation(
        RayGenerationConfig(
            resources=on_demand_resources,
            worker=owner.rollout_worker,
        ),
        _launch_inputs(),
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
