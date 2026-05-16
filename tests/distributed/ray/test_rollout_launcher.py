"""Small Ray rollout launcher tests that avoid starting a real Ray cluster."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace
from typing import Any

from vrl.distributed.ray.rollout.runtime import RayDistributedRuntime
from vrl.engine.core.launch_contract import GenerationRuntimeLaunchContract
from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import GenerationRequest, GenerationSampleRow, OutputBatch
from vrl.rollouts.runtime.config import RolloutBackendConfig


class _PlacementGroupSchedulingStrategy:
    def __init__(
        self,
        *,
        placement_group: Any,
        placement_group_capture_child_tasks: bool,
        placement_group_bundle_index: int,
    ) -> None:
        self.placement_group = placement_group
        self.placement_group_capture_child_tasks = placement_group_capture_child_tasks
        self.placement_group_bundle_index = placement_group_bundle_index


class _FakeRemoteMethod:
    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.calls = 0

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._fn(*args, **kwargs)


class _FakeActorHandle:
    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract,
        strategy: _PlacementGroupSchedulingStrategy,
    ) -> None:
        self.worker_id = worker_id
        self.launch_contract = launch_contract
        self.strategy = strategy
        self.load_policy = _FakeRemoteMethod(lambda: None)
        self.worker_metadata = _FakeRemoteMethod(self._metadata)

    def _metadata(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "node_ip": f"node-for-{self.worker_id}",
            "gpu_ids": (),
        }


class _FakeRemoteClass:
    def __init__(self, ray: _FakeRay, resource_options: dict[str, Any]) -> None:
        self._ray = ray
        self._resource_options = resource_options
        self._strategy: _PlacementGroupSchedulingStrategy | None = None

    def options(
        self,
        *,
        scheduling_strategy: _PlacementGroupSchedulingStrategy,
    ) -> _FakeRemoteClass:
        child = _FakeRemoteClass(self._ray, self._resource_options)
        child._strategy = scheduling_strategy
        return child

    def remote(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract,
    ) -> _FakeActorHandle:
        if self._strategy is None:
            raise AssertionError("launcher must set a scheduling strategy for rollout actors")
        actor = _FakeActorHandle(worker_id, launch_contract, self._strategy)
        self._ray.actors.append(actor)
        return actor


class _FakeRay:
    def __init__(self) -> None:
        self.initialized = False
        self.init_kwargs: dict[str, Any] | None = None
        self.remote_options: list[dict[str, Any]] = []
        self.actors: list[_FakeActorHandle] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **kwargs: Any) -> None:
        self.initialized = True
        self.init_kwargs = dict(kwargs)

    def remote(self, **resource_options: Any) -> Any:
        self.remote_options.append(dict(resource_options))

        def _wrap(_actor_cls: type[Any]) -> _FakeRemoteClass:
            return _FakeRemoteClass(self, resource_options)

        return _wrap

    def get(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.get(item) for item in value]
        return value


class _Gatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _install_fake_ray_modules(monkeypatch: Any) -> None:
    ray_module = ModuleType("ray")
    ray_util_module = ModuleType("ray.util")
    scheduling_module = ModuleType("ray.util.scheduling_strategies")
    scheduling_module.PlacementGroupSchedulingStrategy = _PlacementGroupSchedulingStrategy
    ray_module.util = ray_util_module
    ray_util_module.scheduling_strategies = scheduling_module
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling_module)


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="fake",
        task="t2i",
        policy_version=7,
        runtime_builder="tests.distributed.ray.test_rollout_launcher:_unused_builder",
        executor_cls="tests.distributed.ray.test_rollout_launcher:_UnusedExecutor",
    )


def test_ray_rollout_launcher_builds_worker_runtime_without_real_ray(monkeypatch) -> None:
    import vrl.distributed.ray.rollout.launcher as launcher_mod

    _install_fake_ray_modules(monkeypatch)
    fake_ray = _FakeRay()
    placement_group = object()
    placement = SimpleNamespace(
        placement_group=placement_group,
        ordered_bundle_indices=[3, 1],
        trainer_reservation_actors=["trainer-reservation"],
        trainer_gpu_ids=(),
        rollout_gpu_ids=(),
    )
    monkeypatch.setattr(launcher_mod, "require_ray", lambda: fake_ray)
    monkeypatch.setattr(
        launcher_mod,
        "create_rollout_placement_group",
        lambda config: placement,
    )

    runtime = launcher_mod.RayRolloutLauncher(
        init_ray=True,
        ray_init_kwargs={"ignore_reinit_error": True},
    ).launch(
        RolloutBackendConfig(
            backend="ray",
            num_workers=2,
            gpus_per_worker=0.0,
            cpus_per_worker=2.0,
            sync_trainable_state="disabled",
        ),
        _launch_contract(),
        _Gatherer(),
    )

    assert isinstance(runtime, RayDistributedRuntime)
    assert fake_ray.init_kwargs == {"ignore_reinit_error": True}
    assert fake_ray.remote_options == [{"num_cpus": 2.0, "num_gpus": 0.0}]
    assert runtime.current_policy_version == 7
    assert runtime.weight_sync is None
    assert runtime._owned_actors == ["trainer-reservation"]
    assert runtime._placement_group is placement_group

    workers = runtime.executor.workers
    assert [worker.worker_id for worker in workers] == ["rollout-0", "rollout-1"]
    assert [worker.node_id for worker in workers] == [
        "node-for-rollout-0",
        "node-for-rollout-1",
    ]
    assert [actor.launch_contract for actor in fake_ray.actors] == [
        _launch_contract(),
        _launch_contract(),
    ]
    assert [actor.load_policy.calls for actor in fake_ray.actors] == [1, 1]
    assert [actor.worker_metadata.calls for actor in fake_ray.actors] == [1, 1]
    assert [actor.strategy.placement_group for actor in fake_ray.actors] == [
        placement_group,
        placement_group,
    ]
    assert [actor.strategy.placement_group_bundle_index for actor in fake_ray.actors] == [
        3,
        1,
    ]
    assert all(actor.strategy.placement_group_capture_child_tasks for actor in fake_ray.actors)
