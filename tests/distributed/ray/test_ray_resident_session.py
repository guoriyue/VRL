"""Ray rollout worker resident-session tests without a real Ray cluster."""

from __future__ import annotations

from typing import Any

from vrl.distributed.ray.rollout.worker import RayRolloutWorker
from vrl.engine.core.launch_contract import GenerationRuntimeLaunchContract
from vrl.models.ar.capabilities import ar_discrete_family_capability


class _Executor:
    def __init__(self) -> None:
        self.runtime_caps: dict[str, Any] = {}


def test_ray_rollout_worker_load_policy_is_idempotent(monkeypatch: Any) -> None:
    import vrl.distributed.ray.rollout.worker as worker_mod

    capability = ar_discrete_family_capability("janus_pro", "ar_t2i")
    launch_contract = GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        policy_version=1,
        runtime_builder="unused:builder",
        executor_cls="unused:executor",
        extra={"family_capability": capability.to_dict()},
    )
    built: list[_Executor] = []

    def _build_executor(_: RayRolloutWorker) -> _Executor:
        executor = _Executor()
        built.append(executor)
        return executor

    monkeypatch.setattr(worker_mod.RayRolloutWorker, "_build_executor", _build_executor)
    monkeypatch.setattr(
        worker_mod.RayRolloutWorker,
        "_merge_loaded_capability",
        lambda self, executor: self.capability,
    )

    worker = RayRolloutWorker("rollout-0", launch_contract)
    worker.load_policy()
    first_executor = worker.executor
    worker.load_policy()

    assert len(built) == 1
    assert worker.executor is first_executor
