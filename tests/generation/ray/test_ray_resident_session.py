"""Ray generation worker resident-session tests.

The worker is the real ``RayGenerationWorker`` built from the tiny SANA run's
own launch inputs, called in-process (no actor): the theorems are about what
``load_policy`` / ``release_policy`` do to its executor, which the real family
loader builds over the tiny snapshot.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import Trace
from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
from vrl.config.builders import BuiltConfigs
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.resources import ResourcePlan


def _worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> tuple[RayGenerationWorker, Any]:
    cfg = tiny_sana_online_config(tmp_path)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    built = BuiltConfigs.from_cfg(cfg)
    resources = ResourcePlan.from_root(built.root)
    replay_build = built.family.resolve_model_build(
        built.root,
        torch.device(resources.trainer_torch_device),
        precision=built.precision,
        for_rollout=False,
    )
    replay_identity = replay_build.checkpoint_identity()
    launch_inputs = RayGenerationLaunchInputs.from_run(
        built, resources, replay_build, replay_identity
    )
    return RayGenerationWorker("rollout-0", launch_inputs), launch_inputs


def test_ray_generation_worker_load_policy_is_idempotent(monkeypatch, tmp_path) -> None:
    """A second ``load_policy`` is a no-op: the executor is built once."""

    builds = Trace(monkeypatch)
    builds.watch(GenerationWorkerCore, "_build_executor", "build")
    worker, _ = _worker(monkeypatch, tmp_path)

    worker.load_policy()
    first_executor = worker.core.executor
    worker.load_policy()

    assert builds.events == ["build"]
    assert worker.core.executor is first_executor


def test_ray_generation_worker_rebuilds_executor_after_release(monkeypatch, tmp_path) -> None:
    # The resident-session contract: a released worker tears down its executor and
    # rebuilds a fresh one on the next load_policy (not the cached idempotent path).
    builds = Trace(monkeypatch)
    builds.watch(GenerationWorkerCore, "_build_executor", "build")
    worker, _ = _worker(monkeypatch, tmp_path)

    worker.load_policy()
    first_executor = worker.core.executor

    worker.release_policy()
    assert worker.core.executor is None

    worker.load_policy()

    assert builds.events == ["build", "build"]
    assert worker.core.executor is not None
    assert worker.core.executor is not first_executor
