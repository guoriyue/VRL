"""Ray generation worker resident-session tests.

The worker is the real ``RayGenerationWorker`` built from the tiny SANA run's
own launch inputs, called in-process (no actor): the theorems are about what
``load_policy`` / ``release_policy`` do to its executor, which the real family
loader builds over the tiny snapshot.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.rollouts.collector._helpers import Trace
from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
from vrl import run
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.worker import RayGenerationWorker


def _worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> tuple[RayGenerationWorker, Any]:
    cfg = tiny_sana_online_config(tmp_path)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    launch_inputs = resolved.ray_launch_inputs(replay)
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
