"""Lifecycle FSM: admission gates, terminal idempotency, expected-kill records."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vrl.generation.ray.lifecycle_fsm import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)
from vrl.generation.ray.runtime import RayGenerationRuntime


def _resident_runtime() -> RayGenerationRuntime:
    return RayGenerationRuntime(executor=SimpleNamespace())


def test_transition_table_rejects_invalid_moves() -> None:
    lifecycle = RuntimeLifecycle()
    assert lifecycle.phase is RuntimePhase.STARTING
    with pytest.raises(RuntimeLifecycleError):
        lifecycle.transition(RuntimePhase.STOPPED)  # STARTING cannot skip to STOPPED
    lifecycle.transition(RuntimePhase.RUNNING)
    lifecycle.transition(RuntimePhase.RECOVERING)
    # Single-flight recovery is a property of the table itself.
    with pytest.raises(RuntimeLifecycleError):
        lifecycle.transition(RuntimePhase.RECOVERING)
    lifecycle.transition(RuntimePhase.RUNNING)
    lifecycle.transition(RuntimePhase.DRAINING)
    with pytest.raises(RuntimeLifecycleError):
        lifecycle.transition(RuntimePhase.RUNNING)  # draining never resumes
    lifecycle.transition(RuntimePhase.STOPPED)
    with pytest.raises(RuntimeLifecycleError):
        lifecycle.transition(RuntimePhase.DRAINING)  # terminal


def test_failed_preserves_first_root_cause() -> None:
    lifecycle = RuntimeLifecycle()
    lifecycle.transition(RuntimePhase.RUNNING)
    root = RuntimeError("actor died")
    lifecycle.fail(root)
    lifecycle.fail(RuntimeError("secondary noise"))
    assert lifecycle.failure is root
    with pytest.raises(RuntimeLifecycleError) as caught:
        lifecycle.require_running("generate")
    assert caught.value.__cause__ is root


def test_stopped_runtime_fail_fasts_public_operations() -> None:
    runtime = _resident_runtime()
    asyncio.run(runtime.shutdown())
    assert runtime.lifecycle.phase is RuntimePhase.STOPPED
    # Repeated shutdown is idempotent.
    asyncio.run(runtime.shutdown())
    with pytest.raises(RuntimeLifecycleError, match="generate"):
        asyncio.run(runtime.generate(SimpleNamespace()))
    with pytest.raises(RuntimeLifecycleError, match="update_weights"):
        asyncio.run(runtime.update_weights({}, 1))


def test_shutdown_records_expected_kill_with_operation_id() -> None:
    class _Actor:
        """Fake actor with the two remote methods teardown drives."""

        def __init__(self) -> None:
            self.release_policy = SimpleNamespace(remote=lambda: "ref")

    actor = _Actor()
    worker = SimpleNamespace(actor=actor, worker_id="w0")
    runtime = RayGenerationRuntime(
        executor=SimpleNamespace(), owned_workers=[worker],
    )

    import vrl.generation.ray.runtime as runtime_mod

    class _FakeRayApi:
        @staticmethod
        def get(refs, timeout=None):
            del refs, timeout

        @staticmethod
        def kill(actor, no_restart=True):
            del actor, no_restart

    original = runtime_mod.require_ray
    runtime_mod.require_ray = lambda: _FakeRayApi()
    try:
        asyncio.run(runtime.shutdown())
    finally:
        runtime_mod.require_ray = original

    record = runtime.lifecycle.expected_kill_for(actor)
    assert record is not None
    assert record.reason == "runtime shutdown"
    assert record.operation_id  # a matchable id, not just the phase
    assert record.fleet_generation == 1
    # An actor we never killed does not match any record.
    assert runtime.lifecycle.expected_kill_for(_Actor()) is None


def test_lease_cold_launches_increment_fleet_generation() -> None:
    config = SimpleNamespace(sync_trainable_state=False, gpus_per_worker=0.0, resources=None)
    runtime = RayGenerationRuntime.with_release_after_collect(
        config, SimpleNamespace(), SimpleNamespace(), placement=SimpleNamespace(),
    )
    assert runtime._fleet_generation == 0  # nothing launched yet

    launches: list[int] = []

    class _FakeLauncher:
        def launch(self, *_args, **_kwargs):
            launches.append(1)
            return SimpleNamespace()

    import vrl.generation.ray.launcher as launcher_mod

    original = launcher_mod.RayGenerationLauncher
    launcher_mod.RayGenerationLauncher = _FakeLauncher
    try:
        asyncio.run(runtime._ensure_runtime())
        assert runtime._fleet_generation == 1
        # Teardown release then reacquire -> a NEW fleet generation.
        runtime._release_after_collect.runtime = None
        asyncio.run(runtime._ensure_runtime())
        assert runtime._fleet_generation == 2
    finally:
        launcher_mod.RayGenerationLauncher = original
    assert len(launches) == 2
