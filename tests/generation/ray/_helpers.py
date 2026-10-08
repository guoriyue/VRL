"""Shared helpers for the Ray generation tests.

``ray_sana_runtime`` is the construction these tests start from: a real
``RayGenerationRuntime`` launched by the real launcher into a real placement
group on the package's local cluster, its ``RayGenerationWorker`` actors
serving the tiny SANA snapshot (see ``conftest.py``).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vrl.generation.ray.worker import RayGenerationWorker
from vrl.utils.lifecycle import RuntimePhase


class SlowGenerationWorker(RayGenerationWorker):
    """The real worker; a ``"slow"`` prompt holds the engine for ``HOLD_S`` first.

    A fixed hold inside the actor gives tests that race a second call against
    an in-flight generation a lower bound that does not depend on how fast the
    host finishes a tiny-SANA request.
    """

    HOLD_S = 1.0

    def execute_batch(self, envelope: Any) -> Any:
        if envelope.request.prompts == ["slow"]:
            time.sleep(self.HOLD_S)
        return super().execute_batch(envelope)


def install_slow_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next ``ray_sana_runtime`` launch ``SlowGenerationWorker`` actors."""

    from vrl.generation.ray import launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationWorker", SlowGenerationWorker)


@dataclass
class RaySanaRuntime:
    """A launched real Ray generation runtime and what it was launched from."""

    runtime: Any
    resolved: Any
    placement_owner: Any
    pipeline: Any

    def request(self, prompts: list[str], *, group_size: int = 2, **kwargs: Any) -> Any:
        """A real generation request for ``prompts``, built by the collector's builder."""

        from vrl.rollouts.collector.requests import GenerationRequestBuilder

        builder = GenerationRequestBuilder(
            entry=self.resolved.family, config=self.resolved.collector
        )
        return builder.build(prompts, group_size, **kwargs).request

    def trainable_state(self) -> dict[str, Any]:
        """The trainer's real weight-sync payload for this run's policy."""

        from vrl import run
        from vrl.trainers.strategy import SingleProcessStrategy

        replay = run.resolve_model(
            self.resolved.family,
            self.resolved.built.root,
            self.resolved.device,
            precision=self.resolved.built.precision,
            for_rollout=False,
        )
        bundle = replay.materialize(context="ray test weight payload")
        return SingleProcessStrategy().export_rollout_state(bundle)


@contextlib.asynccontextmanager
async def ray_sana_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: Path,
    *,
    overrides: tuple[str, ...] = (),
) -> AsyncIterator[RaySanaRuntime]:
    """Launch a real Ray generation runtime over ``snapshot`` and release it after.

    Uses the online recipe's own path: resolve the tiny SANA run, create the
    run-level placement group, launch with ``RayGenerationLauncher``. On exit
    the runtime is shut down (its actors killed) and the placement released.
    """

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl import run
    from vrl.generation.ray.launcher import RayGenerationLauncher
    from vrl.ray.placement import GlobalRayPlacementOwner

    cfg = tiny_sana_online_config(
        tmp_path,
        snapshot=snapshot,
        overrides=("distributed.rollout.cpus_per_worker=0.5", *overrides),
    )
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    owner = GlobalRayPlacementOwner(resolved.resources, resolved.generation.worker)
    owner.create()
    try:
        runtime = RayGenerationLauncher().create_runtime(
            resolved.generation,
            resolved.ray_launch_inputs(replay),
            placement=owner.rollout_placement,
        )
        try:
            yield RaySanaRuntime(
                runtime=runtime, resolved=resolved, placement_owner=owner, pipeline=pipeline
            )
        finally:
            # A test that already drove the runtime to TERMINATED has nothing
            # left to release; otherwise a failing shutdown is a real failure.
            if runtime.lifecycle.phase is not RuntimePhase.TERMINATED:
                await runtime.shutdown()
    finally:
        owner.shutdown()
