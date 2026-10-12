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
    built: Any
    resources: Any
    worker: Any
    placement_owner: Any
    pipeline: Any

    def request(self, prompts: list[str], *, group_size: int = 2, **kwargs: Any) -> Any:
        """A real generation request for ``prompts``, built by the collector's builder."""

        from vrl.rollouts.collector.config import RolloutCollectorConfig
        from vrl.rollouts.collector.requests import GenerationRequestBuilder

        builder = GenerationRequestBuilder(
            entry=self.built.family, config=RolloutCollectorConfig.from_root(self.built.root)
        )
        return builder.build(prompts, group_size, **kwargs).request

    def trainable_state(self) -> dict[str, Any]:
        """The trainer's real weight-sync payload for this run's policy."""

        import torch

        from vrl import run
        from vrl.trainers.strategy import SingleProcessStrategy

        replay = run.ResolvedModel.from_root(
            self.built.family,
            self.built.root,
            torch.device(self.resources.trainer_torch_device),
            precision=self.built.precision,
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

    import torch

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl import run
    from vrl.config.builders import BuiltConfigs
    from vrl.generation.ray.config import RolloutWorkerConfig
    from vrl.generation.ray.launch_inputs import build_launch_inputs
    from vrl.generation.ray.launcher import RayGenerationLauncher
    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import ResolvedDistributedResources

    cfg = tiny_sana_online_config(
        tmp_path,
        snapshot=snapshot,
        overrides=("distributed.rollout.cpus_per_worker=0.5", *overrides),
    )
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    built = BuiltConfigs.from_cfg(cfg)
    resources = ResolvedDistributedResources.from_root(built.root)
    worker = RolloutWorkerConfig.from_root(built.root)
    replay = run.ResolvedModel.from_root(
        built.family,
        built.root,
        torch.device(resources.trainer_torch_device),
        precision=built.precision,
        for_rollout=False,
    )
    owner = GlobalRayPlacementOwner(resources, worker)
    owner.create()
    try:
        runtime = RayGenerationLauncher().create_runtime(
            resources,
            worker,
            build_launch_inputs(built, resources, replay),
            placement=owner.rollout_placement,
        )
        try:
            yield RaySanaRuntime(
                runtime=runtime,
                built=built,
                resources=resources,
                worker=worker,
                placement_owner=owner,
                pipeline=pipeline,
            )
        finally:
            # A test that already drove the runtime to TERMINATED has nothing
            # left to release; otherwise a failing shutdown is a real failure.
            if runtime.lifecycle.phase is not RuntimePhase.TERMINATED:
                await runtime.shutdown()
    finally:
        owner.shutdown()
