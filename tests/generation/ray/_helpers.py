"""Shared helpers for the Ray generation tests.

``ray_sana_runtime`` is the construction these tests start from: a real
``RayGenerationRuntime`` launched by the real launcher into a real placement
group on the package's local cluster, its ``RayGenerationWorker`` actors
serving the tiny SANA snapshot (see ``conftest.py``). ``run_launch_inputs`` is
the driver-side half of that launch for tests that resolve a run themselves.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from vrl.generation.ray.worker import RayGenerationWorker
from vrl.utils.lifecycle import RuntimePhase


def resolve_replay_build(built: Any, resources: Any) -> Any:
    """The driver's replay-model build for a resolved run, as the online recipe resolves it."""

    import torch

    return built.family.resolve_model_build(
        built.root,
        torch.device(resources.trainer_torch_device),
        precision=built.precision,
        for_rollout=False,
    )


def run_launch_inputs(built: Any, resources: Any) -> Any:
    """The launch inputs the online recipe ships to its rollout workers for a resolved run."""

    from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs

    replay_build = resolve_replay_build(built, resources)
    return RayGenerationLaunchInputs.from_run(
        built, resources, replay_build, replay_build.checkpoint_identity()
    )


def is_dead(ray: Any, actor: Any, *, settle_s: float = 10.0) -> bool:
    """Whether the actor stops answering; ``ray.kill`` takes effect asynchronously."""

    deadline = time.monotonic() + settle_s
    while True:
        try:
            ray.get(actor.worker_metadata.remote(), timeout=30)
        except ray.exceptions.RayActorError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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
        """A fresh copy of the trainer's real weight-sync payload for this run's policy."""

        from vrl.trainers.strategy import SingleProcessStrategy

        replay_build = resolve_replay_build(self.built, self.resources)
        bundle = self.built.family.build_replay(
            replay_build, expected_identity=replay_build.checkpoint_identity()
        )
        return SingleProcessStrategy().export_rollout_state(bundle)

    def actors(self) -> list[Any]:
        """The rank actors of the runtime's current session."""

        session = self.runtime._session
        assert session is not None
        return [rank.actor for rank in session.rank_handles]


@contextlib.asynccontextmanager
async def ray_sana_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: Path,
    *,
    overrides: tuple[str, ...] = (),
    on_demand: bool = False,
) -> AsyncIterator[RaySanaRuntime]:
    """Launch a real Ray generation runtime over ``snapshot`` and release it after.

    Uses the online recipe's own path: resolve the tiny SANA run, create the
    run-level placement group, launch with ``RayGenerationLauncher``. On exit
    the runtime is shut down (its actors killed) and the placement released.

    ``on_demand`` pins the trainer and the rollout to one card in the plan the
    launcher receives -- a CPU host cannot resolve that topology itself -- so
    ``create_runtime`` takes its on-demand branch: no session until
    ``activate``, the launcher's own async launch as the cold-start factory.
    The launch config is still the resolved run's, so CPU workers park through
    their real move ledger.
    """

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl.config.builders import BuiltConfigs
    from vrl.generation.ray.config import RolloutWorkerConfig
    from vrl.generation.ray.launcher import RayGenerationLauncher
    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import ResourcePlan

    cfg = tiny_sana_online_config(
        tmp_path,
        snapshot=snapshot,
        overrides=("distributed.rollout.cpus_per_worker=0.5", *overrides),
    )
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    built = BuiltConfigs.from_cfg(cfg)
    resources = ResourcePlan.from_root(built.root)
    worker = RolloutWorkerConfig.from_root(built.root)
    launch_plan = resources
    if on_demand:
        launch_plan = replace(
            resources,
            lifecycle=replace(resources.lifecycle, trainer=(0,), rollout=(0,)),
        )
    owner = GlobalRayPlacementOwner(resources, worker)
    owner.create()
    try:
        runtime = RayGenerationLauncher().create_runtime(
            launch_plan,
            worker,
            run_launch_inputs(built, resources),
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
