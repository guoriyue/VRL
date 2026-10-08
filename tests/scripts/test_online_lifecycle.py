"""Shutdown order and error propagation of ``run_online_recipe``, on a real run.

Every role is real: the tiny SANA online config (``tiny_sana_online_config``,
a CPU sharpness reward), the real family loader over ``TinySanaPipeline`` (the
Hub load is the one model double, installed in the Ray worker processes too),
a real local Ray cluster, the real placement owner, rollout launcher and Ray
generation worker, the real reward runtime, collector, trainer, rollout
schedule and checkpoint writer.

The theorems are *which* role the recipe releases, in what order, and which
error survives. A ``Trace`` wraps the real role methods to record that order
and to inject the one-shot failures; the wrapped methods still run.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.conftest import real_local_ray
from tests.rollouts.collector._helpers import Trace
from tests.scripts.eval.fixtures import (
    TinySanaPipeline,
    tiny_sana_online_config,
    tiny_sana_ray_cluster,
    tiny_sana_ray_runtime_env,
    write_tiny_sana_snapshot,
)
from vrl import run as resolved_run
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.run import ResolvedReward
from vrl.scripts.common import online
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def ray_snapshot(tmp_path_factory) -> Path:
    """The one tiny snapshot the module's Ray workers serve.

    A Ray worker installs ``TinySanaPipeline`` for a fixed path at process
    start, so every run that reaches Ray in this module loads this directory.
    """

    return write_tiny_sana_snapshot(tmp_path_factory.mktemp("ray") / "sana-snapshot")


@pytest.fixture()
def preinitialized_ray(ray_snapshot):
    """A real local Ray cluster whose workers serve ``ray_snapshot``.

    The recipe sees an is_initialized() driver, takes the embedding-caller
    ownership branch, and must leave the cluster running. One cluster per test:
    the owned-cluster test shuts Ray down, so a module-shared cluster would not
    survive it.
    """

    with tiny_sana_ray_cluster(ray_snapshot) as started:
        yield started


class _RealRun:
    """One real tiny SANA run: config, snapshot and the pipeline the loader serves."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        snapshot: Path | None = None,
        prompts: tuple[str, ...] = ("a cat",),
        overrides: tuple[str, ...] = (),
        on_load: Any = None,
    ) -> None:
        self.cfg = tiny_sana_online_config(
            tmp_path,
            prompts=prompts,
            snapshot=snapshot,
            overrides=(
                "trainer.save_freq=0",
                "distributed.rollout.cpus_per_worker=0.5",
                *overrides,
            ),
        )
        self.snapshot = snapshot or tmp_path / "sana-snapshot"
        self.output_dir = tmp_path / "run"
        self.pipeline = TinySanaPipeline()
        self.pipeline.install(monkeypatch, self.snapshot, on_load=on_load)

    def evidence(self) -> list[dict[str, Any]]:
        directory = self.output_dir / "run_evidence"
        return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _roles(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """Record (and let tests fail) the real Ray-side role transitions."""

    roles = Trace(monkeypatch)
    roles.watch(online.GlobalRayPlacementOwner, "create", "owner.create")
    roles.watch(online.GlobalRayPlacementOwner, "shutdown", "owner.shutdown")
    roles.watch(online.RayGenerationLauncher, "create_runtime", "launch")
    roles.watch(ResolvedReward, "build_function", "reward.build")
    roles.watch(online.RolloutCollector, "from_family", "collector.build")
    roles.watch(online.OnlineTrainer, "step", "trainer.step")
    roles.watch(StrictOnPolicyRolloutSchedule, "shutdown", "schedule.shutdown")
    roles.watch(online.RolloutCollector, "shutdown", "collector.shutdown")
    roles.watch(RewardFunctionRuntime, "shutdown", "reward.shutdown")
    roles.watch(online.OnlineRecipeRun, "save_checkpoint", "checkpoint.save")
    roles.watch(online.TrainingRunTrace, "capture", "evidence.capture")
    return roles


_SHUTDOWNS = ("schedule.shutdown", "collector.shutdown", "reward.shutdown", "owner.shutdown")


def _shutdown_order(roles: Trace) -> list[str]:
    return [event for event in roles.events if event in _SHUTDOWNS]


def _reached_ray(roles: Trace) -> bool:
    return any(event in roles.events for event in ("owner.create", "launch", "reward.build"))


def _spy_replay_transformer_load(monkeypatch, *, after_load) -> list[str]:
    """Observe the real replay-bundle transformer load (diffusers' ``from_pretrained``
    on ``transformer/``); ``after_load`` runs while the bundle is still being built."""

    from diffusers import SanaTransformer2DModel

    real_from_pretrained = SanaTransformer2DModel.from_pretrained.__func__
    loads: list[str] = []

    def from_pretrained(cls, path, **kwargs):
        transformer = real_from_pretrained(cls, path, **kwargs)
        loads.append(str(path))
        after_load()
        return transformer

    monkeypatch.setattr(SanaTransformer2DModel, "from_pretrained", classmethod(from_pretrained))
    return loads


def test_owned_ray_session_retries_shutdown_before_committing_closed(monkeypatch) -> None:
    """A failed Ray shutdown leaves the owned session open for the retry."""

    with real_local_ray() as started:
        session = online._RayClusterSession(ray=started, shutdown_on_exit=True)
        calls = Trace(monkeypatch)
        calls.watch(started, "shutdown", "ray.shutdown")
        calls.fail("ray.shutdown", "ray shutdown transport failed")

        with pytest.raises(RuntimeError, match="ray shutdown transport failed"):
            session.shutdown()
        assert session._closed is False
        assert started.is_initialized()

        session.shutdown()
        session.shutdown()

        # The retry shuts Ray down; the closed session never calls it again.
        assert session._closed is True
        assert calls.events == ["ray.shutdown", "ray.shutdown"]
        assert not ray.is_initialized()


# --------------------------------------------------------------- preflight


@pytest.mark.asyncio
async def test_checkpoint_identity_preflight_runs_before_prompt_or_model_build(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    first = _RealRun(monkeypatch, tmp_path / "first", snapshot=ray_snapshot)
    await online.run_online_recipe(first.cfg)
    checkpoint = first.output_dir / "checkpoint-final"
    run = _RealRun(
        monkeypatch,
        tmp_path,
        snapshot=ray_snapshot,
        overrides=(f"trainer.resume_from={checkpoint}",),
    )
    roles = _roles(monkeypatch)
    roles.watch(online.TrainingCheckpoint, "validate_compatibility", "identity.validate")

    class _ReachedPromptBoundary(RuntimeError):
        pass

    roles.watch(online, "load_prompt_examples_from_config", "prompts.load")
    roles.fail("prompts.load", _ReachedPromptBoundary())

    with pytest.raises(_ReachedPromptBoundary):
        await online.run_online_recipe(run.cfg)

    validate = roles.calls[roles.events.index("identity.validate")][1]
    assert validate[0].checkpoint_dir == checkpoint
    assert roles.events.index("identity.validate") < roles.events.index("prompts.load")
    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_checkpoint_identity_mismatch_stops_before_prompt_model_or_ray(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    """A checkpoint saved from a different model directory is refused by the
    real validator before prompts, the model or Ray."""

    first = _RealRun(monkeypatch, tmp_path / "first", snapshot=ray_snapshot)
    await online.run_online_recipe(first.cfg)
    checkpoint = first.output_dir / "checkpoint-final"
    run = _RealRun(monkeypatch, tmp_path, overrides=(f"trainer.resume_from={checkpoint}",))
    (run.snapshot / "provenance.txt").write_text("another checkout", encoding="utf-8")
    roles = _roles(monkeypatch)
    roles.watch(online, "load_prompt_examples_from_config", "prompts.load")

    with pytest.raises(ValueError, match="model identity mismatch"):
        await online.run_online_recipe(run.cfg)

    assert "prompts.load" not in roles.events
    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_missing_prompt_rng_stops_before_prompt_model_or_ray(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    """The by-rank variant of this admission is a unit test on restore_rng_state
    (tests/trainers/test_checkpoint_rng_distributed.py); here one real run
    proves the recipe stops before prompts, the model or Ray."""

    first = _RealRun(monkeypatch, tmp_path / "first", snapshot=ray_snapshot)
    await online.run_online_recipe(first.cfg)
    checkpoint_dir = first.output_dir / "checkpoint-final"
    run = _RealRun(
        monkeypatch,
        tmp_path,
        snapshot=ray_snapshot,
        overrides=(f"trainer.resume_from={checkpoint_dir}",),
    )
    roles = _roles(monkeypatch)
    roles.watch(online, "load_prompt_examples_from_config", "prompts.load")
    rng_state = online.capture_rng_state(prompt_generator=torch.Generator())
    rng_state["generators"] = {"probe": torch.Generator().get_state()}
    checkpoint = online.TrainingCheckpoint.load(checkpoint_dir)
    checkpoint.payload["rng"] = rng_state
    monkeypatch.setattr(
        online.TrainingCheckpoint, "load_for_resume", staticmethod(lambda _resume: checkpoint)
    )

    with pytest.raises(ValueError, match="missing requested generators: prompt_generator"):
        await online.run_online_recipe(run.cfg)

    assert "prompts.load" not in roles.events
    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_checkpoint_source_change_stops_after_model_before_ray_or_reward(
    monkeypatch, tmp_path
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    roles = _roles(monkeypatch)
    loads = _spy_replay_transformer_load(
        monkeypatch,
        after_load=lambda: (run.snapshot / "extra-weights.bin").write_bytes(b"drift"),
    )
    roles.watch(online, "require_ray", "ray.require")

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during replay bundle construction",
    ):
        await online.run_online_recipe(run.cfg)

    assert loads == [str(run.snapshot)]
    assert "ray.require" not in roles.events
    assert not _reached_ray(roles)


def _pin_torchrun_rank(monkeypatch, cuda_devices, *, gpus: int, world_size: int) -> None:
    """A torchrun rank-0 process on a host with ``gpus`` cards, pinned at torch + env."""

    cuda_devices(gpus)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


@pytest.mark.asyncio
async def test_distributed_disjoint_rollout_fails_before_model_or_ray_launch(
    monkeypatch, tmp_path, cuda_devices
) -> None:
    """Multi-rank ranks must not duplicate one global dedicated rollout plan.

    Two FSDP ranks on a four-GPU host with the rollout pinned to the two spare
    cards: the real resource plan resolves a disjoint rollout, so the real
    topology guard rejects it before any model or Ray work."""

    _pin_torchrun_rank(monkeypatch, cuda_devices, gpus=4, world_size=2)
    run = _RealRun(
        monkeypatch,
        tmp_path,
        overrides=(
            "distributed.training.strategy=fsdp",
            "distributed.training.num_nodes=1",
            "distributed.training.gpus_per_node=2",
            "distributed.resources.rollout.devices=[2,3]",
        ),
    )
    roles = _roles(monkeypatch)

    with pytest.raises(
        NotImplementedError,
        match="every torchrun rank would independently initialize Ray",
    ):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_shared_gpu_parking_capability_fails_before_model_or_ray_launch(
    monkeypatch, tmp_path, cuda_devices
) -> None:
    """A strategy that cannot park trainer state off a shared rollout GPU is
    rejected before any model or Ray work: one DDP rank whose rollout shares
    its only card resolves an on-demand rollout lease, and the real
    ``DDPStrategy.validate_training_state_parking`` refuses it."""

    _pin_torchrun_rank(monkeypatch, cuda_devices, gpus=1, world_size=1)
    run = _RealRun(
        monkeypatch,
        tmp_path,
        overrides=(
            "distributed.training.strategy=ddp",
            "distributed.training.num_nodes=1",
            "distributed.training.gpus_per_node=1",
            "distributed.resources.rollout.num_gpus=1",
        ),
    )
    roles = _roles(monkeypatch)

    with pytest.raises(NotImplementedError, match="Use disjoint rollout GPUs"):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_reference_conditioned_task_fails_before_model_or_ray_launch(
    monkeypatch, tmp_path
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    roles = _roles(monkeypatch)
    # The guard reads the family's declared task; SANA is t2i, so the entry is
    # relabelled v2w for this one theorem (the config carries no conditioning).
    entry = resolved_run.resolve_online_run(run.cfg).family
    monkeypatch.setattr(type(entry), "task", "v2w")

    with pytest.raises(ValueError, match=r"data\.preprocessing\.conditioning=reference_image"):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert not _reached_ray(roles)


@pytest.mark.asyncio
async def test_process_seed_is_applied_before_actual_model_build(monkeypatch, tmp_path):
    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    previous = capture_rng_state()
    observed = []

    class ReachedModelBuild(RuntimeError):
        pass

    def inspect_build() -> None:
        observed.append(torch.nn.Linear(4, 3).weight.detach().clone())
        raise ReachedModelBuild()

    run = _RealRun(monkeypatch, tmp_path)
    roles = _roles(monkeypatch)
    _spy_replay_transformer_load(monkeypatch, after_load=inspect_build)
    try:
        for seed in (123, 456):
            torch.manual_seed(seed)
            with pytest.raises(ReachedModelBuild):
                await online.run_online_recipe(run.cfg)
        assert torch.equal(observed[0], observed[1])
        assert not _reached_ray(roles)
    finally:
        restore_rng_state(previous)


# ------------------------------------------------------------- real runs


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_success(
    monkeypatch, tmp_path, ray_snapshot
) -> None:
    # No preexisting driver: the recipe starts an owned local cluster (its
    # workers get the tiny pipeline through the injected job runtime env) and
    # closes it on exit.
    ray.shutdown()
    monkeypatch.setenv(
        "RAY_JOB_CONFIG_JSON_ENV_VAR",
        json.dumps({"runtime_env": tiny_sana_ray_runtime_env(ray_snapshot)}),
    )
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    replay_loads = _spy_replay_transformer_load(monkeypatch, after_load=lambda: None)

    await online.run_online_recipe(run.cfg)

    assert roles.events.count("owner.create") == 1
    assert roles.events.count("trainer.step") == 1
    (save,) = [args for event, args in roles.calls if event == "checkpoint.save"]
    assert save[1].name == "checkpoint-final"
    assert (run.output_dir / "checkpoint-final").is_dir()
    # The driver's replay bundle is the real snapshot transformer, loaded once.
    assert replay_loads == [str(run.snapshot)]
    (launch,) = [args for event, args in roles.calls if event == "launch"]
    _launcher, generation_config, launch_inputs = launch
    assert generation_config.worker.cpus_per_worker == 0.5
    # The launch contract carries the identity the real evidence record froze.
    (evidence,) = run.evidence()
    assert launch_inputs.launch_contract.expected_model_identity == evidence["model_identity"]
    assert evidence["resumed"] is False
    # Once the trainer exists, the recipe releases the rollout pipeline through
    # the schedule alone; the schedule cascades into the collector, which
    # releases its reward, and the placement owner goes last.
    assert _shutdown_order(roles) == [
        "schedule.shutdown",
        "collector.shutdown",
        "reward.shutdown",
        "owner.shutdown",
    ]
    assert not ray.is_initialized()


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_resume_releases_full_checkpoint_payload_before_training(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    first = _RealRun(monkeypatch, tmp_path / "first", snapshot=ray_snapshot)
    await online.run_online_recipe(first.cfg)
    checkpoint_dir = first.output_dir / "checkpoint-final"
    run = _RealRun(
        monkeypatch,
        tmp_path,
        snapshot=ray_snapshot,
        overrides=(f"trainer.resume_from={checkpoint_dir}", "trainer.total_epochs=2"),
    )
    roles = _roles(monkeypatch)
    roles.watch(online.TrainingCheckpoint, "load_for_resume", "checkpoint.load")
    roles.watch(online, "restore_training_checkpoint", "checkpoint.restore")
    roles.watch(gc, "collect", "gc.collect")

    await online.run_online_recipe(run.cfg)

    (checkpoint,) = roles.results["checkpoint.load"]
    assert checkpoint.checkpoint_dir == checkpoint_dir
    # The full payload is restored into the real trainer, then released and
    # collected before the first training step.
    assert checkpoint.payload == {}
    restored = roles.events.index("checkpoint.restore")
    collected = roles.events.index("gc.collect", restored)
    assert collected < roles.events.index("trainer.step")
    assert roles.events.count("trainer.step") == 1
    (evidence,) = run.evidence()
    assert evidence["resumed"] is True


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_online_preview_matches_the_next_epoch_prompt_batch(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(
        monkeypatch,
        tmp_path,
        snapshot=ray_snapshot,
        prompts=("prompt-0", "prompt-1", "prompt-2"),
        overrides=("trainer.total_epochs=2",),
    )
    batches: list[dict[str, Any]] = []
    real_step = online.OnlineTrainer.step

    async def step(self, example_batch, *, next_prompts=None):
        batches.append(
            {
                "current": [example.prompt for example in example_batch],
                "next": None
                if next_prompts is None
                else [example.prompt for example in next_prompts],
            }
        )
        return await real_step(self, example_batch, next_prompts=next_prompts)

    monkeypatch.setattr(online.OnlineTrainer, "step", step)

    await online.run_online_recipe(run.cfg)

    assert len(batches) == 2
    assert batches[0]["next"] == batches[1]["current"]
    assert batches[1]["next"] is None
    assert preinitialized_ray.is_initialized()


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_rollout_sync_getter_routes_through_strategy(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    """Every rollout weight push carries a fresh strategy export.

    The strategy produces rollout-facing weights, so FSDP controls its gathers
    without the recipe selecting or flattening model parameters itself.
    """

    from vrl.trainers.strategy import SingleProcessStrategy

    # Train on every group (tiny images can tie on sharpness) so the step
    # really updates and pushes again after the optimizer step.
    run = _RealRun(
        monkeypatch,
        tmp_path,
        snapshot=ray_snapshot,
        overrides=("actor.drop_zero_advantage=false",),
    )
    roles = _roles(monkeypatch)
    roles.watch(SingleProcessStrategy, "export_rollout_state", "strategy.export")
    roles.watch(RayRuntimeWeightSyncer, "push", "weights.push")

    await online.run_online_recipe(run.cfg)

    pushes = [args for event, args in roles.calls if event == "weights.push"]
    exports = roles.results["strategy.export"]
    # The initial push and the post-step push each re-export live state.
    assert len(pushes) == len(exports) == 2
    assert [push[1] for push in pushes] == exports
    assert exports[0] is not exports[1]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_create_failure(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("owner.create", "owner create boom")

    with pytest.raises(RuntimeError, match="owner create boom"):
        await online.run_online_recipe(run.cfg)

    assert roles.events.count("owner.create") == 1
    assert "launch" not in roles.events
    assert "reward.build" not in roles.events
    assert _shutdown_order(roles) == ["owner.shutdown"]
    # Even on failure, the embedding caller's Ray connection is not ours to close.
    assert preinitialized_ray.is_initialized()


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_rollout_launch_failure(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("launch", "launch boom")

    with pytest.raises(RuntimeError, match="launch boom"):
        await online.run_online_recipe(run.cfg)

    # No schedule exists yet, so the recipe falls back to the collector, whose
    # own cascade releases the reward; the recipe never touches the reward.
    assert _shutdown_order(roles) == [
        "collector.shutdown",
        "reward.shutdown",
        "owner.shutdown",
    ]


@pytest.mark.slow_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reward.build", "collector.build"])
async def test_run_online_recipe_shutdowns_owner_after_component_build_failure(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot, failure
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail(failure, f"{failure} boom")

    with pytest.raises(RuntimeError, match=f"{failure} boom"):
        await online.run_online_recipe(run.cfg)

    assert roles.events.count("owner.create") == 1
    assert "launch" not in roles.events
    # Partial-acquisition fallback: with neither schedule nor collector, the
    # recipe shuts the standalone reward runtime down directly. A failed reward
    # build leaves nothing to release.
    if failure == "reward.build":
        assert _shutdown_order(roles) == ["owner.shutdown"]
    else:
        assert _shutdown_order(roles) == ["reward.shutdown", "owner.shutdown"]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_final_checkpoint_failure(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("checkpoint.save", "save boom")

    with pytest.raises(RuntimeError, match="save boom"):
        await online.run_online_recipe(run.cfg)

    # The schedule exists by this point, so the recipe releases the pipeline
    # through it alone.
    assert _shutdown_order(roles) == [
        "schedule.shutdown",
        "collector.shutdown",
        "reward.shutdown",
        "owner.shutdown",
    ]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_do_not_hide_training_error(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("trainer.step", "train boom")
    roles.fail("schedule.shutdown", "schedule shutdown boom")
    roles.fail("owner.shutdown", "owner shutdown boom")

    with pytest.raises(RuntimeError, match="train boom"):
        await online.run_online_recipe(run.cfg)

    # Both releases are attempted once; neither failure replaces the training error.
    assert _shutdown_order(roles) == ["schedule.shutdown", "owner.shutdown"]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_after_success_run_all_cleanups(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("schedule.shutdown", "schedule shutdown boom")

    with pytest.raises(RuntimeError, match="rollout_schedule shutdown failed"):
        await online.run_online_recipe(run.cfg)

    assert _shutdown_order(roles) == ["schedule.shutdown", "owner.shutdown"]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_launch_evidence_failure_stops_before_training_and_cleans_up(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("evidence.capture", OSError("evidence storage full"))

    with pytest.raises(OSError, match="evidence storage full"):
        await online.run_online_recipe(run.cfg)

    assert "trainer.step" not in roles.events
    # The schedule already exists when evidence is captured, so the pipeline
    # is released through it, never the collector directly.
    assert _shutdown_order(roles) == [
        "schedule.shutdown",
        "collector.shutdown",
        "reward.shutdown",
        "owner.shutdown",
    ]


# ------------------------------------------------- terminal cleanup policy


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_terminal_schedule_is_the_only_collector_shutdown_owner(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    """With a schedule, a collector and a standalone reward all registered, the
    lifecycle releases through the schedule only, then cleans the strategy."""

    from vrl.trainers.strategy import SingleProcessStrategy

    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.watch(SingleProcessStrategy, "shutdown", "strategy.shutdown")
    lifecycles: list[Any] = []
    real_shutdown = online._OnlineRecipeLifecycle.shutdown

    async def shutdown(self, *, run_error):
        lifecycles.append(self)
        return await real_shutdown(self, run_error=run_error)

    monkeypatch.setattr(online._OnlineRecipeLifecycle, "shutdown", shutdown)

    await online.run_online_recipe(run.cfg)

    (lifecycle,) = lifecycles
    assert lifecycle.rollout_schedule is not None
    assert lifecycle.collector is not None
    assert lifecycle.reward_runtime is not None
    assert [
        event
        for event in roles.events
        if event in (*_SHUTDOWNS, "strategy.shutdown") and event != "owner.shutdown"
    ] == ["schedule.shutdown", "collector.shutdown", "reward.shutdown", "strategy.shutdown"]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_failed_role_cleanup_abandons_parked_restore_but_cleans_strategy(
    monkeypatch, tmp_path, preinitialized_ray, ray_snapshot
) -> None:
    from vrl.trainers.strategy import SingleProcessStrategy

    run = _RealRun(monkeypatch, tmp_path, snapshot=ray_snapshot)
    roles = _roles(monkeypatch)
    roles.fail("trainer.step", "training failed")
    roles.fail("schedule.shutdown", "rollout cleanup failed")
    restore_permissions: list[bool] = []
    real_shutdown = SingleProcessStrategy.shutdown

    def shutdown(self, *, restore_parked: bool = True) -> None:
        restore_permissions.append(restore_parked)
        return real_shutdown(self, restore_parked=restore_parked)

    monkeypatch.setattr(SingleProcessStrategy, "shutdown", shutdown)

    with pytest.raises(RuntimeError, match="training failed"):
        await online.run_online_recipe(run.cfg)

    assert restore_permissions == [False]
