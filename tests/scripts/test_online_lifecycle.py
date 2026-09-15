"""Shutdown order and error propagation of ``run_online_recipe``.

Everything up to the Ray side is real: the SANA aesthetic GRPO preset resolved
onto the tiny local snapshot (``tiny_sana_online_config``), the real family
loader over ``TinySanaPipeline`` (the Hub load is the one model double), the
real local-directory checkpoint identity, the real resource plan on a GPU-less
rollout fleet, the real launch contract and the real training launch evidence.

The Ray-side roles -- placement owner, rollout launcher/runtime, collector,
reward runtime, trainer and its rollout schedule -- are call recorders: the
theorems here are *which* of them the recipe releases, in what order, and which
error survives, so the recorders write a ``shutdown_order`` ledger and raise on
demand. None of them re-implements a production cascade (see the docstrings on
``_FakeCollector`` / ``_FakeSchedule``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.conftest import real_local_ray
from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
from tests.trainers._checkpoint_helpers import _Trainer
from vrl import run as resolved_run
from vrl.algorithms.types import TrainStepMetrics
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.scripts.common import online
from vrl.trainers.checkpointing import save_training_checkpoint
from vrl.trainers.data.prompts import PromptExample

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def preinitialized_ray():
    """Shared real local Ray cluster. The recipe sees an is_initialized()
    driver, takes the embedding-caller ownership branch, and must leave the
    cluster running — asserted per-test via ray.is_initialized().

    Module-scoped through the one cluster owner in tests/conftest.py: the
    hand-rolled version was function-scoped but only initialized when no cluster
    was up, and a separate autouse module fixture did the closing — module scope
    says the same thing in one place. Going through ``real_local_ray`` is also
    what restores ``RAY_ENABLE_UV_RUN_RUNTIME_ENV``: this fixture never flipped
    it and was only surviving under ``uv run`` because
    tests/generation/ray/test_rollout_launcher.py leaked it process-wide.
    """

    with real_local_ray() as started:
        yield started


class _FakeReward:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.task = "t2i"

    async def preflight(self) -> None:
        return None

    async def shutdown(self) -> None:
        self._state["reward_shutdowns"] += 1
        self._state["shutdown_order"].append("reward")


class _FakeCollector:
    """Pure call recorder; it does NOT re-implement the production cascade.

    The generation→reward shutdown cascade (runtime first, retain the reward
    asleep and raise on runtime failure, ``_reward_shutdown_complete``
    idempotence) is owned by ``RolloutCollector.shutdown``
    (vrl/rollouts/collector/core.py) and has direct real coverage in
    tests/rollouts/collector/test_runtime.py. The recipe under test only ever
    calls ``collector.shutdown()`` itself.
    """

    def __init__(self, state: dict[str, Any], reward: _FakeReward) -> None:
        self._state = state
        self._reward = reward
        self._generation_runtime: Any | None = None

    def set_generation_runtime(self, runtime: Any) -> None:
        self._generation_runtime = runtime
        self._state["collector_set_generation_runtime"] += 1

    @property
    def generation_runtime(self) -> Any:
        return self._generation_runtime

    async def shutdown(self) -> None:
        self._state["collector_shutdowns"] += 1
        self._state["shutdown_order"].append("collector")
        if self._state.get("collector_shutdown_raises"):
            raise RuntimeError("collector shutdown boom")


class _FakeRuntime:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def preflight(self) -> None:
        self._state.setdefault("runtime_preflights", 0)
        self._state["runtime_preflights"] += 1

    async def shutdown(self) -> None:
        self._state["runtime_shutdowns"] += 1
        self._state["shutdown_order"].append("runtime")


class _FakeSchedule:
    """Pure call recorder; it does NOT re-implement the production cascade.

    The recipe releases the whole rollout pipeline through the schedule alone
    (``_OnlineRecipeLifecycle.shutdown``); the real schedule's cascade into the
    collector (StrictOnPolicyRolloutSchedule.shutdown → shutdown_collector_runtime)
    is schedule-internal and must not be re-implemented here.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def shutdown(self) -> None:
        self._state["schedule_shutdowns"] += 1
        self._state["shutdown_order"].append("schedule")
        if self._state.get("schedule_shutdown_raises"):
            raise RuntimeError("schedule shutdown boom")


class _FakeLauncher:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def create_runtime(
        self,
        config: Any,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: Any,
    ) -> _FakeRuntime:
        del placement
        self._state["launcher_worker"] = config.worker
        self._state["launcher_model_identity"] = (
            launch_inputs.launch_contract.expected_model_identity
        )
        self._state["launches"] += 1
        if self._state.get("launch_raises"):
            raise RuntimeError("launch boom")
        return _FakeRuntime(self._state)


class _FakePlacementOwner:
    def __init__(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.rollout_worker = args[1] if len(args) > 1 else kwargs["rollout_worker"]
        self._state = state
        self._state["placement_worker"] = self.rollout_worker
        self.rollout_placement = object()
        self.reward_placement = None
        self.layout = SimpleNamespace(bundle_gpu_ids=())

    def required_local_cluster_cpus(self) -> int:
        self._state["owner_cpu_plans"] += 1
        return 1

    def create(self) -> None:
        self._state["owner_creates"] += 1
        if self._state.get("owner_create_raises"):
            raise RuntimeError("owner create boom")

    def shutdown(self) -> None:
        self._state["owner_shutdowns"] += 1
        self._state["shutdown_order"].append("owner")
        if self._state.get("owner_shutdown_raises"):
            raise RuntimeError("owner shutdown boom")


class _FakeTrainer:
    def __init__(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del args
        self._state = state
        self.state = SimpleNamespace(global_step=0)
        self.rollout_schedule = _FakeSchedule(state)

    async def step(
        self,
        example_batch: list[Any],
        *,
        next_prompts: list[Any] | None = None,
    ) -> Any:
        self._state["trainer_prompt_batches"].append(
            {
                "current": list(example_batch),
                "next": None if next_prompts is None else list(next_prompts),
            },
        )
        self._state["trainer_steps"] += 1
        if self._state.get("trainer_step_raises"):
            raise RuntimeError("train boom")
        self.state.global_step += 1
        # The real metrics CSV reads every field of the real step result.
        return TrainStepMetrics()


def _state() -> dict[str, Any]:
    return {
        "collector_set_generation_runtime": 0,
        "collector_shutdowns": 0,
        "runtime_shutdowns": 0,
        "schedule_shutdowns": 0,
        "reward_shutdowns": 0,
        "owner_creates": 0,
        "owner_cpu_plans": 0,
        "owner_shutdowns": 0,
        "launches": 0,
        "launcher_worker": None,
        "launcher_model_identity": None,
        "placement_worker": None,
        "trainer_steps": 0,
        "trainer_prompt_batches": [],
        "checkpoint_paths": [],
        "shutdown_order": [],
    }


def test_owned_ray_session_retries_shutdown_before_committing_closed() -> None:
    class _Ray:
        def __init__(self) -> None:
            self.calls = 0

        def shutdown(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ray shutdown transport failed")

    ray_api = _Ray()
    session = online._RayClusterSession(ray=ray_api, shutdown_on_exit=True)

    with pytest.raises(RuntimeError, match="ray shutdown transport failed"):
        session.shutdown()
    assert session._closed is False

    session.shutdown()
    session.shutdown()
    assert session._closed is True
    assert ray_api.calls == 2


class _RealRun:
    """One real tiny SANA run: config, snapshot and the pipeline the loader serves."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        prompts: tuple[str, ...] = ("a cat",),
        overrides: tuple[str, ...] = (),
        on_load: Any = None,
    ) -> None:
        self.cfg = tiny_sana_online_config(
            tmp_path,
            prompts=prompts,
            overrides=(
                "trainer.save_freq=0",
                "distributed.rollout.cpus_per_worker=0.5",
                *overrides,
            ),
        )
        self.snapshot = tmp_path / "sana-snapshot"
        self.output_dir = tmp_path / "run"
        self.pipeline = TinySanaPipeline()
        self.pipeline.install(monkeypatch, self.snapshot, on_load=on_load)

    def replay_identity(self) -> dict[str, Any]:
        """The identity the recipe resolves for this run's replay model."""

        resolved = resolved_run.resolve_online_run(self.cfg)
        return resolved_run.resolve_model(
            resolved.family,
            resolved.built.root,
            resolved.device,
            precision=resolved.built.precision,
            for_rollout=False,
        ).identity

    def save_checkpoint(self, path: Path) -> Path:
        """A real checkpoint of this run's replay bundle, for ``trainer.resume_from``."""

        resolved = resolved_run.resolve_online_run(self.cfg)
        replay = resolved_run.resolve_model(
            resolved.family,
            resolved.built.root,
            resolved.device,
            precision=resolved.built.precision,
            for_rollout=False,
        )
        bundle = replay.materialize(context="lifecycle test checkpoint")
        save_training_checkpoint(
            path,
            trainer=_Trainer(),
            bundle=bundle,
            family="sana",
            progress={"next_epoch": 1, "next_step": 1},
            rng_state=online.capture_rng_state(prompt_generator=torch.Generator()),
            model_identity=replay.identity,
        )
        self.pipeline.loads = 0
        return path

    def evidence(self) -> list[dict[str, Any]]:
        directory = self.output_dir / "run_evidence"
        return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


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


def _install_ray_side_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: dict[str, Any],
) -> _FakeReward:
    """Replace the Ray-side roles with shutdown recorders; everything else is real."""

    reward = _FakeReward(state)
    collector = _FakeCollector(state, reward)
    monkeypatch.setattr(
        online,
        "GlobalRayPlacementOwner",
        lambda *args, **kwargs: _FakePlacementOwner(state, *args, **kwargs),
    )
    monkeypatch.setattr(online, "build_reward_runtime", lambda *args, **kwargs: reward)
    monkeypatch.setattr(
        online.RolloutCollector,
        "from_family",
        lambda *args, **kwargs: collector,
    )
    monkeypatch.setattr(online, "RayGenerationLauncher", lambda: _FakeLauncher(state))
    monkeypatch.setattr(
        online, "OnlineTrainer", lambda *args, **kwargs: _FakeTrainer(state, *args, **kwargs)
    )
    monkeypatch.setattr(
        online.RayRuntimeWeightSyncer,
        "if_supported",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    # The recorder trainer has no optimizer state to save and sealing hashes the
    # written checkpoint-final, so both stay recorders keyed on the ledger.
    monkeypatch.setattr(
        online.OnlineRecipeRun,
        "save_checkpoint",
        lambda self, path, *args, **kwargs: state["checkpoint_paths"].append(path.name),
    )
    monkeypatch.setattr(
        online.TrainingRunTrace, "seal_artifacts", lambda self: self.artifacts_path
    )
    return reward


@pytest.mark.asyncio
async def test_injected_prompt_examples_bypass_manifest_loader(monkeypatch, tmp_path) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    provided = PromptExample(prompt="frozen prompt", metadata={"source": "snapshot"})
    monkeypatch.setattr(
        online,
        "load_prompt_examples_from_config",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("manifest path must not reopen")),
    )
    seen: list[PromptExample] = []

    class _ReachedResolvedPrompt(RuntimeError):
        pass

    def _stop_after_selection(example: PromptExample, **_kwargs: Any) -> PromptExample:
        seen.append(example)
        raise _ReachedResolvedPrompt

    monkeypatch.setattr(online, "resolve_prompt_example_references", _stop_after_selection)

    with pytest.raises(_ReachedResolvedPrompt):
        await online.run_online_recipe(run.cfg, prompt_examples=(provided,))

    assert seen == [provided]
    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0


@pytest.mark.asyncio
async def test_checkpoint_identity_preflight_runs_before_prompt_or_model_build(
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    checkpoint = run.save_checkpoint(tmp_path / "checkpoint-1")
    run = _RealRun(monkeypatch, tmp_path, overrides=(f"trainer.resume_from={checkpoint}",))
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    validated: list[tuple[Path, str, bool]] = []
    real_validate = online.validate_checkpoint_compatibility

    def spy_validate(checkpoint, *, family, expected_model_identity, strict):
        validated.append((checkpoint.checkpoint_dir, family, strict))
        real_validate(
            checkpoint,
            family=family,
            expected_model_identity=expected_model_identity,
            strict=strict,
        )

    monkeypatch.setattr(online, "validate_checkpoint_compatibility", spy_validate)

    class _ReachedPromptBoundary(RuntimeError):
        pass

    def _stop_at_prompt(_cfg: Any) -> list[Any]:
        assert validated == [(checkpoint, "sana", True)]
        raise _ReachedPromptBoundary

    monkeypatch.setattr(online, "load_prompt_examples_from_config", _stop_at_prompt)

    with pytest.raises(_ReachedPromptBoundary):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.asyncio
async def test_checkpoint_identity_mismatch_stops_before_prompt_model_or_ray(
    monkeypatch,
    tmp_path,
) -> None:
    """A checkpoint saved from a different model directory is refused by the
    real validator before prompts, the model or Ray."""

    other = _RealRun(monkeypatch, tmp_path / "other")
    (other.snapshot / "provenance.txt").write_text("another checkout", encoding="utf-8")
    checkpoint = other.save_checkpoint(tmp_path / "checkpoint-1")
    run = _RealRun(monkeypatch, tmp_path, overrides=(f"trainer.resume_from={checkpoint}",))
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    monkeypatch.setattr(
        online,
        "load_prompt_examples_from_config",
        lambda _cfg: (_ for _ in ()).throw(
            AssertionError("prompt loading must not run after identity mismatch"),
        ),
    )

    with pytest.raises(ValueError, match="model identity mismatch"):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("world_size, missing_rank", [(1, 0), (4, 3)])
async def test_missing_prompt_rng_stops_before_prompt_model_or_ray(
    monkeypatch,
    tmp_path,
    world_size,
    missing_rank,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    checkpoint_dir = run.save_checkpoint(tmp_path / "checkpoint-1")
    run = _RealRun(monkeypatch, tmp_path, overrides=(f"trainer.resume_from={checkpoint_dir}",))
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    rank_states = [
        online.capture_rng_state(prompt_generator=torch.Generator()) for _ in range(world_size)
    ]
    rank_states[missing_rank]["generators"] = {"probe": torch.Generator().get_state()}
    rng_state = (
        rank_states[0]
        if world_size == 1
        else {
            "world_size": world_size,
            "by_rank": rank_states,
        }
    )
    checkpoint = online.TrainingCheckpoint.load(checkpoint_dir)
    checkpoint.payload["rng"] = rng_state
    monkeypatch.setattr(
        online.TrainingCheckpoint, "load_for_resume", staticmethod(lambda _resume: checkpoint)
    )
    monkeypatch.setattr(
        online.DistributedTrainingContext,
        "from_root",
        classmethod(
            lambda _cls, _cfg, **_kwargs: SimpleNamespace(
                world_size=world_size, rank=0, device=torch.device("cpu")
            )
        ),
    )
    monkeypatch.setattr(
        online, "_require_supported_distributed_rollout_topology", lambda *_args: None
    )
    monkeypatch.setattr(
        online,
        "build_strategy",
        lambda *_args: SimpleNamespace(validate_training_state_parking=lambda: None),
    )
    monkeypatch.setattr(
        online,
        "load_prompt_examples_from_config",
        lambda _cfg: (_ for _ in ()).throw(
            AssertionError("manifest must not load before RNG admission")
        ),
    )
    with pytest.raises(ValueError, match="missing requested generators: prompt_generator"):
        await online.run_online_recipe(run.cfg)
    assert run.pipeline.loads == state["owner_creates"] == state["launches"] == 0


@pytest.mark.asyncio
async def test_checkpoint_source_change_stops_after_model_before_ray_or_reward(
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    loads = _spy_replay_transformer_load(
        monkeypatch,
        after_load=lambda: (run.snapshot / "extra-weights.bin").write_bytes(b"drift"),
    )
    monkeypatch.setattr(
        online,
        "require_ray",
        lambda: (_ for _ in ()).throw(
            AssertionError("Ray must not start after checkpoint source changes"),
        ),
    )
    monkeypatch.setattr(
        online,
        "build_reward_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reward must not build after checkpoint source changes"),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during replay bundle construction",
    ):
        await online.run_online_recipe(run.cfg)

    assert loads == [str(run.snapshot)]
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_success(monkeypatch, tmp_path) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    # No preexisting driver: the recipe must start an owned local cluster and
    # close it on exit. Real Ray cannot append to the shutdown_order ledger, so
    # the post-run is_initialized() check stands in for the trailing "ray" entry.
    ray.shutdown()
    replay_loads = _spy_replay_transformer_load(monkeypatch, after_load=lambda: None)

    await online.run_online_recipe(run.cfg)

    assert state["owner_creates"] == 1
    assert state["owner_cpu_plans"] == 1
    assert state["trainer_steps"] == 1
    assert state["checkpoint_paths"] == ["checkpoint-final"]
    # The driver's replay bundle is the real snapshot transformer, loaded once.
    assert replay_loads == [str(run.snapshot)]
    assert state["collector_shutdowns"] == 0
    assert state["runtime_shutdowns"] == 0
    assert state["schedule_shutdowns"] == 1
    assert state["reward_shutdowns"] == 0
    assert state["owner_shutdowns"] == 1
    assert state["launcher_worker"] is state["placement_worker"]
    assert state["launcher_worker"].cpus_per_worker == 0.5
    # The launch contract carries the identity the real evidence record froze.
    (evidence,) = run.evidence()
    assert state["launcher_model_identity"] == evidence["model_identity"]
    assert evidence["resumed"] is False
    # Once the trainer exists, the recipe releases the rollout pipeline through
    # the schedule alone: collector, runtime, and reward are the schedule's to
    # cascade into, never the recipe's — so the recorders must stay untouched.
    assert state["shutdown_order"] == ["schedule", "owner"]
    assert not ray.is_initialized()


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_resume_releases_full_checkpoint_payload_before_training(
    monkeypatch,
    tmp_path,
    preinitialized_ray,
) -> None:
    del preinitialized_ray
    run = _RealRun(monkeypatch, tmp_path)
    checkpoint_dir = run.save_checkpoint(tmp_path / "checkpoint-1")
    run = _RealRun(
        monkeypatch,
        tmp_path,
        overrides=(f"trainer.resume_from={checkpoint_dir}", "trainer.total_epochs=2"),
    )
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    loaded: list[Any] = []
    real_load = online.TrainingCheckpoint.load_for_resume

    def spy_load(resume):
        checkpoint = real_load(resume)
        loaded.append(checkpoint)
        return checkpoint

    monkeypatch.setattr(online.TrainingCheckpoint, "load_for_resume", staticmethod(spy_load))
    # The recorder trainer owns no optimizer state to restore into.
    monkeypatch.setattr(online, "restore_training_checkpoint", lambda *args, **kwargs: None)
    collect_calls: list[bool] = []
    monkeypatch.setattr(online.gc, "collect", lambda: collect_calls.append(True))

    await online.run_online_recipe(run.cfg)

    (checkpoint,) = loaded
    assert checkpoint.checkpoint_dir == checkpoint_dir
    assert checkpoint.payload == {}
    assert collect_calls == [True]
    assert state["trainer_steps"] == 1
    (evidence,) = run.evidence()
    assert evidence["resumed"] is True


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_online_preview_matches_the_next_epoch_prompt_batch(
    monkeypatch,
    tmp_path,
    preinitialized_ray,
) -> None:
    run = _RealRun(
        monkeypatch,
        tmp_path,
        prompts=("prompt-0", "prompt-1", "prompt-2"),
        overrides=("trainer.total_epochs=2",),
    )
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    await online.run_online_recipe(run.cfg)

    batches = state["trainer_prompt_batches"]
    assert len(batches) == 2
    assert batches[0]["next"] == batches[1]["current"]
    assert batches[1]["next"] is None
    assert preinitialized_ray.is_initialized()


def _pin_torchrun_rank(monkeypatch, cuda_devices, *, gpus: int, world_size: int) -> None:
    """A torchrun rank-0 process on a host with ``gpus`` cards, pinned at torch + env."""

    cuda_devices(gpus)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


@pytest.mark.asyncio
async def test_distributed_disjoint_rollout_fails_before_model_or_ray_launch(
    monkeypatch,
    tmp_path,
    cuda_devices,
) -> None:
    """Multi-rank ranks must not duplicate one global dedicated rollout plan.

    Two FSDP ranks on a four-GPU host with a dedicated two-GPU rollout pool: the
    real resource plan resolves rollout to the spare cards, so the real topology
    guard rejects it before any model or Ray work."""

    _pin_torchrun_rank(monkeypatch, cuda_devices, gpus=4, world_size=2)
    run = _RealRun(
        monkeypatch,
        tmp_path,
        overrides=(
            "distributed.training.strategy=fsdp",
            "distributed.training.num_nodes=1",
            "distributed.training.gpus_per_node=2",
            "distributed.resources.rollout.num_gpus=2",
            "distributed.resources.rollout.gpu_pool=dedicated",
        ),
    )
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(
        NotImplementedError,
        match="every torchrun rank would independently initialize Ray",
    ):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.asyncio
async def test_shared_gpu_parking_capability_fails_before_model_or_ray_launch(
    monkeypatch,
    tmp_path,
    cuda_devices,
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
            "distributed.resources.rollout.gpu_pool=trainer",
        ),
    )
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(NotImplementedError, match="Use disjoint rollout GPUs"):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.asyncio
async def test_reference_conditioned_task_fails_before_model_or_ray_launch(
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    # The guard reads the family's declared task; SANA is t2i, so the entry is
    # relabelled v2w for this one theorem (the config carries no conditioning).
    entry = resolved_run.resolve_online_run(run.cfg).family
    monkeypatch.setattr(type(entry), "task", "v2w")

    with pytest.raises(
        ValueError,
        match=r"data\.preprocessing\.conditioning=reference_image",
    ):
        await online.run_online_recipe(run.cfg)

    assert run.pipeline.loads == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_rollout_sync_getter_routes_through_strategy(
    preinitialized_ray, monkeypatch, tmp_path
) -> None:
    """The recipe binds the rollout sync getter to the strategy, not the raw helper.

    The strategy produces rollout-facing weights, so FSDP controls its gathers
    without requiring the recipe to select or flatten model parameters itself.
    """
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> _FakeTrainer:
        captured.update(kwargs)
        return _FakeTrainer(state, *args, **kwargs)

    monkeypatch.setattr(online, "OnlineTrainer", _capture)

    await online.run_online_recipe(run.cfg)

    from vrl.trainers.strategy import SingleProcessStrategy

    assert isinstance(captured["strategy"], SingleProcessStrategy)

    export_calls: list[Any] = []
    sentinel = {"adapter.weight": torch.ones(1)}
    monkeypatch.setattr(
        captured["strategy"],
        "export_rollout_state",
        lambda bundle: export_calls.append(bundle) or sentinel,
    )
    # The getter must delegate to strategy.export_rollout_state on every call,
    # re-reading live state rather than a value snapshotted at build time.
    assert captured["sync_state_getter"]() is sentinel
    assert len(export_calls) == 1


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_create_failure(
    preinitialized_ray, monkeypatch, tmp_path
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    state["owner_create_raises"] = True
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="owner create boom"):
        await online.run_online_recipe(run.cfg)

    assert state["owner_creates"] == 1
    assert state["owner_shutdowns"] == 1
    assert state["launches"] == 0
    assert state["collector_shutdowns"] == 0
    assert state["reward_shutdowns"] == 0
    # Even on failure, the embedding caller's Ray connection is not ours to close.
    assert preinitialized_ray.is_initialized()


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_rollout_launch_failure(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    state["launch_raises"] = True
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="launch boom"):
        await online.run_online_recipe(run.cfg)

    assert state["owner_creates"] == 1
    assert state["collector_shutdowns"] == 1
    # No schedule exists yet, so the recipe falls back to the collector. It
    # never touches the reward directly: the real collector owns the
    # generation→reward cascade (vrl/rollouts/collector/core.py), covered by
    # tests/rollouts/collector/test_runtime.py.
    assert state["reward_shutdowns"] == 0
    assert state["owner_shutdowns"] == 1
    assert state["shutdown_order"] == ["collector", "owner"]


@pytest.mark.slow_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reward", "collector"])
async def test_run_online_recipe_shutdowns_owner_after_component_build_failure(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
    failure,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    if failure == "reward":
        monkeypatch.setattr(
            online,
            "build_reward_runtime",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reward build boom")),
        )
        message = "reward build boom"
    else:
        monkeypatch.setattr(
            online.RolloutCollector,
            "from_family",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("collector build boom")),
        )
        message = "collector build boom"

    with pytest.raises(RuntimeError, match=message):
        await online.run_online_recipe(run.cfg)

    assert state["owner_creates"] == 1
    assert state["owner_shutdowns"] == 1
    assert state["collector_shutdowns"] == 0
    # Partial-acquisition fallback: with neither schedule nor collector, the
    # recipe shuts the standalone reward runtime down directly. In the "reward"
    # case the reward build itself failed so nothing exists; in the "collector"
    # case the built reward is still standalone.
    assert state["reward_shutdowns"] == (0 if failure == "reward" else 1)
    expected_order = ["owner"] if failure == "reward" else ["reward", "owner"]
    assert state["shutdown_order"] == expected_order


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_final_checkpoint_failure(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    def raise_checkpoint(self: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        del self, path, args, kwargs
        raise RuntimeError("save boom")

    monkeypatch.setattr(online.OnlineRecipeRun, "save_checkpoint", raise_checkpoint)

    with pytest.raises(RuntimeError, match="save boom"):
        await online.run_online_recipe(run.cfg)

    # The schedule exists by this point, so the recipe releases the pipeline
    # through it alone; collector/reward are the schedule's to cascade into.
    assert state["schedule_shutdowns"] == 1
    assert state["collector_shutdowns"] == 0
    assert state["reward_shutdowns"] == 0
    assert state["owner_shutdowns"] == 1


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_do_not_hide_training_error(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    state["trainer_step_raises"] = True
    state["schedule_shutdown_raises"] = True
    state["owner_shutdown_raises"] = True
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="train boom"):
        await online.run_online_recipe(run.cfg)

    # Both releases are attempted on the error path and again in the final
    # cleanup; neither failure replaces the training error.
    assert state["shutdown_order"] == ["schedule", "schedule", "owner", "owner"]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_after_success_run_all_cleanups(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    state["schedule_shutdown_raises"] = True
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="rollout_schedule shutdown failed"):
        await online.run_online_recipe(run.cfg)

    assert state["shutdown_order"] == ["schedule", "schedule", "owner"]


@pytest.mark.asyncio
async def test_terminal_schedule_is_the_only_collector_shutdown_owner() -> None:
    calls: list[str] = []

    class _Collector:
        async def shutdown(self) -> None:
            calls.append("collector")

    collector = _Collector()

    class _Schedule:
        async def shutdown(self) -> None:
            calls.append("schedule")
            await collector.shutdown()

    class _StandaloneReward:
        async def shutdown(self) -> None:
            calls.append("standalone_reward")

    class _Strategy:
        def shutdown(self, *, restore_parked: bool = True) -> None:
            calls.append(f"strategy:{restore_parked}")

    lifecycle = online._OnlineRecipeLifecycle(
        placement_owner=None,
        strategy=_Strategy(),
        rollout_schedule=_Schedule(),
        collector=collector,
        reward_runtime=_StandaloneReward(),
    )
    await lifecycle.shutdown(run_error=None)

    assert calls == ["schedule", "collector", "strategy:True"]


@pytest.mark.asyncio
async def test_terminal_placement_and_ray_cleanup_retry_once() -> None:
    class _FlakyCleanup:
        def __init__(self) -> None:
            self.calls = 0

        def shutdown(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient cleanup failure")

    placement = _FlakyCleanup()
    ray_session = _FlakyCleanup()

    lifecycle = online._OnlineRecipeLifecycle(
        placement_owner=placement,
        strategy=None,
        ray_session=ray_session,
    )
    await lifecycle.shutdown(run_error=None)

    assert placement.calls == 2
    assert ray_session.calls == 2


@pytest.mark.asyncio
async def test_failed_role_cleanup_abandons_parked_restore_but_cleans_strategy() -> None:
    restore_permissions: list[bool] = []

    class _FailingSchedule:
        async def shutdown(self) -> None:
            raise RuntimeError("rollout cleanup failed")

    class _Strategy:
        def shutdown(self, *, restore_parked: bool = True) -> None:
            restore_permissions.append(restore_parked)

    lifecycle = online._OnlineRecipeLifecycle(
        placement_owner=None,
        strategy=_Strategy(),
        rollout_schedule=_FailingSchedule(),
        collector=object(),
    )
    await lifecycle.shutdown(run_error=RuntimeError("training failed"))

    assert restore_permissions == [False]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_launch_evidence_failure_stops_before_training_and_cleans_up(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    def fail_evidence(*args, **kwargs):
        raise OSError("evidence storage full")

    monkeypatch.setattr(online.TrainingRunTrace, "capture", fail_evidence)
    with pytest.raises(OSError, match="evidence storage full"):
        await online.run_online_recipe(run.cfg)
    assert state["trainer_steps"] == 0
    # The schedule already exists when evidence is captured, so the pipeline
    # is released through it, never the collector.
    assert state["shutdown_order"] == ["schedule", "owner"]


@pytest.mark.asyncio
async def test_artifact_sealing_runs_after_final_checkpoint_before_cleanup(monkeypatch, tmp_path):
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    sealed = []

    def seal(run_evidence):
        assert state["checkpoint_paths"][-1] == "checkpoint-final"
        # Nothing has been released yet when the artifacts are sealed.
        assert state["shutdown_order"] == []
        sealed.append(run_evidence.launch_path)
        return run_evidence.artifacts_path

    monkeypatch.setattr(online.TrainingRunTrace, "seal_artifacts", seal)
    await online.run_online_recipe(run.cfg)
    (evidence_path,) = sorted((run.output_dir / "run_evidence").glob("*.json"))
    assert sealed == [evidence_path]
    assert state["shutdown_order"] == ["schedule", "owner"]


@pytest.mark.asyncio
async def test_artifact_sealing_failure_still_cleans_up(monkeypatch, tmp_path):
    run = _RealRun(monkeypatch, tmp_path)
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)

    def fail_seal(path):
        raise OSError("artifact disk read failed")

    monkeypatch.setattr(online.TrainingRunTrace, "seal_artifacts", fail_seal)
    with pytest.raises(OSError, match="artifact disk read failed"):
        await online.run_online_recipe(run.cfg)
    assert state["shutdown_order"] == ["schedule", "owner"]


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
    state = _state()
    _install_ray_side_fakes(monkeypatch, tmp_path, state)
    _spy_replay_transformer_load(monkeypatch, after_load=inspect_build)
    try:
        for seed in (123, 456):
            torch.manual_seed(seed)
            with pytest.raises(ReachedModelBuild):
                await online.run_online_recipe(run.cfg)
        assert torch.equal(observed[0], observed[1])
    finally:
        restore_rng_state(previous)
