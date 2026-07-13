from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.models.interfaces import ReplayResult
from vrl.scripts.common import online
from vrl.scripts.common.types import OnlineRecipeDefinition

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _module_ray_teardown():
    """Leave the shared cluster up between tests, close it when the module ends."""
    yield
    ray.shutdown()


@pytest.fixture()
def preinitialized_ray():
    """Shared real local Ray cluster. The recipe sees an is_initialized()
    driver, takes the embedding-caller ownership branch, and must leave the
    cluster running — asserted per-test via ray.is_initialized()."""
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
        )
    yield ray


class _FakeModel:
    device = torch.device("cpu")

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: Any | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        return ReplayResult(log_probs=torch.zeros(1), metadata={})

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        self.state_dict = dict(state_dict)


class _FakeReward:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def preflight(self) -> None:
        return None

    async def shutdown(self) -> None:
        self._state["reward_shutdowns"] += 1
        self._state["shutdown_order"].append("reward")


class _FakeCollector:
    def __init__(self, state: dict[str, Any], reward: _FakeReward) -> None:
        self._state = state
        self._reward = reward
        self._runtime: Any | None = None
        self._runtime_shutdown_complete = False
        self._reward_shutdown_complete = False

    def set_runtime(self, runtime: Any) -> None:
        self._runtime = runtime
        self._state["collector_set_runtime"] += 1

    @property
    def runtime(self) -> Any:
        return self._runtime

    async def shutdown(self) -> None:
        self._state["collector_shutdowns"] += 1
        self._state["shutdown_order"].append("collector")
        if not self._runtime_shutdown_complete:
            shutdown = getattr(self._runtime, "shutdown", None)
            if shutdown is not None:
                await shutdown()
            self._runtime_shutdown_complete = True
        if not self._reward_shutdown_complete:
            await self._reward.shutdown()
            self._reward_shutdown_complete = True
        if self._state.get("collector_shutdown_raises"):
            raise RuntimeError("collector shutdown boom")


class _FakeRuntime:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def shutdown(self) -> None:
        self._state["runtime_shutdowns"] += 1
        self._state["shutdown_order"].append("runtime")


class _FakeSchedule:
    def __init__(self, state: dict[str, Any], collector: Any) -> None:
        self._state = state
        self._collector = collector

    async def shutdown(self) -> None:
        self._state["schedule_shutdowns"] += 1
        self._state["shutdown_order"].append("schedule")
        await self._collector.shutdown()


class _FakeLauncher:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def build_inputs(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            launch_contract=object(),
            gatherer=object(),
        )

    def launch_from_cfg(self, *args: Any, **kwargs: Any) -> _FakeRuntime:
        del args, kwargs
        self._state["launches"] += 1
        if self._state.get("launch_raises"):
            raise RuntimeError("launch boom")
        return _FakeRuntime(self._state)


class _FakePlacementOwner:
    def __init__(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._state = state
        self.rollout_placement = object()
        self.reward_placement = object()

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
        self.rollout_schedule = _FakeSchedule(state, kwargs["collector"])

    async def step(self, example_batch: list[Any]) -> Any:
        del example_batch
        self._state["trainer_steps"] += 1
        if self._state.get("trainer_step_raises"):
            raise RuntimeError("train boom")
        self.state.global_step += 1
        return SimpleNamespace()


def _state() -> dict[str, Any]:
    return {
        "collector_set_runtime": 0,
        "collector_shutdowns": 0,
        "runtime_shutdowns": 0,
        "schedule_shutdowns": 0,
        "reward_shutdowns": 0,
        "owner_creates": 0,
        "owner_shutdowns": 0,
        "launches": 0,
        "trainer_steps": 0,
        "checkpoint_paths": [],
        "shutdown_order": [],
    }


def _trainer_config(tmp_path: Any) -> SimpleNamespace:
    return SimpleNamespace(
        profile=False,
        resume_strict=True,
        output_dir=str(tmp_path),
        total_epochs=1,
        seed=0,
        prompts_per_batch=1,
        n_samples_per_prompt=1,
        save_freq=0,
        rollout_orchestration=SimpleNamespace(schedule_mode="strict_on_policy"),
    )


def _cfg() -> Any:
    return OmegaConf.create(
        {
            "data": {"sampler": {"type": "random_without_replacement"}},
            "distributed": {"rollout": {"cpus_per_worker": 0.5}},
            "algorithm": {"kl_coef": 0.0},
            "model": {"family": "sd3_5", "use_lora": False},
        },
    )


def _definition() -> OnlineRecipeDefinition:
    return OnlineRecipeDefinition(
        build_replay_bundle=lambda build: SimpleNamespace(
            model=_FakeModel(),
            scheduler=object(),
            trainable_modules={},
        ),
    )


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


def _install_common_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    state: dict[str, Any],
) -> _FakeReward:
    trainer_config = _trainer_config(tmp_path)
    reward = _FakeReward(state)
    collector = _FakeCollector(state, reward)
    resources = SimpleNamespace(
        cross_node=False,
        lifecycle=SimpleNamespace(
            handoff=SimpleNamespace(
                release_rollout_before_train=False,
                release_rollout_before_reward=False,
                release_trainer_before_reward=False,
                release_reward_after_score=False,
            ),
        ),
    )

    monkeypatch.setattr(online, "_preflight_production_video_reward", lambda cfg: None)
    precision = SimpleNamespace(
        rollout="float32",
        rollout_base_precision="float32",
    )
    monkeypatch.setattr(
        online,
        "build_configs",
        lambda cfg: {
            "trainer": trainer_config,
            "precision": precision,
            "reward": ({"kling_video_reward": 1.0}, {}),
        },
    )
    monkeypatch.setattr(online, "load_training_checkpoint_from_config", lambda cfg: None)
    monkeypatch.setattr(
        online,
        "prepare_model_config_for_training_resume",
        lambda cfg, checkpoint, *, strict: None,
    )
    monkeypatch.setattr(online, "resolve_distributed_resources", lambda cfg, **kwargs: resources)
    monkeypatch.setattr(online, "format_distributed_resource_plan", lambda resources: "resources")
    monkeypatch.setattr(online, "trainer_torch_device", lambda resources: "cpu")
    monkeypatch.setattr(
        online,
        "reward_torch_device",
        lambda resources, *, trainer_device: "cpu",
    )
    monkeypatch.setattr(online, "load_prompt_examples_from_config", lambda cfg: ["prompt"])
    monkeypatch.setattr(
        online,
        "resolve_entry_model_build",
        lambda entry, cfg, device, **kwargs: SimpleNamespace(
            family=str(cfg.model.family),
        ),
    )
    monkeypatch.setattr(online, "log_host_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        online,
        "GlobalRayPlacementOwner",
        lambda *args, **kwargs: _FakePlacementOwner(state, *args, **kwargs),
    )
    monkeypatch.setattr(
        online,
        "build_online_recipe_components",
        lambda *args, **kwargs: SimpleNamespace(
            collector_config=object(),
            reward_fn=reward,
            algorithm=object(),
            evaluator=None,
        ),
    )
    monkeypatch.setattr(
        online,
        "build_collector_from_cfg",
        lambda *args, **kwargs: collector,
    )
    monkeypatch.setattr(online, "RayGenerationLauncher", lambda: _FakeLauncher(state))
    monkeypatch.setattr(
        online, "OnlineTrainer", lambda *args, **kwargs: _FakeTrainer(state, *args, **kwargs)
    )
    monkeypatch.setattr(online, "build_runtime_weight_syncer", lambda *args, **kwargs: object())
    monkeypatch.setattr(online, "save_resolved_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        online.OnlineRecipeRun,
        "prepare_metrics_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(online, "sample_prompt_indices", lambda *args, **kwargs: [0])
    monkeypatch.setattr(online.OnlineRecipeRun, "write_metric_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        online.OnlineRecipeRun,
        "save_checkpoint",
        lambda self, path, *args, **kwargs: state["checkpoint_paths"].append(path.name),
    )
    return reward


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_success(monkeypatch, tmp_path) -> None:
    state = _state()
    _install_common_fakes(monkeypatch, tmp_path, state)
    # No preexisting driver: the recipe must start an owned local cluster and
    # close it on exit. Real Ray cannot append to the shutdown_order ledger, so
    # the post-run is_initialized() check stands in for the trailing "ray" entry.
    ray.shutdown()

    await online.run_online_recipe(_cfg(), _definition())

    assert state["owner_creates"] == 1
    assert state["trainer_steps"] == 1
    assert state["checkpoint_paths"] == ["checkpoint-final"]
    assert state["collector_shutdowns"] == 1
    assert state["runtime_shutdowns"] == 1
    assert state["schedule_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1
    assert state["shutdown_order"] == [
        "schedule",
        "collector",
        "runtime",
        "reward",
        "owner",
    ]
    assert not ray.is_initialized()


def test_require_supported_online_strategy_allows_fsdp() -> None:
    """fsdp uses the same per-rank-local symmetric-colocated path as ddp."""
    from vrl.trainers.distributed import DistributedTrainingContext

    ctx = DistributedTrainingContext(
        strategy="fsdp",
        distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        is_primary=True,
        device=torch.device("cpu"),
    )
    online._require_supported_online_strategy(ctx)  # no raise


def test_require_supported_online_strategy_allows_single_process() -> None:
    from vrl.trainers.distributed import DistributedTrainingContext

    ctx = DistributedTrainingContext(
        strategy="single_process",
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )
    online._require_supported_online_strategy(ctx)  # no raise


def test_require_supported_online_strategy_allows_ddp() -> None:
    """ddp is the supported per-rank-local symmetric-colocated path (each rank runs
    its own local Ray + colocated rollout on its GPU; only grads all-reduce)."""
    from vrl.trainers.distributed import DistributedTrainingContext

    ctx = DistributedTrainingContext(
        strategy="ddp",
        distributed=True,
        rank=1,
        local_rank=0,
        world_size=2,
        is_primary=False,
        device=torch.device("cuda:0"),
    )
    online._require_supported_online_strategy(ctx)  # no raise


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("release_rollout_before_train", "release_trainer_before_reward"),
    [(True, False), (False, True)],
)
async def test_shared_gpu_parking_capability_fails_before_model_or_ray_launch(
    monkeypatch,
    tmp_path,
    release_rollout_before_train,
    release_trainer_before_reward,
) -> None:
    state = _state()
    _install_common_fakes(monkeypatch, tmp_path, state)
    resources = SimpleNamespace(
        cross_node=False,
        lifecycle=SimpleNamespace(
            handoff=SimpleNamespace(
                release_rollout_before_train=release_rollout_before_train,
                release_rollout_before_reward=False,
                release_trainer_before_reward=release_trainer_before_reward,
                release_reward_after_score=release_trainer_before_reward,
            ),
        ),
    )
    monkeypatch.setattr(online, "resolve_distributed_resources", lambda _cfg: resources)
    monkeypatch.setattr(
        online,
        "validate_reward_memory_parking_from_cfg",
        lambda *_args, **_kwargs: None,
    )

    class _UnsupportedStrategy:
        def validate_training_state_parking(self) -> None:
            raise NotImplementedError("Use disjoint rollout GPUs")

    monkeypatch.setattr(online, "build_strategy", lambda _cfg, _context: _UnsupportedStrategy())
    model_builds = 0

    def _build(*_args, **_kwargs):
        nonlocal model_builds
        model_builds += 1
        return SimpleNamespace(model=_FakeModel(), scheduler=object(), trainable_modules={})

    definition = OnlineRecipeDefinition(build_replay_bundle=_build)

    with pytest.raises(NotImplementedError, match="Use disjoint rollout GPUs"):
        await online.run_online_recipe(_cfg(), definition)

    assert model_builds == 0
    assert state["owner_creates"] == 0
    assert state["launches"] == 0


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_rollout_sync_getter_routes_through_strategy(
    preinitialized_ray, monkeypatch, tmp_path
) -> None:
    """The recipe binds the rollout sync getter to the strategy, not the raw helper.

    Locks sprint P3 ownership: the strategy seam -- not a direct
    ``build_trainable_state_sync_getter(bundle)`` call -- is what produces
    rollout-facing weights, so the future FSDP strategy controls what leaves the
    trainer without the recipe changing.
    """
    state = _state()
    _install_common_fakes(monkeypatch, tmp_path, state)

    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> _FakeTrainer:
        captured.update(kwargs)
        return _FakeTrainer(state, *args, **kwargs)

    monkeypatch.setattr(online, "OnlineTrainer", _capture)

    await online.run_online_recipe(_cfg(), _definition())

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
    state = _state()
    state["owner_create_raises"] = True
    _install_common_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="owner create boom"):
        await online.run_online_recipe(_cfg(), _definition())

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
    state = _state()
    state["launch_raises"] = True
    _install_common_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="launch boom"):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["owner_creates"] == 1
    assert state["collector_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1


@pytest.mark.slow_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reward", "collector"])
async def test_run_online_recipe_shutdowns_owner_after_component_build_failure(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
    failure,
) -> None:
    state = _state()
    reward = _install_common_fakes(monkeypatch, tmp_path, state)
    if failure == "reward":
        monkeypatch.setattr(
            online,
            "build_online_recipe_components",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reward build boom")),
        )
        message = "reward build boom"
    else:
        monkeypatch.setattr(
            online,
            "build_collector_from_cfg",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("collector build boom")),
        )
        message = "collector build boom"

    with pytest.raises(RuntimeError, match=message):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["owner_creates"] == 1
    assert state["owner_shutdowns"] == 1
    assert state["collector_shutdowns"] == 0
    assert state["reward_shutdowns"] == (0 if failure == "reward" else 1)
    assert reward is not None


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_final_checkpoint_failure(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    state = _state()
    _install_common_fakes(monkeypatch, tmp_path, state)

    def raise_checkpoint(self: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        del self, path, args, kwargs
        raise RuntimeError("save boom")

    monkeypatch.setattr(online.OnlineRecipeRun, "save_checkpoint", raise_checkpoint)

    with pytest.raises(RuntimeError, match="save boom"):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["collector_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_do_not_hide_training_error(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    state = _state()
    state["trainer_step_raises"] = True
    state["collector_shutdown_raises"] = True
    state["owner_shutdown_raises"] = True
    _install_common_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="train boom"):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["collector_shutdowns"] == 2
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 2


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_after_success_run_all_cleanups(
    preinitialized_ray,
    monkeypatch,
    tmp_path,
) -> None:
    state = _state()
    state["collector_shutdown_raises"] = True
    _install_common_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="rollout_schedule shutdown failed"):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["collector_shutdowns"] == 2
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1


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

    await online._shutdown_online_recipe_runtime(
        rollout_schedule=_Schedule(),
        collector=collector,
        reward_fn=_StandaloneReward(),
        placement_owner=None,
        strategy=_Strategy(),
        ray_session=None,
        run_error=None,
    )

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

    await online._shutdown_online_recipe_runtime(
        rollout_schedule=None,
        collector=None,
        reward_fn=None,
        placement_owner=placement,
        strategy=None,
        ray_session=ray_session,
        run_error=None,
    )

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

    await online._shutdown_online_recipe_runtime(
        rollout_schedule=_FailingSchedule(),
        collector=object(),
        reward_fn=None,
        placement_owner=None,
        strategy=_Strategy(),
        ray_session=None,
        run_error=RuntimeError("training failed"),
    )

    assert restore_permissions == [False]
