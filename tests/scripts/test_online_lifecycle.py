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
        self.last_components: dict[str, list[float]] = {}

    def reset_components(self) -> None:
        self._state["reward_resets"] += 1

    async def shutdown(self) -> None:
        self._state["reward_shutdowns"] += 1
        self._state["shutdown_order"].append("reward")


class _FakeCollector:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self._runtime: Any | None = None

    def set_runtime(self, runtime: Any) -> None:
        self._runtime = runtime
        self._state["collector_set_runtime"] += 1

    @property
    def runtime(self) -> Any:
        return self._runtime

    async def shutdown(self) -> None:
        self._state["collector_shutdowns"] += 1
        self._state["shutdown_order"].append("collector")
        if self._state.get("collector_shutdown_raises"):
            raise RuntimeError("collector shutdown boom")
        shutdown = getattr(self._runtime, "shutdown", None)
        if shutdown is not None:
            await shutdown()


class _FakeRuntime:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def shutdown(self) -> None:
        self._state["runtime_shutdowns"] += 1
        self._state["shutdown_order"].append("runtime")


class _FakeSchedule:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def shutdown(self) -> None:
        self._state["schedule_shutdowns"] += 1
        self._state["shutdown_order"].append("schedule")


class _FakeLauncher:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

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


class _FakeRecipeRay:
    __version__ = "test-ray"

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def init(self, **kwargs: Any) -> Any:
        self._initialized = True
        self._state["ray_init_calls"].append(dict(kwargs))
        return SimpleNamespace(address_info={"session_dir": "/tmp/test-ray"})

    def shutdown(self) -> None:
        self._initialized = False
        self._state["ray_shutdowns"] += 1
        self._state["shutdown_order"].append("ray")


class _FakeTrainer:
    def __init__(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._state = state
        self.state = SimpleNamespace(global_step=0)
        self.rollout_schedule = _FakeSchedule(state)

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
        "reward_resets": 0,
        "reward_shutdowns": 0,
        "owner_creates": 0,
        "owner_shutdowns": 0,
        "launches": 0,
        "trainer_steps": 0,
        "checkpoint_paths": [],
        "shutdown_order": [],
        "ray_init_calls": [],
        "ray_shutdowns": 0,
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
    )


def _cfg() -> Any:
    return OmegaConf.create(
        {
            "data": {"sampler": {"type": "random_without_replacement"}},
            "distributed": {"rollout": {"cpus_per_worker": 0.5}},
            "algorithm": {"kl_coef": 0.0},
            "model": {"use_lora": False},
        },
    )


def _definition() -> OnlineRecipeDefinition:
    return OnlineRecipeDefinition(
        family="sd3_5",
        build_bundle=lambda cfg, device, weight_dtype: SimpleNamespace(
            model=_FakeModel(),
            scheduler=object(),
            trainable_modules={},
        ),
    )


def _install_common_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    state: dict[str, Any],
) -> _FakeReward:
    trainer_config = _trainer_config(tmp_path)
    reward = _FakeReward(state)
    collector = _FakeCollector(state)
    resources = SimpleNamespace(cross_node=False)

    monkeypatch.setattr(online, "_preflight_production_video_reward", lambda cfg: None)
    monkeypatch.setattr(online, "build_configs", lambda cfg: {"trainer": trainer_config})
    monkeypatch.setattr(online, "_apply_precision_policy", lambda cfg, trainer: None)
    monkeypatch.setattr(online, "load_training_checkpoint_from_config", lambda cfg: None)
    monkeypatch.setattr(
        online,
        "prepare_model_config_for_training_resume",
        lambda cfg, checkpoint, *, strict: None,
    )
    monkeypatch.setattr(online, "resolve_distributed_resources", lambda cfg, **kwargs: resources)
    monkeypatch.setattr(online, "format_distributed_resource_plan", lambda resources: "resources")
    monkeypatch.setattr(online, "trainer_torch_device", lambda resources: "cpu")
    monkeypatch.setattr(online, "torch_dtype_for_trainer_precision", lambda trainer, torch: torch.float32)
    monkeypatch.setattr(
        online,
        "resolve_precision_policy",
        lambda cfg: SimpleNamespace(rollout="float32"),
    )
    monkeypatch.setattr(online, "load_prompt_examples_from_config", lambda cfg: ["prompt"])
    monkeypatch.setattr(online, "log_host_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        online,
        "GlobalRayPlacementOwner",
        lambda *args, **kwargs: _FakePlacementOwner(state, *args, **kwargs),
    )
    fake_ray = _FakeRecipeRay(state)
    monkeypatch.setattr(online, "require_ray", lambda: fake_ray)
    monkeypatch.setattr(
        online,
        "build_online_recipe_components",
        lambda *args, **kwargs: SimpleNamespace(
            built={"reward": ({"kling_video_reward": 1.0}, {})},
            trainer_config=trainer_config,
            family="sd3_5",
            family_entry="sd3_5",
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
    monkeypatch.setattr(
        online,
        "build_ray_generation_inputs_for_family",
        lambda *args, **kwargs: SimpleNamespace(
            launch_contract=object(),
            gatherer=object(),
        ),
    )
    monkeypatch.setattr(online, "RayGenerationLauncher", lambda: _FakeLauncher(state))
    monkeypatch.setattr(online, "OnlineTrainer", lambda *args, **kwargs: _FakeTrainer(state, *args, **kwargs))
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


@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_success(monkeypatch, tmp_path) -> None:
    state = _state()
    _install_common_fakes(monkeypatch, tmp_path, state)

    await online.run_online_recipe(_cfg(), _definition())

    assert state["owner_creates"] == 1
    assert state["trainer_steps"] == 1
    assert state["checkpoint_paths"] == ["checkpoint-final"]
    assert state["collector_shutdowns"] == 1
    assert state["runtime_shutdowns"] == 1
    assert state["schedule_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1
    assert state["ray_init_calls"] == [{"address": "local"}]
    assert state["ray_shutdowns"] == 1
    assert state["shutdown_order"] == [
        "schedule",
        "collector",
        "runtime",
        "reward",
        "owner",
        "ray",
    ]


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
async def test_rollout_sync_getter_routes_through_strategy(monkeypatch, tmp_path) -> None:
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


@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_create_failure(monkeypatch, tmp_path) -> None:
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


@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_rollout_launch_failure(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reward", "collector"])
async def test_run_online_recipe_shutdowns_owner_after_component_build_failure(
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


@pytest.mark.asyncio
async def test_run_online_recipe_shutdowns_owner_after_final_checkpoint_failure(
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


@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_do_not_hide_training_error(
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

    assert state["collector_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1


@pytest.mark.asyncio
async def test_run_online_recipe_shutdown_errors_after_success_run_all_cleanups(
    monkeypatch,
    tmp_path,
) -> None:
    state = _state()
    state["collector_shutdown_raises"] = True
    _install_common_fakes(monkeypatch, tmp_path, state)

    with pytest.raises(RuntimeError, match="collector shutdown failed"):
        await online.run_online_recipe(_cfg(), _definition())

    assert state["collector_shutdowns"] == 1
    assert state["reward_shutdowns"] == 1
    assert state["owner_shutdowns"] == 1
