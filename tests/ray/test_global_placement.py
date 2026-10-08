"""Tests for the run-level role bundle plan and GlobalRayPlacementOwner.

The bundle-plan tests are pure (no Ray): they pin how a resolved resource plan
collapses to placement-group bundles for each supported GPU topology. The owner
tests at the bottom run against the package's shared real cluster
(``tests/ray/conftest.py``), which offers 8 CPUs and 4 logical GPUs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

from tests.rollouts.collector._helpers import Trace
from vrl.config.schema import parse_config
from vrl.ray import dependencies as ray_dependencies
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import BundleLayout, GlobalRayPlacementOwner, RolePlacement, _ProbeActor
from vrl.ray.resources import ResolvedDistributedResources


def _resolve(resources: dict):
    # Release scheduling is derived from topology, so only the resources block is
    # needed to pin a placement plan.
    return ResolvedDistributedResources.from_root(
        parse_config(OmegaConf.create({"distributed": {"resources": resources}})),
    )


# ----------------------------------------------------------------- bundle plan


def test_bundle_plan_groups_rollout_bundles_per_engine() -> None:
    """gpus_per_engine=2 over 4 rollout GPUs -> two consecutive bundle groups."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2, 3, 4],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1, 2, 3, 4], "gpus_per_engine": 2},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.rollout_gpus_per_engine == 2
    # The grouping rule lives on RolePlacement (the type that owns the bundles),
    # which is what the generation launcher calls to size its engine fleet.
    placement = RolePlacement(
        placement_group=None,
        bundle_indices=plan.rollout_bundle_indices,
        expected_gpu_ids=(),
    )
    groups = placement.engine_bundle_groups(plan.rollout_gpus_per_engine)
    assert len(groups) == 2
    assert all(len(group) == 2 for group in groups)
    assert tuple(index for group in groups for index in group) == plan.rollout_bundle_indices
    with pytest.raises(RuntimeError, match="not divisible"):
        placement.engine_bundle_groups(3)


def test_bundle_plan_dedicated_trainer_rollout_reward_distinct_bundles() -> None:
    """Trainer/rollout/reward on distinct GPUs => one bundle each, reward owned."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [2]},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.bundle_gpu_ids == (0, 1, 2)
    assert plan.rollout_bundle_indices == (1,)
    assert plan.reward_bundle_indices == (2,)
    assert set(plan.rollout_bundle_indices).isdisjoint(plan.reward_bundle_indices)
    assert plan.total_bundles == 3


def test_bundle_plan_multi_rollout_worker_one_bundle_per_gpu() -> None:
    """Pinned split: trainer reserved bundle + one rollout bundle per rollout GPU."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2, 3],
            "trainer": {"num_gpus": 1},
            "rollout": {"devices": [1, 2, 3], "num_engines": "auto"},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.bundle_gpu_ids == (0, 1, 2, 3)
    assert plan.rollout_bundle_indices == (1, 2, 3)
    assert plan.reward_bundle_indices == ()
    assert plan.total_bundles == 4


def test_bundle_plan_shared_reward_reuses_rollout_bundle() -> None:
    """Reward sharing the rollout GPU reuses the rollout bundle index."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [1]},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    # Reward bundle index == rollout bundle index (same physical GPU 1): that
    # overlap IS the "shared GPU" fact, read off the indices, no stored flag.
    assert plan.rollout_bundle_indices == plan.reward_bundle_indices
    assert plan.bundle_gpu_ids == (0, 1)
    assert plan.total_bundles == 2


def test_bundle_plan_colocated_debug_single_bundle_no_trainer_reservation() -> None:
    """Trainer+rollout share GPU 0 (debug): one bundle, no reserved trainer bundle."""
    resolved = _resolve(
        {
            "visible_devices": [0],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [0]},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.rollout_bundle_indices == (0,)
    assert plan.bundle_gpu_ids == (0,)
    assert plan.total_bundles == 1


def test_bundle_plan_cross_node_skips_trainer_reservation() -> None:
    """Cross-node: no trainer-reserved bundle (head node carries no Ray GPUs)."""
    resolved = _resolve(
        {
            "visible_devices": "auto",
            "cross_node": True,
            "trainer": {"num_gpus": 1},
            "rollout": {"num_gpus": 2, "num_engines": 2},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    # Rollout ordinals are budget tokens (1, 2) under cross_node.
    assert plan.rollout_bundle_indices == (0, 1)
    assert plan.bundle_gpu_ids == (1, 2)


def test_bundle_plan_cpu_only_rollout_uses_cpu_bundles() -> None:
    """CPU rollout: one CPU (None) bundle per worker, no GPU bundles."""
    resolved = _resolve(
        {
            "visible_devices": [],
            "trainer": {"num_gpus": 0},
            "rollout": {"num_gpus": 0, "num_engines": 2},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.bundle_gpu_ids == (None, None)
    assert plan.rollout_bundle_indices == (0, 1)
    assert plan.total_bundles == 2


def test_bundle_plan_dedicated_reward_appends_fresh_bundle() -> None:
    """A reward pinned to its own GPU appends a dedicated bundle."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [2]},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.reward_bundle_indices == (2,)
    assert plan.rollout_bundle_indices == (1,)
    assert set(plan.rollout_bundle_indices).isdisjoint(plan.reward_bundle_indices)


# ------------------------------------------------- probe-then-assign (no GPU)
#
# ``assign_roles`` is a pure function of a ``bundle_index -> gpu_id`` probe
# result, so the multi-GPU remapping logic is verified on hand-written probe
# maps without multi-GPU hardware.


@dataclass(frozen=True, slots=True)
class _WorkerCPUConfig:
    cpus_per_worker: float


def _worker(*, cpus_per_worker: float = 1.0) -> _WorkerCPUConfig:
    return _WorkerCPUConfig(cpus_per_worker=cpus_per_worker)


def _owner(
    resources: dict,
    *,
    worker: _WorkerCPUConfig | None = None,
) -> GlobalRayPlacementOwner:
    return GlobalRayPlacementOwner(_resolve(resources), worker or _worker())


def test_assign_roles_matches_requested_ordinals_under_permuted_probe() -> None:
    """Ray placed bundles on shuffled GPUs; roles still bind to their devices."""
    owner = _owner(
        {
            "visible_devices": [0, 1, 2],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [2]},
        },
    )
    # Plan bundles are [gpu0(trainer), gpu1(rollout), gpu2(reward)] but Ray put
    # them on physical GPUs 2,0,1 respectively.
    probed = {0: 2, 1: 0, 2: 1}
    roles = owner.assign_roles(probed)

    # rollout wants GPU 1 -> bundle 2; reward wants GPU 2 -> bundle 0.
    assert roles["rollout"] == (2,)
    assert roles["reward"] == (0,)
    requirements = owner._bundle_requirements()
    for bundle_index in roles["rollout"]:
        assert requirements[bundle_index]["CPU"] >= owner.rollout_worker.cpus_per_worker


def test_assign_roles_shared_reward_binds_same_bundle_as_rollout() -> None:
    """Shared reward resolves to the very bundle rollout uses (same GPU)."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [1]},
        },
    )
    probed = {0: 0, 1: 1}
    roles = owner.assign_roles(probed)

    assert roles["rollout"] == roles["reward"]


def test_assign_roles_raises_when_requested_gpu_absent_from_probe() -> None:
    """A rollout device missing from the probed PG is a hard error, not silent."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
        },
    )
    with pytest.raises(RuntimeError, match="rollout device GPU 1"):
        owner.assign_roles({0: 0})  # bundle for GPU 1 never reported


def test_assign_roles_cross_node_keeps_positional_bundles() -> None:
    """Cross-node ordinals are tokens: roles keep plan-positional bundles."""
    owner = _owner(
        {
            "visible_devices": "auto",
            "cross_node": True,
            "trainer": {"num_gpus": 1},
            "rollout": {"num_gpus": 2, "num_engines": 2},
        },
    )
    # Real remote GPU ids bear no relation to the token ordinals (1, 2).
    roles = owner.assign_roles({0: 7, 1: 3})
    assert roles["rollout"] == owner.layout.rollout_bundle_indices == (0, 1)


def test_assign_roles_rejects_duplicate_probed_gpu() -> None:
    """Two bundles on the same physical GPU is a placement error."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"num_gpus": 1},
            "rollout": {"num_gpus": 1, "num_engines": 1},
        },
    )
    with pytest.raises(RuntimeError, match="two bundles probed to GPU 0"):
        owner.assign_roles({0: 0, 1: 0})


def test_gpu_bundles_reserve_rollout_cpu_before_role_assignment() -> None:
    """Every interchangeable GPU bundle can host the rollout worker."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
            "reward": {"devices": [1]},
        },
        worker=_worker(cpus_per_worker=2.0),
    )
    requirements = owner._bundle_requirements()
    shared_bundle = owner.layout.rollout_bundle_indices[0]
    assert requirements[shared_bundle]["CPU"] == 2.0
    assert requirements[shared_bundle]["GPU"] == 1.0
    # Probing may swap the planned trainer and rollout bundle assignments.
    trainer_bundle = owner.layout.bundle_gpu_ids.index(owner.resources.trainer_devices[0])
    assert requirements[trainer_bundle] == {"CPU": 2.0, "GPU": 1.0}
    assert owner.required_local_cluster_cpus() == 4


def test_required_local_cluster_cpus_uses_placement_bundle_sum() -> None:
    """Fractional role bundles are summed and rounded once at node startup."""
    owner = _owner(
        {
            "visible_devices": [0],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [0], "num_engines": 1},
        },
        worker=_worker(cpus_per_worker=4.0),
    )

    # No CPU reward bundle: in-process CPU rewards run in the driver.
    assert owner._bundle_requirements() == [
        {"CPU": 4.0, "GPU": 1.0},
    ]
    assert owner.required_local_cluster_cpus() == 4


def test_required_local_cluster_cpus_rejects_invalid_bundle_cpu() -> None:
    owner = _owner(
        {
            "visible_devices": [0],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [0]},
        },
        worker=_worker(cpus_per_worker=0.0),
    )

    with pytest.raises(ValueError, match="finite and > 0"):
        owner.required_local_cluster_cpus()


def test_placement_owner_consumes_exact_rollout_cpu_capability() -> None:
    worker = _worker(cpus_per_worker=2.5)
    owner = _owner(
        {
            "visible_devices": [],
            "trainer": {"num_gpus": 0},
            "rollout": {"num_gpus": 0, "num_engines": 1},
        },
        worker=worker,
    )

    assert owner.rollout_worker is worker
    assert owner._bundle_requirements() == [{"CPU": 2.5}]


# ------------------------------------------- owner failure paths (real Ray)
#
# These run on the package's shared real cluster: real placement groups, real
# probe actors, real ``ray.get`` deadlines. A fault is either a real condition
# (an unsatisfiable bundle request, a probe actor that stalls) or a one-shot
# failure wrapped around a real call that otherwise delegates to it.


def _two_gpu_owner() -> GlobalRayPlacementOwner:
    return _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1]},
        },
    )


def _removal_fails_once(monkeypatch) -> list[object]:
    """The first removal reports failure and removes nothing; later ones are real."""

    from vrl.ray import placement as placement_module

    real_remove = placement_module.remove_placement_group
    calls: list[object] = []

    def remove(pg):
        calls.append(pg)
        if len(calls) == 1:
            return RuntimeError("placement remove failed")
        return real_remove(pg)

    monkeypatch.setattr(placement_module, "remove_placement_group", remove)
    return calls


def _removed(pg) -> bool:
    from ray.util.placement_group import placement_group_table

    return placement_group_table(pg)["state"] == "REMOVED"


def _probe_dead(ray, actor, *, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            ray.get(actor.gpus.remote(), timeout=5)
        except ray.exceptions.RayActorError:
            return True
        time.sleep(0.05)
    return False


class _StalledProbe(_ProbeActor):
    """The production probe actor body, stalled past the probe deadline."""

    def gpus(self) -> tuple[int, ...]:
        time.sleep(30.0)
        return super().gpus()


@pytest.mark.slow_test
def test_shutdown_retries_same_placement_group_after_remove_failure(
    local_ray, monkeypatch
) -> None:
    del local_ray
    owner = _two_gpu_owner()
    owner.create()
    placement_group = owner._placement_group
    calls = _removal_fails_once(monkeypatch)

    with pytest.raises(RuntimeError, match="placement remove failed"):
        owner.shutdown()
    assert owner._placement_group is placement_group
    assert not _removed(placement_group)

    owner.shutdown()
    assert calls == [placement_group, placement_group]
    assert owner._placement_group is None
    assert _removed(placement_group)


@pytest.mark.slow_test
def test_create_failure_retains_placement_for_cleanup_retry(local_ray, monkeypatch) -> None:
    del local_ray
    owner = _two_gpu_owner()
    probe = Trace(monkeypatch)
    probe.watch(GlobalRayPlacementOwner, "_probe_gpu_bundles", "probe")
    probe.fail("probe", "probe failed")
    calls = _removal_fails_once(monkeypatch)

    with pytest.raises(RuntimeError, match="probe failed") as caught:
        owner.create()
    assert any("retained the handle" in note for note in caught.value.__notes__)
    placement_group = owner._placement_group
    assert placement_group is not None
    assert owner._placement_ready is False

    with pytest.raises(RuntimeError, match="cleanup is still pending"):
        owner.create()
    owner.shutdown()

    assert calls == [placement_group, placement_group]
    assert owner._placement_group is None
    assert _removed(placement_group)


@pytest.mark.slow_test
def test_ready_failure_retains_exact_placement_for_shutdown_retry(local_ray, monkeypatch) -> None:
    # Five GPU bundles on a cluster that advertises four: never schedulable.
    owner = _owner(
        {
            "visible_devices": [0, 1, 2, 3, 4],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1, 2, 3, 4]},
        },
    )
    monkeypatch.setattr("vrl.ray.placement._PLACEMENT_READY_TIMEOUT_S", 1.0)
    calls = _removal_fails_once(monkeypatch)

    with pytest.raises(RuntimeError, match="placement group not ready") as caught:
        owner.create()

    assert isinstance(caught.value.__cause__, local_ray.exceptions.GetTimeoutError)
    assert any("retained the handle" in note for note in caught.value.__notes__)
    placement_group = owner._placement_group
    assert placement_group is not None
    assert owner._placement_ready is False

    owner.shutdown()

    assert calls == [placement_group, placement_group]
    assert owner._placement_group is None
    assert _removed(placement_group)


@pytest.mark.slow_test
def test_probe_partial_actor_construction_cleans_created_handles(local_ray, monkeypatch) -> None:
    """Probe fan-out is all-or-nothing: when actor 2 of 2 fails to construct,
    actor 1 must be killed rather than left holding a bundle."""

    from vrl.ray import placement as placement_module

    ray = local_ray
    owner = _two_gpu_owner()
    pg = placement_module._create_raw_placement_group(
        owner._bundle_requirements(), strategy="PACK"
    )
    real_strategy = placement_module.actor_scheduling_strategy
    real_kill = placement_module.kill_actors
    strategies = 0
    killed: list[object] = []

    def strategy(*args, **kwargs):
        nonlocal strategies
        strategies += 1
        if strategies == 2:
            raise RuntimeError("probe actor construction failed")
        return real_strategy(*args, **kwargs)

    def kill(ray_api, actors):
        killed.extend(actors)
        return real_kill(ray_api, actors)

    monkeypatch.setattr(placement_module, "actor_scheduling_strategy", strategy)
    monkeypatch.setattr(placement_module, "kill_actors", kill)
    try:
        ray.get(pg.ready(), timeout=30)
        with pytest.raises(RuntimeError, match="probe actor construction failed"):
            owner._probe_gpu_bundles(ray, pg)

        assert len(killed) == 1
        assert _probe_dead(ray, killed[0])
    finally:
        placement_module.remove_placement_group(pg)


@pytest.mark.slow_test
def test_probe_timeout_cancels_refs_kills_actors_and_removes_placement(
    local_ray, monkeypatch
) -> None:
    from vrl.ray import placement as placement_module

    ray = local_ray
    owner = _two_gpu_owner()
    monkeypatch.setattr(placement_module, "_ProbeActor", _StalledProbe)
    monkeypatch.setattr(placement_module, "_PLACEMENT_READY_TIMEOUT_S", 3.0)
    real_cancel = ray.cancel
    real_kill = placement_module.kill_actors
    real_remove = placement_module.remove_placement_group
    cancelled: list[tuple[object, bool]] = []
    killed: list[object] = []
    removed: list[object] = []

    def cancel(ref, *, force=False, **kwargs):
        cancelled.append((ref, force))
        return real_cancel(ref, force=force, **kwargs)

    def kill(ray_api, actors):
        killed.extend(actors)
        return real_kill(ray_api, actors)

    def remove(pg):
        removed.append(pg)
        return real_remove(pg)

    monkeypatch.setattr(ray, "cancel", cancel)
    monkeypatch.setattr(placement_module, "kill_actors", kill)
    monkeypatch.setattr(placement_module, "remove_placement_group", remove)

    with pytest.raises(RayOperationTimeout, match=r"placement\.gpu_metadata_probe"):
        owner.create()

    # One stalled probe call per GPU bundle, each cancelled without force.
    assert len(cancelled) == 2
    assert all(isinstance(ref, ray.ObjectRef) and force is False for ref, force in cancelled)
    assert len(killed) == 2
    assert all(_probe_dead(ray, actor) for actor in killed)
    (placement_group,) = removed
    assert _removed(placement_group)
    assert owner._placement_group is None
    assert owner._placement_ready is False


# ----------------------------------------------- simulated multi-GPU (real Ray)
#
# Ray's num_gpus is logical accounting, so a single-physical-GPU host can still
# exercise the owner's real multi-bundle placement: probe actors only call
# ray.get_gpu_ids() (logical ids), never real CUDA. That is why the shared cluster
# can offer 4 GPUs on this host at all -- see real_local_ray in tests/conftest.py.
# This validates the part that matters most -- trainer-GPU reservation and
# role->GPU binding under a live PG.


@pytest.mark.slow_test
def test_owner_reserves_trainer_gpu_and_binds_roles_on_simulated_gpus(local_ray) -> None:
    """3-GPU dedicated plan: trainer GPU stays empty, rollout/reward bind right."""
    ray = local_ray
    owner = GlobalRayPlacementOwner(
        _resolve(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1]},
                "reward": {"devices": [2]},
            },
        ),
        _worker(),
    )
    try:
        owner.create()
        probed = owner._probe_gpu_bundles(ray, owner._placement_group)
        rollout = owner.rollout_placement
        reward = owner.reward_placement
        # Rollout/reward actually land on their requested GPUs.
        assert probed[rollout.bundle_indices[0]] == 1
        assert probed[reward.bundle_indices[0]] == 2
        assert reward.expected_gpu_ids == (2,)
        # The bundle on GPU 0 (the trainer) is held by no role -> reserved empty.
        trainer_bundle = next(b for b, g in probed.items() if g == 0)
        used = set(rollout.bundle_indices) | set(reward.bundle_indices)
        assert trainer_bundle not in used
    finally:
        # The cluster is shared: release the bundles, never the cluster.
        owner.shutdown()


@pytest.mark.slow_test
@pytest.mark.parametrize("cpus_per_worker", [1.0, 0.0005])
def test_owner_shares_one_bundle_for_rollout_and_reward_on_simulated_gpus(
    local_ray, monkeypatch, cpus_per_worker
) -> None:
    """Shared reward time-multiplexes the rollout GPU: one bundle, both roles."""
    ray = local_ray
    monkeypatch.setattr("vrl.ray.placement._PLACEMENT_READY_TIMEOUT_S", 5.0)
    owner = GlobalRayPlacementOwner(
        _resolve(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1]},
                "reward": {"devices": [1]},
            },
        ),
        _worker(cpus_per_worker=cpus_per_worker),
    )
    try:
        owner.create()
        rollout = owner.rollout_placement
        reward = owner.reward_placement
        assert reward is not None
        assert rollout.bundle_indices == reward.bundle_indices
        probed = owner._probe_gpu_bundles(ray, owner._placement_group)
        assert probed[rollout.bundle_indices[0]] == 1
    finally:
        # The cluster is shared: release the bundles, never the cluster.
        owner.shutdown()


@pytest.mark.slow_test
def test_probe_actor_kill_failure_is_a_create_failure(local_ray, monkeypatch) -> None:
    """A probe actor that cannot be killed fails create(), removes the placement
    group anyway, and releases ownership.

    Only the kill OUTCOME is injected -- a healthy cluster will not fail a
    ``ray.kill`` on demand. Everything else on the asserted path is real: a real
    placement group, a real ``_ProbeActor`` scheduled into a real bundle, a real
    ``pg.ready()``, and the real ``remove_placement_group``. That is also why the
    old ``remove_calls == [placement_group]`` assertion is gone -- real removal
    keeps no ledger, and ``_placement_group is None`` is the same invariant read
    off production state instead of off a double.
    """

    del local_ray  # the owner reaches Ray through require_ray(); the cluster is the fixture
    owner = _owner(
        {
            "visible_devices": [0],
            "trainer": {"num_gpus": 0},
            "rollout": {"devices": [0]},
        },
    )
    cleanup_error = RuntimeError("probe kill failed")

    def failing_kill(ray, actors):
        # Kill them for real first, then report the failure: an abandoned probe
        # actor would hold a bundle of the shared cluster for the rest of the run.
        ray_dependencies.kill_actors(ray, actors)
        return [(actors[0], cleanup_error)]

    monkeypatch.setattr("vrl.ray.placement.kill_actors", failing_kill)

    with pytest.raises(RuntimeError, match="probe actor cleanup incomplete") as caught:
        owner.create()

    assert caught.value.__cause__ is cleanup_error
    assert owner._placement_group is None
    assert owner._placement_ready is False
