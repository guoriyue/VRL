"""Tests for the run-level role bundle plan and GlobalRayPlacementOwner.

The bundle-plan tests are pure (no Ray): they pin how a resolved resource plan
collapses to placement-group bundles for each supported GPU topology. The owner
tests at the bottom run against the package's shared real cluster
(``tests/ray/conftest.py``), which offers 8 CPUs and 4 logical GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

from vrl.ray import dependencies as ray_dependencies
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import BundleLayout, GlobalRayPlacementOwner
from vrl.ray.resources import resolve_distributed_resources

# The retry tests below all inject the same class of Ray failure for the same
# reason, so the label is a module constant rather than the same string three
# times (the `requires_fp8` precedent in tests/nn/quantization/test_fp8.py).
_REAL_RAY_PLACEMENT = pytest.mark.real_cover(
    "tests/ray/test_global_placement.py"
    "::test_owner_reserves_trainer_gpu_and_binds_roles_on_simulated_gpus",
    why=(
        "a live Ray cluster cannot be told to fail remove_placement_group or pg.ready() on "
        "demand, and what these tests assert is that the handle survives that failure for a "
        "later retry; the same create/probe/shutdown path against a real cluster is the "
        "slow_test twin below"
    ),
)
_REAL_RAY_PROBE_TIMEOUT = pytest.mark.real_cover(
    "tests/ray/test_global_placement.py"
    "::test_owner_reserves_trainer_gpu_and_binds_roles_on_simulated_gpus",
    why=(
        "a live cluster cannot deterministically stall only the metadata probe; "
        "the real twin drives the same probe actors and placement-group boundary, "
        "while this test injects timeout and records cancellation/cleanup"
    ),
)


def _resolve(resources: dict):
    # Release scheduling is derived from topology, so only the resources block is
    # needed to pin a placement plan.
    return resolve_distributed_resources(
        OmegaConf.create({"distributed": {"resources": resources}}),
    )


# ----------------------------------------------------------------- bundle plan


def test_bundle_plan_dedicated_trainer_rollout_reward_distinct_bundles() -> None:
    """Trainer/rollout/reward on distinct GPUs => one bundle each, reward owned."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu", "devices": [2]},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.bundle_gpu_ids == (0, 1, 2)
    assert plan.rollout_bundle_indices == (1,)
    assert plan.reward_bundle_indices == (2,)
    assert set(plan.rollout_bundle_indices).isdisjoint(plan.reward_bundle_indices)
    assert plan.total_bundles == 3


def test_bundle_plan_multi_rollout_worker_one_bundle_per_gpu() -> None:
    """Auto split: trainer reserved bundle + one rollout bundle per rollout GPU."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2, 3],
            "trainer": {"num_gpus": 1},
            "rollout": {"num_gpus": "auto", "gpus_per_worker": 1, "num_workers": "auto"},
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
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu", "gpu_pool": "rollout"},
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
            "rollout": {"devices": [0], "gpus_per_worker": 1},
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
            "rollout": {"num_gpus": 2, "gpus_per_worker": 1, "num_workers": 2},
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
            "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 2},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.bundle_gpu_ids == (None, None)
    assert plan.rollout_bundle_indices == (0, 1)
    assert plan.total_bundles == 2


def test_bundle_plan_dedicated_reward_appends_fresh_bundle() -> None:
    """Auto reward placement onto a spare GPU appends a dedicated bundle."""
    resolved = _resolve(
        {
            "visible_devices": [0, 1, 2],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu"},
        },
    )
    plan = BundleLayout.from_resources(resolved)

    assert plan.reward_bundle_indices == (2,)
    assert plan.rollout_bundle_indices == (1,)
    assert set(plan.rollout_bundle_indices).isdisjoint(plan.reward_bundle_indices)


# ------------------------------------------------- probe-then-assign (no GPU)
#
# These inject a fake bundle_index -> gpu_id probe so the multi-GPU remapping
# logic is verified deterministically without multi-GPU hardware.


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
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu", "devices": [2]},
        },
    )
    # Plan bundles are [gpu0(trainer), gpu1(rollout), gpu2(reward)] but Ray put
    # them on physical GPUs 2,0,1 respectively.
    probed = {0: 2, 1: 0, 2: 1}
    roles = owner.assign_roles(probed)

    # rollout wants GPU 1 -> bundle 2; reward wants GPU 2 -> bundle 0.
    assert roles["rollout"] == (2,)
    assert roles["reward"] == (0,)


def test_assign_roles_shared_reward_binds_same_bundle_as_rollout() -> None:
    """Shared reward resolves to the very bundle rollout uses (same GPU)."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu", "gpu_pool": "rollout"},
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
            "rollout": {"devices": [1], "gpus_per_worker": 1},
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
            "rollout": {"num_gpus": 2, "gpus_per_worker": 1, "num_workers": 2},
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
            "rollout": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
        },
    )
    with pytest.raises(RuntimeError, match="two bundles probed to GPU 0"):
        owner.assign_roles({0: 0, 1: 0})


def test_bundle_requirements_size_shared_bundle_to_max_role_cpu() -> None:
    """A shared rollout/reward bundle reserves the larger of the two CPU asks."""
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
            "reward": {"device": "gpu", "gpu_pool": "rollout"},
        },
        worker=_worker(cpus_per_worker=2.0),
    )
    # reward_cpus_per_worker default is 0.5; shared bundle must take 2.0.
    requirements = owner._bundle_requirements()
    shared_bundle = owner.layout.rollout_bundle_indices[0]
    assert requirements[shared_bundle]["CPU"] == 2.0
    assert requirements[shared_bundle]["GPU"] == 1.0
    # Trainer-reserved bundle holds the GPU with only a token CPU.
    trainer_bundle = owner.layout.bundle_gpu_ids.index(owner.resources.trainer_devices[0])
    assert requirements[trainer_bundle] == {"CPU": 0.001, "GPU": 1.0}


def test_required_local_cluster_cpus_uses_placement_bundle_sum() -> None:
    """Fractional role bundles are summed and rounded once at node startup."""
    owner = _owner(
        {
            "visible_devices": [0],
            "trainer": {"devices": [0]},
            "rollout": {
                "gpu_pool": "trainer",
                "devices": [0],
                "gpus_per_worker": 1,
                "num_workers": 1,
            },
            "reward": {"device": "cpu"},
        },
        worker=_worker(cpus_per_worker=4.0),
    )

    # No CPU reward bundle: in-process CPU rewards run in the driver.
    assert owner._bundle_requirements() == [
        {"CPU": 4.0, "GPU": 1.0},
    ]
    assert owner.required_local_cluster_cpus() == 4


@pytest.mark.parametrize("quantity", [0.0, -1.0, float("nan"), float("inf")])
def test_required_local_cluster_cpus_rejects_invalid_bundle_cpu(quantity: float) -> None:
    owner = _owner(
        {
            "visible_devices": [0],
            "trainer": {"devices": [0]},
            "rollout": {
                "gpu_pool": "trainer",
                "devices": [0],
                "gpus_per_worker": 1,
            },
        },
        worker=_worker(cpus_per_worker=quantity),
    )

    with pytest.raises(ValueError, match="finite and > 0"):
        owner.required_local_cluster_cpus()


def test_placement_owner_consumes_exact_rollout_cpu_capability() -> None:
    worker = _worker(cpus_per_worker=2.5)
    owner = _owner(
        {
            "visible_devices": [],
            "trainer": {"num_gpus": 0},
            "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
        },
        worker=worker,
    )

    assert owner.rollout_worker is worker
    assert owner._bundle_requirements() == [{"CPU": 2.5}]


@_REAL_RAY_PLACEMENT
def test_shutdown_retries_same_placement_group_after_remove_failure(monkeypatch) -> None:
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
        },
    )
    placement_group = object()
    owner._placement_group = placement_group
    calls: list[object] = []

    def remove(pg):
        calls.append(pg)
        if len(calls) == 1:
            return RuntimeError("placement remove failed")
        return None

    monkeypatch.setattr("vrl.ray.placement.remove_placement_group", remove)

    with pytest.raises(RuntimeError, match="placement remove failed"):
        owner.shutdown()
    assert owner._placement_group is placement_group

    owner.shutdown()
    assert calls == [placement_group, placement_group]
    assert owner._placement_group is None


@_REAL_RAY_PLACEMENT
def test_create_failure_retains_placement_for_cleanup_retry(monkeypatch) -> None:
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
        },
    )

    class _PlacementGroup:
        @staticmethod
        def ready():
            return object()

    placement_group = _PlacementGroup()
    remove_calls: list[object] = []

    monkeypatch.setattr(
        "vrl.ray.placement._create_raw_placement_group",
        lambda *_args, **_kwargs: placement_group,
    )
    monkeypatch.setattr(
        "vrl.ray.placement.require_ray",
        lambda: type("_Ray", (), {"get": staticmethod(lambda *_args, **_kwargs: [None])})(),
    )
    monkeypatch.setattr(
        GlobalRayPlacementOwner,
        "_probe_gpu_bundles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )

    def remove(pg):
        remove_calls.append(pg)
        if len(remove_calls) == 1:
            return RuntimeError("initial remove failed")
        return None

    monkeypatch.setattr("vrl.ray.placement.remove_placement_group", remove)

    with pytest.raises(RuntimeError, match="probe failed") as caught:
        owner.create()
    assert any("retained the handle" in note for note in caught.value.__notes__)
    assert owner._placement_group is placement_group
    assert owner._placement_ready is False

    with pytest.raises(RuntimeError, match="cleanup is still pending"):
        owner.create()
    owner.shutdown()

    assert remove_calls == [placement_group, placement_group]
    assert owner._placement_group is None


@_REAL_RAY_PLACEMENT
def test_ready_failure_retains_exact_placement_for_shutdown_retry(monkeypatch) -> None:
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
        },
    )

    class _PlacementGroup:
        def ready(self):
            return object()

    placement_group = _PlacementGroup()

    class _Ray:
        @staticmethod
        def get(*_args, **_kwargs):
            raise RuntimeError("ready transport failed")

    remove_calls: list[object] = []

    def remove(pg):
        remove_calls.append(pg)
        if len(remove_calls) == 1:
            return RuntimeError("initial remove failed")
        return None

    monkeypatch.setattr("vrl.ray.placement.require_ray", lambda: _Ray())
    monkeypatch.setattr(
        "vrl.ray.placement._create_raw_placement_group",
        lambda *_args, **_kwargs: placement_group,
    )
    monkeypatch.setattr("vrl.ray.placement.remove_placement_group", remove)

    with pytest.raises(RuntimeError, match="placement group not ready") as caught:
        owner.create()

    assert caught.value.__cause__ is not None
    assert "ready transport failed" in str(caught.value.__cause__)
    assert any("retained the handle" in note for note in caught.value.__notes__)
    assert owner._placement_group is placement_group
    assert owner._placement_ready is False

    owner.shutdown()

    assert remove_calls == [placement_group, placement_group]
    assert owner._placement_group is None


@pytest.mark.real_cover(
    "tests/ray/test_global_placement.py::test_probe_actor_kill_failure_is_a_create_failure",
    why=(
        "a live cluster cannot be told to fail the SECOND probe actor's construction while the "
        "first succeeds, and the partially-built fleet is exactly what this asserts gets killed; "
        "the real probe-create-then-kill path against a real placement group is the slow_test "
        "twin named here"
    ),
)
def test_probe_partial_actor_construction_cleans_created_handles(monkeypatch) -> None:
    """Probe fan-out is all-or-nothing: when actor 2 of 2 fails to construct,
    actor 1 must be killed rather than left holding a bundle."""

    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
        },
    )
    first_actor = object()
    create_calls = 0
    killed: list[object] = []

    class _RemoteProbe:
        def options(self, **_kwargs):
            return self

        def remote(self):
            nonlocal create_calls
            create_calls += 1
            if create_calls == 2:
                raise RuntimeError("probe actor construction failed")
            return first_actor

    class _Ray:
        @staticmethod
        def remote(**_kwargs):
            return lambda _actor_cls: _RemoteProbe()

    def kill(_ray, actors):
        killed.extend(actors)
        return []

    monkeypatch.setattr("vrl.ray.placement.actor_scheduling_strategy", lambda *_a, **_k: object())
    monkeypatch.setattr("vrl.ray.placement.kill_actors", kill)

    with pytest.raises(RuntimeError, match="probe actor construction failed"):
        owner._probe_gpu_bundles(_Ray(), object())

    assert killed == [first_actor]


@_REAL_RAY_PROBE_TIMEOUT
def test_probe_timeout_cancels_refs_kills_actors_and_removes_placement(
    monkeypatch,
) -> None:
    owner = _owner(
        {
            "visible_devices": [0, 1],
            "trainer": {"devices": [0]},
            "rollout": {"devices": [1], "gpus_per_worker": 1},
        },
    )
    placement_group = type("_PlacementGroup", (), {"ready": lambda self: object()})()
    refs: list[object] = []
    actors: list[object] = []
    cancelled: list[tuple[object, bool]] = []
    killed: list[object] = []
    removed: list[object] = []

    class _GetTimeoutError(TimeoutError):
        pass

    class _RemoteCall:
        def __init__(self, ref):
            self.ref = ref

        def remote(self):
            return self.ref

    class _ProbeHandle:
        def __init__(self):
            ref = object()
            refs.append(ref)
            self.node_and_gpus = _RemoteCall(ref)

    class _RemoteProbe:
        def options(self, **_kwargs):
            return self

        def remote(self):
            actor = _ProbeHandle()
            actors.append(actor)
            return actor

    class _Ray:
        exceptions = type("_Exceptions", (), {"GetTimeoutError": _GetTimeoutError})

        def __init__(self):
            self.get_calls = 0

        def get(self, _refs, *, timeout):
            assert timeout > 0
            self.get_calls += 1
            if self.get_calls == 1:
                return None
            raise _GetTimeoutError("probe stalled")

        @staticmethod
        def remote(**_kwargs):
            return lambda _actor_cls: _RemoteProbe()

        @staticmethod
        def cancel(ref, *, force):
            cancelled.append((ref, force))

    ray = _Ray()
    monkeypatch.setattr("vrl.ray.placement.require_ray", lambda: ray)
    monkeypatch.setattr(
        "vrl.ray.placement._create_raw_placement_group",
        lambda *_args, **_kwargs: placement_group,
    )
    monkeypatch.setattr(
        "vrl.ray.placement.actor_scheduling_strategy",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vrl.ray.placement.kill_actors",
        lambda _ray, candidates: killed.extend(candidates) or [],
    )
    monkeypatch.setattr(
        "vrl.ray.placement.remove_placement_group",
        lambda pg: removed.append(pg),
    )

    with pytest.raises(RayOperationTimeout, match=r"placement\.gpu_metadata_probe"):
        owner.create()

    assert ray.get_calls == 2
    assert cancelled == [(ref, False) for ref in refs]
    assert killed == actors
    assert removed == [placement_group]
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
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"device": "gpu", "devices": [2]},
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
def test_owner_shares_one_bundle_for_rollout_and_reward_on_simulated_gpus(local_ray) -> None:
    """Shared reward time-multiplexes the rollout GPU: one bundle, both roles."""
    ray = local_ray
    owner = GlobalRayPlacementOwner(
        _resolve(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"device": "gpu", "gpu_pool": "rollout"},
            },
        ),
        _worker(),
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
            "rollout": {"devices": [0], "gpus_per_worker": 1},
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
