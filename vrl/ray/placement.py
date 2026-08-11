"""Run-level Ray placement owner.

A single :class:`GlobalRayPlacementOwner` builds one placement group for the
whole run from the resolved :class:`BundleLayout` and probes which physical GPU
each bundle actually landed on. The rollout runtime consumes its role placement
handle; the in-process reward runtime consumes the reserved device directly.
The public reward role mapping remains the protocol boundary for a future remote
runtime. The owner removes the placement group exactly once at run shutdown.

This replaces two independent placement groups (one built by the rollout
launcher, one by the reward actor runtime) and the reward ``gpu_reservation_count``
offset math. The trainer stays the driver process — never a Ray actor — so its
GPU is protected by a *reserved* (empty) bundle rather than a reservation actor.

Why probe-then-assign: Ray decides the bundle->physical-GPU mapping when the PG
is created, and that mapping is not controllable up front. So the owner probes
the live PG once to learn ``bundle_index -> gpu_id``, then matches each role's
*requested* device ordinals to the bundles that actually hold those GPUs. Naive
"bundle i serves GPU i" would let a trainer-reserved bundle land on a rollout
GPU (or vice versa) and silently collide the driver with a rollout worker.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from vrl.ray.dependencies import (
    current_gpu_ids,
    current_node_ip,
    inspect_cluster,
    require_ray,
)
from vrl.ray.operation_deadline import get_ray_refs
from vrl.ray.resource_cleanup import kill_actors, remove_placement_group
from vrl.ray.resources import (
    BundleLayout,
    ResolvedDistributedResources,
    build_bundle_layout,
)

logger = logging.getLogger(__name__)


class _ActorPlacementMetadata(Protocol):
    worker_id: str
    node_ip: str
    gpu_ids: Sequence[int]


_ActorPlacementInput = Mapping[str, Any] | _ActorPlacementMetadata

# Seconds to wait for the run-level placement group to become ready before the
# owner declares the cluster unable to satisfy its bundles.
_PLACEMENT_READY_TIMEOUT_S = 600.0


class _RolloutCPUConfig(Protocol):
    """Read-only resource view shared with the generation worker snapshot."""

    @property
    def cpus_per_worker(self) -> float: ...


def _create_raw_placement_group(
    bundles: Sequence[Mapping[str, float]],
    *,
    strategy: str,
) -> Any:
    """Lazy Ray adapter that returns an unready placement-group handle."""

    if not bundles:
        raise ValueError("Ray placement group requires at least one bundle")
    from ray.util.placement_group import placement_group

    return placement_group([dict(bundle) for bundle in bundles], strategy=str(strategy))


def actor_scheduling_strategy(
    placement_group: Any,
    *,
    bundle_index: int | None = None,
    capture_child_tasks: bool = True,
) -> Any:
    """Build a placement-group scheduling strategy for one actor."""

    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return PlacementGroupSchedulingStrategy(
        placement_group=placement_group,
        placement_group_bundle_index=bundle_index,
        placement_group_capture_child_tasks=capture_child_tasks,
    )


def cross_node_preflight(ray: Any, resources: ResolvedDistributedResources) -> None:
    """Fail fast when the live Ray cluster cannot host cross-node rollout.

    Runs after ``ray.init()`` (resolution earlier in the pipeline cannot see the
    cluster). Verifies that non-driver nodes expose enough GPUs for rollout, and
    that the driver/head node does not expose Ray GPUs — otherwise rollout actors
    could be scheduled onto the trainer GPU, since cross-node mode intentionally
    drops the placement-group trainer reservation.
    """

    topology = inspect_cluster(ray)

    needed = resources.rollout_num_gpus
    if topology.non_driver_gpus < needed:
        raise RuntimeError(
            f"cross_node rollout needs {needed} GPU(s) on non-driver Ray nodes, but "
            f"only {topology.non_driver_gpus:g} are available. Join more rollout "
            "workers, e.g. `ray start --address=<head>:6379 --num-gpus=<n>`.",
        )
    if topology.driver_gpus > 0:
        raise RuntimeError(
            f"cross_node rollout: the driver/head node exposes {topology.driver_gpus:g} "
            "Ray GPU(s), so rollout could be scheduled onto the trainer GPU. Start the "
            "head with `ray start --head --num-gpus=0` so the trainer GPU stays out "
            "of Ray's scheduling pool.",
        )


def validate_actor_gpu_ids(
    metadata: Sequence[_ActorPlacementInput],
    *,
    expected_gpu_ids: Sequence[int],
    role: str,
    cross_node: bool = False,
    driver_node_ip: str | None = None,
) -> tuple[int, ...]:
    """Validate that Ray actors received only the expected GPU IDs.

    Single-node mode asserts every actor's node-local GPU id falls inside the
    globally resolved ordinal set. Cross-node mode cannot use that assumption
    (each node has its own ordinal space and Ray remaps ``CUDA_VISIBLE_DEVICES``
    per actor), so it instead validates that every worker (a) got a GPU, (b) does
    not run on the driver/head node, and (c) holds a unique ``(node_ip, gpu_id)``
    pair so no two workers share a physical GPU.
    """

    if cross_node:
        return _validate_cross_node_actor_gpu_ids(
            metadata,
            role=role,
            driver_node_ip=driver_node_ip,
        )

    expected = {int(gpu_id) for gpu_id in expected_gpu_ids}
    if not expected:
        return ()

    actual: set[int] = set()
    for meta in metadata:
        worker_id = str(_placement_meta_get(meta, "worker_id", "unknown"))
        worker_gpu_ids = tuple(int(gpu_id) for gpu_id in _placement_meta_get(meta, "gpu_ids", ()))
        if not worker_gpu_ids:
            raise RuntimeError(f"Ray {role} worker {worker_id} has no assigned GPU ids")
        outside = set(worker_gpu_ids) - expected
        if outside:
            raise RuntimeError(
                f"Ray {role} worker {worker_id} assigned GPU ids "
                f"{sorted(worker_gpu_ids)}, outside resolved {role} devices "
                f"{sorted(expected)}",
            )
        actual.update(worker_gpu_ids)

    if actual != expected:
        raise RuntimeError(
            f"Ray {role} placement did not cover the resolved {role} devices: "
            f"actual={sorted(actual)} expected={sorted(expected)}",
        )
    return tuple(sorted(actual))


def _validate_cross_node_actor_gpu_ids(
    metadata: Sequence[_ActorPlacementInput],
    *,
    role: str,
    driver_node_ip: str | None,
) -> tuple[int, ...]:
    """Node-aware GPU validation for cross-node rollout actors."""

    seen_pairs: set[tuple[str, int]] = set()
    gpu_ids: list[int] = []
    for meta in metadata:
        worker_id = str(_placement_meta_get(meta, "worker_id", "unknown"))
        node_ip = str(_placement_meta_get(meta, "node_ip", ""))
        worker_gpu_ids = tuple(int(gpu_id) for gpu_id in _placement_meta_get(meta, "gpu_ids", ()))
        if not worker_gpu_ids:
            raise RuntimeError(f"Ray {role} worker {worker_id} has no assigned GPU ids")
        if driver_node_ip is not None and node_ip == str(driver_node_ip):
            raise RuntimeError(
                f"Ray {role} worker {worker_id} landed on the driver/head node "
                f"{node_ip}; cross-node rollout must run off the trainer node. "
                "Start the head with `ray start --head --num-gpus=0` so the trainer "
                "GPU stays out of Ray's scheduling pool.",
            )
        for gpu_id in worker_gpu_ids:
            pair = (node_ip, gpu_id)
            if pair in seen_pairs:
                raise RuntimeError(
                    f"Ray {role} workers share GPU {gpu_id} on node {node_ip}; "
                    "each rollout worker must own a distinct physical GPU.",
                )
            seen_pairs.add(pair)
            gpu_ids.append(gpu_id)
    return tuple(sorted(gpu_ids))


def _placement_meta_get(meta: _ActorPlacementInput, key: str, default: Any) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


@dataclass(frozen=True, slots=True)
class RolePlacement:
    """Placement handle for one execution role.

    The role's runtime schedules its actors into ``bundle_indices`` of the
    shared ``placement_group`` and never removes the group itself.
    ``expected_gpu_ids`` are the device ordinals the role's actors must land on
    (empty under cross-node, where node-aware validation is used instead).
    """

    placement_group: Any
    bundle_indices: tuple[int, ...]
    expected_gpu_ids: tuple[int, ...]


class _ProbeActor:
    def node_and_gpus(self) -> tuple[str, tuple[int, ...]]:
        return current_node_ip(), tuple(current_gpu_ids())


@dataclass(slots=True)
class GlobalRayPlacementOwner:
    """Owns the single run-level Ray placement group and role->bundle mapping."""

    resources: ResolvedDistributedResources
    rollout_worker: _RolloutCPUConfig
    layout: BundleLayout = field(init=False)
    _placement_group: Any | None = field(default=None, init=False, repr=False)
    _placement_ready: bool = field(default=False, init=False, repr=False)
    _role_bundles: dict[str, tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.layout = build_bundle_layout(self.resources)

    # -- lifecycle -----------------------------------------------------------

    def create(self) -> None:
        """Create the placement group, probe GPU bundles, and assign roles."""

        if self._placement_ready:
            return
        if self._placement_group is not None:
            raise RuntimeError(
                "placement creation previously failed and cleanup is still pending; "
                "call shutdown() before retrying create()",
            )
        if self.layout.total_bundles == 0:
            # Trainer-only / no-rollout plans need no Ray placement.
            self._role_bundles = {"rollout": (), "reward": ()}
            self._placement_ready = True
            return

        ray = require_ray()
        bundle_requirements = self._bundle_requirements()
        # Cross-node spreads bundles across nodes; single-node packs them.
        strategy = "SPREAD" if self.resources.cross_node else "PACK"
        pg = _create_raw_placement_group(bundle_requirements, strategy=strategy)
        # Claim the raw handle before waiting for readiness. A ready/probe/assign
        # failure whose removal also fails must leave this exact placement group
        # reachable by terminal shutdown.
        self._placement_group = pg
        try:
            try:
                ray.get(pg.ready(), timeout=_PLACEMENT_READY_TIMEOUT_S)
            except Exception as exc:
                raise RuntimeError(
                    f"Ray placement group not ready after {_PLACEMENT_READY_TIMEOUT_S:.0f}s: "
                    f"bundles={bundle_requirements} strategy={strategy!r}. "
                    "The cluster cannot satisfy these bundles -- check whether "
                    "resident actors hold the GPUs this group is trying to reserve.",
                ) from exc
            probed = self._probe_gpu_bundles(ray, pg)
            self._role_bundles = self.assign_roles(probed)
        except BaseException as error:
            cleanup_error = remove_placement_group(pg)
            if cleanup_error is None:
                self._placement_group = None
            else:
                error.add_note(
                    "initial placement cleanup failed; the owner retained the "
                    f"handle for shutdown retry: {cleanup_error!r}",
                )
            raise
        self._placement_ready = True
        logger.info(
            "GlobalRayPlacementOwner created: bundles=%s probed_gpus=%s roles=%s",
            self.layout.bundle_gpu_ids,
            probed,
            self._role_bundles,
        )

    def required_local_cluster_cpus(self) -> int:
        """Return the Ray node CPU capacity needed by this placement plan."""

        quantities = [float(bundle["CPU"]) for bundle in self._bundle_requirements()]
        invalid = [
            quantity for quantity in quantities if not math.isfinite(quantity) or quantity <= 0
        ]
        if invalid:
            raise ValueError(
                f"Ray placement CPU quantities must be finite and > 0, got {invalid}",
            )
        # Placement bundles accept fractional CPU reservations, while Ray node
        # startup requires an integer capacity. Sum the actual bundle plan so
        # colocated roles retain its max-not-sum semantics.
        return max(1, math.ceil(math.fsum(quantities)))

    def shutdown(self) -> None:
        """Remove the placement group, retaining ownership until success."""

        pg = self._placement_group
        if pg is None:
            return
        error = remove_placement_group(pg)
        if error is not None:
            raise error
        self._placement_group = None
        self._placement_ready = False
        self._role_bundles.clear()

    # -- role placement handles ---------------------------------------------

    @property
    def rollout_placement(self) -> RolePlacement | None:
        bundles = self._role_bundles.get("rollout", ())
        if not bundles or self._placement_group is None:
            return None
        return RolePlacement(
            placement_group=self._placement_group,
            bundle_indices=bundles,
            expected_gpu_ids=(
                () if self.resources.cross_node else tuple(self.resources.rollout_devices)
            ),
        )

    # -- role assignment (pure; unit-tested with an injected probe) ----------

    def assign_roles(self, probed: dict[int, int]) -> dict[str, tuple[int, ...]]:
        """Map roles to bundle indices given a ``bundle_index -> gpu_id`` probe.

        Single-node: match each role's requested device ordinals to the bundle
        that actually holds that GPU, so a reserved trainer bundle never doubles
        as a rollout bundle. Cross-node: device ordinals are budget tokens that
        cannot be matched to real remote GPU ids, so roles keep the plan's
        positional bundle indices and rely on node-aware validation at launch.
        CPU bundles (no GPU) always keep their positional plan indices.
        """

        if self.resources.cross_node:
            return {
                "rollout": self.layout.rollout_bundle_indices,
                "reward": self.layout.reward_bundle_indices,
            }

        gpu_to_bundle: dict[int, int] = {}
        for bundle_index, gpu_id in probed.items():
            if gpu_id in gpu_to_bundle:
                raise RuntimeError(
                    f"two bundles probed to GPU {gpu_id}: "
                    f"{gpu_to_bundle[gpu_id]} and {bundle_index}",
                )
            gpu_to_bundle[gpu_id] = bundle_index

        return {
            "rollout": self._match_gpu_bundles(
                "rollout",
                self.resources.rollout_devices,
                self.resources.rollout_gpus_per_worker,
                self.layout.rollout_bundle_indices,
                gpu_to_bundle,
            ),
            "reward": self._match_gpu_bundles(
                "reward",
                self.resources.reward_devices,
                self.resources.reward_gpus_per_worker,
                self.layout.reward_bundle_indices,
                gpu_to_bundle,
            ),
        }

    def _match_gpu_bundles(
        self,
        role: str,
        devices: tuple[int, ...],
        gpus_per_worker: float,
        cpu_plan_indices: tuple[int, ...],
        gpu_to_bundle: dict[int, int],
    ) -> tuple[int, ...]:
        if gpus_per_worker <= 0:
            # CPU-only role: positional plan bundles (probe does not cover them).
            return cpu_plan_indices
        matched: list[int] = []
        for gpu_id in devices:
            bundle_index = gpu_to_bundle.get(gpu_id)
            if bundle_index is None:
                raise RuntimeError(
                    f"{role} device GPU {gpu_id} has no bundle in the probed "
                    f"placement group (probed GPUs={sorted(gpu_to_bundle)})",
                )
            matched.append(bundle_index)
        return tuple(matched)

    # -- internals -----------------------------------------------------------

    def _bundle_requirements(self) -> list[dict[str, float]]:
        requirements: list[dict[str, float]] = []
        for bundle_index, gpu_id in enumerate(self.layout.bundle_gpu_ids):
            cpu = self._bundle_cpu(bundle_index)
            bundle: dict[str, float] = {"CPU": cpu}
            if gpu_id is not None:
                bundle["GPU"] = 1.0
            requirements.append(bundle)
        return requirements

    def _bundle_cpu(self, bundle_index: int) -> float:
        """CPU a bundle reserves = max over the roles that may run in it.

        A trainer-reserved GPU bundle (no role) only needs a token CPU so the
        empty bundle is schedulable while still holding the GPU out of Ray's
        pool.
        """

        cpus: list[float] = []
        if bundle_index in self.layout.rollout_bundle_indices:
            cpus.append(float(self.rollout_worker.cpus_per_worker))
        if bundle_index in self.layout.reward_bundle_indices:
            cpus.append(float(self.resources.reward_cpus_per_worker))
        if not cpus:
            return 0.001
        return max(cpus)

    def _probe_gpu_bundles(self, ray: Any, pg: Any) -> dict[int, int]:
        gpu_bundles = [
            index for index, gpu_id in enumerate(self.layout.bundle_gpu_ids) if gpu_id is not None
        ]
        if not gpu_bundles:
            return {}
        remote_probe = ray.remote(num_cpus=0.001, num_gpus=1.0)(_ProbeActor)
        actors: list[Any] = []
        try:
            for bundle_index in gpu_bundles:
                actor = remote_probe.options(
                    scheduling_strategy=actor_scheduling_strategy(
                        placement_group=pg,
                        bundle_index=bundle_index,
                        capture_child_tasks=True,
                    ),
                ).remote()
                actors.append(actor)
            results = get_ray_refs(
                ray,
                [actor.node_and_gpus.remote() for actor in actors],
                operation="placement.gpu_metadata_probe",
                timeout_s=_PLACEMENT_READY_TIMEOUT_S,
                context=f"bundles={gpu_bundles}",
            )
        except BaseException as error:
            actor_failures = kill_actors(ray, actors)
            if actor_failures:
                error.add_note(
                    "placement probe actor cleanup also failed: "
                    f"{len(actor_failures)} actor(s) retained by the placement group",
                )
            raise
        actor_failures = kill_actors(ray, actors)
        if actor_failures:
            raise RuntimeError(
                "placement probe actor cleanup incomplete: "
                f"{len(actor_failures)} actor kill(s) failed",
            ) from actor_failures[0][1]
        probed: dict[int, int] = {}
        for bundle_index, (_node_ip, gpu_ids) in zip(gpu_bundles, results, strict=True):
            if not gpu_ids:
                raise RuntimeError(
                    f"placement-group bundle {bundle_index} received no GPU",
                )
            probed[bundle_index] = int(gpu_ids[0])
        return probed


__all__ = [
    "GlobalRayPlacementOwner",
    "RolePlacement",
    "actor_scheduling_strategy",
    "cross_node_preflight",
    "validate_actor_gpu_ids",
]
