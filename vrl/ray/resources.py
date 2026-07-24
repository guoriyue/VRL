"""Role-level resource resolution for distributed VRL runs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vrl.config.reward_inference import reward_inference_configs_from_cfg
from vrl.utils.config import cfg_get, to_builtin


@dataclass(frozen=True, slots=True)
class RoleResourceConfig:
    """GPU ownership request for one execution role."""

    num_gpus: int | str | None = "auto"
    devices: list[int] | str = "auto"


@dataclass(frozen=True, slots=True)
class RolloutResourceConfig(RoleResourceConfig):
    """GPU ownership request for rollout workers."""

    gpus_per_worker: float = 1.0
    num_workers: int | str = "auto"
    # Rollout GPU pool source (public key: distributed.resources.rollout.gpu_pool).
    # Mirrors reward.gpu_pool so all roles share one "which pool do I borrow" grammar:
    #   "auto"      derive from topology: a dedicated spare GPU when one exists,
    #               else overlap the trainer GPU (single-GPU colocated fallback).
    #   "trainer"   share the trainer GPU pool (colocated). Shared roles always
    #               hand the GPU over between phases; gpu_pool=trainer is itself
    #               the overlap permission (no allow_overlap needed).
    #   "dedicated" require a dedicated spare rollout GPU; error if none exists.
    gpu_pool: str = "auto"


@dataclass(frozen=True, slots=True)
class RewardResourceConfig(RoleResourceConfig):
    """GPU ownership request for in-process reward inference."""

    gpus_per_worker: float = 1.0
    num_workers: int | str = "auto"
    # Reward GPU pool source (public key: distributed.resources.reward.gpu_pool;
    # the legacy share_with_rollout bool maps here at parse time):
    #   "auto"      derive from topology: a dedicated spare GPU when one exists,
    #               otherwise share the rollout pool.
    #   "rollout"   always share the rollout GPU pool.
    #   "dedicated" require a dedicated spare reward GPU; error if none exists.
    gpu_pool: str = "auto"


@dataclass(frozen=True, slots=True)
class DistributedResourceConfig:
    """Top-level role resource request."""

    visible_devices: list[int] | str = "auto"
    trainer: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    rollout: RolloutResourceConfig = field(default_factory=RolloutResourceConfig)
    reward: RewardResourceConfig = field(
        default_factory=lambda: RewardResourceConfig(num_gpus=0, devices=[]),
    )
    allow_overlap: bool = False
    reward_cpus_per_worker: float = 0.5
    cross_node: bool = False


@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    """Whether a role keeps its accelerator active across phase boundaries.

    ``resident`` keeps serving across phases because the role owns a dedicated
    GPU. ``on_demand`` yields a shared GPU at a handoff and activates again on
    next use. The rollout backend parks its actors in host RAM; this policy does
    not require process destruction.
    """

    mode: Literal["resident", "on_demand"]


@dataclass(frozen=True, slots=True)
class PhaseHandoffPolicy:
    """Which resident-vs-shared roles must step off their GPU at each boundary.

    A flag is True only when two roles share a GPU and the next phase needs the
    first to release it. Derived once from topology so no runtime re-decides it
    per call.
    """

    release_rollout_before_train: bool
    release_rollout_before_reward: bool
    release_trainer_before_reward: bool
    release_reward_after_score: bool


@dataclass(frozen=True, slots=True)
class RayLifecyclePlan:
    """Single topology-derived answer to "which role yields its GPU when".

    Built by :func:`resolve_distributed_resources` from GPU ownership so the
    launcher, collector, and reward runtime read one declarative plan instead of
    each re-deriving ``release_after_*`` from raw device sets. Real behavior reads
    ``resolved.lifecycle.*``; no flat release-after-collect mirror is retained.
    """

    rollout: ActorLeasePolicy
    # display/provenance-only: compact formatter summary derived from handoff;
    # runtime behavior consumes the boundary-specific handoff flags instead.
    reward: ActorLeasePolicy
    handoff: PhaseHandoffPolicy


@dataclass(frozen=True, slots=True)
class ResolvedDistributedResources:
    """Concrete resource plan consumed by trainer and Ray role launchers."""

    # display/provenance-only: the full visible GPU pool, printed by
    # format_distributed_resource_plan. No behavioral consumer reads it; it
    # records which GPUs the machine exposed (may exceed the role union's spare).
    visible_devices: tuple[int, ...]
    trainer_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]
    reward_devices: tuple[int, ...]
    # Rewards execute in-process. When no GPU reservation exists, an active
    # reward follows the trainer's rank-local device instead of disappearing
    # from the topology. Consumers use this flag to require trainer parking and
    # reward release without conflating execution with Ray reservations.
    reward_uses_trainer_device: bool
    rollout_num_workers: int
    rollout_gpus_per_worker: float
    reward_num_workers: int
    reward_gpus_per_worker: float
    reward_cpus_per_worker: float
    cross_node: bool
    # Named view over release decisions: lease mode per role plus the per-boundary
    # handoff. The launcher/collector/reward read this instead of re-deriving from
    # device sets.
    lifecycle: RayLifecyclePlan

    @property
    def rollout_num_gpus(self) -> int:
        return len(self.rollout_devices)

    @property
    def colocated(self) -> bool:
        return bool(set(self.trainer_devices) & set(self.rollout_devices))

    @property
    def requires_trainer_reservation(self) -> bool:
        return (
            bool(self.trainer_devices)
            and self.rollout_gpus_per_worker > 0
            and not self.colocated
            and self.rollout_num_workers > 0
            and not self.cross_node
        )


_MISSING = object()


def resolve_distributed_resources(cfg: Any) -> ResolvedDistributedResources:
    """Resolve role-level resource config into concrete CUDA ordinals.

    This is the single source of truth for trainer/rollout/reward GPU
    ownership. It intentionally does static ownership checks only; memory
    pressure is still a runtime concern.
    """

    config = _distributed_resource_config_from_cfg(cfg)
    reward_inference = reward_inference_configs_from_cfg(cfg)
    local_reward_configured = any(
        inference.kind == "in_process" for inference in reward_inference.values()
    )
    if reward_inference and not local_reward_configured:
        # External services own their accelerator and process placement. Ignore
        # inherited reward presets here instead of creating a phantom local GPU
        # or CPU bundle; the HTTP runtime is a driver-side client.
        config = replace(
            config,
            reward=RewardResourceConfig(
                num_gpus=0,
                devices=[],
                gpus_per_worker=1.0,
                num_workers="auto",
            ),
        )
    training = cfg_get(cfg_get(cfg, "distributed", {}), "training", {})
    training_strategy = str(cfg_get(training, "strategy", "single_process"))
    training_world_size = int(cfg_get(training, "num_nodes", 1)) * int(
        cfg_get(training, "gpus_per_node", 1),
    )
    if config.cross_node:
        visible_devices = _resolve_cross_node_visible_devices(config)
    else:
        parsed_visible_devices = _parse_devices(config.visible_devices)
        visible_devices = (
            _auto_visible_cuda_devices()
            if parsed_visible_devices == "auto"
            else tuple(
                _dedupe_ints(
                    parsed_visible_devices,
                    field_name="distributed.resources.visible_devices",
                ),
            )
        )

    # fsdp has two topologies. ASYMMETRIC: one resolver owns the whole training
    # world (trainer = num_nodes*gpus_per_node GPUs, rollout/reward on separate
    # cards) — the single-node multi-GPU model. SYMMETRIC COLOCATED: one torchrun
    # rank per node, each owning its 1 local GPU with a colocated rollout — the same
    # per-rank-local model ddp uses (SPRINT_symmetric_colocated_ddp), signaled by
    # rollout.gpu_pool=trainer. Only asymmetric fsdp sizes the trainer to the whole
    # world; symmetric fsdp follows the per-rank single-GPU rule like ddp.
    fsdp_symmetric_colocated = training_strategy == "fsdp" and config.rollout.gpu_pool == "trainer"
    fsdp_asymmetric = training_strategy == "fsdp" and not fsdp_symmetric_colocated
    trainer_default_auto = (
        training_world_size if fsdp_asymmetric else (1 if visible_devices else 0)
    )
    trainer_devices = _resolve_role_devices(
        role="trainer",
        visible_devices=visible_devices,
        role_config=config.trainer,
        default_auto_count=trainer_default_auto,
    )
    _validate_trainer_device_count(
        training_strategy,
        trainer_devices,
        training_world_size,
        symmetric_colocated=fsdp_symmetric_colocated,
    )

    rollout_gpus_per_worker = float(config.rollout.gpus_per_worker)
    if rollout_gpus_per_worker not in {0.0, 1.0}:
        raise ValueError(
            "distributed.resources.rollout.gpus_per_worker currently supports "
            f"0 or 1, got {rollout_gpus_per_worker}",
        )

    # rollout.gpu_pool=trainer borrows the trainer GPU (colocated) and is itself the
    # overlap permission, so it doesn't also need distributed.resources.allow_overlap.
    rollout_devices = _resolve_rollout_devices(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_config=config.rollout,
        allow_overlap=config.allow_overlap,
        gpu_pool=config.rollout.gpu_pool,
    )
    rollout_num_gpus = len(rollout_devices)

    if rollout_gpus_per_worker == 0 and rollout_num_gpus > 0:
        raise ValueError(
            "distributed.resources.rollout.gpus_per_worker=0 requires zero "
            f"resolved rollout GPUs, got {list(rollout_devices)}. Remove the GPU "
            "request or set gpus_per_worker=1.",
        )
    if rollout_gpus_per_worker > 0 and rollout_num_gpus == 0:
        raise ValueError(
            "No rollout GPUs are available after reserving trainer devices "
            f"{list(trainer_devices)} with distributed.resources.allow_overlap=false. "
            "Expose more GPUs, or set "
            "distributed.resources.rollout.gpu_pool=trainer to time-share a trainer GPU.",
        )

    rollout_num_workers = _resolve_rollout_num_workers(
        rollout_config=config.rollout,
        rollout_num_gpus=rollout_num_gpus,
        gpus_per_worker=rollout_gpus_per_worker,
    )

    colocated = bool(set(trainer_devices) & set(rollout_devices))
    if colocated and not config.allow_overlap and config.rollout.gpu_pool != "trainer":
        raise ValueError(
            "Trainer and rollout devices overlap but "
            "distributed.resources.allow_overlap=false: "
            f"trainer={list(trainer_devices)} rollout={list(rollout_devices)}. "
            "Set distributed.resources.rollout.gpu_pool=trainer to colocate, or "
            "allow_overlap=true.",
        )

    reward_gpus_per_worker = float(config.reward.gpus_per_worker)
    if reward_gpus_per_worker not in {0.0, 1.0}:
        raise ValueError(
            "distributed.resources.reward.gpus_per_worker currently supports "
            f"0 or 1, got {reward_gpus_per_worker}",
        )
    reward_devices = _resolve_reward_devices(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_devices=rollout_devices,
        reward_config=config.reward,
        allow_overlap=config.allow_overlap,
    )
    reward_num_gpus = len(reward_devices)
    if reward_gpus_per_worker == 0 and reward_num_gpus > 0:
        raise ValueError(
            "distributed.resources.reward.gpus_per_worker=0 requires zero "
            f"resolved reward GPUs, got {list(reward_devices)}. Remove the GPU "
            "request or set gpus_per_worker=1.",
        )
    reward_num_workers = _resolve_role_num_workers(
        role="reward",
        num_workers=config.reward.num_workers,
        num_gpus=reward_num_gpus,
        gpus_per_worker=reward_gpus_per_worker,
    )
    if reward_gpus_per_worker > 0 and reward_num_gpus == 0 and reward_num_workers > 0:
        raise ValueError(
            "distributed.resources.reward requested GPU workers but no reward GPUs were resolved",
        )

    reserved_reward_overlaps_trainer = bool(set(reward_devices) & set(trainer_devices))
    if reserved_reward_overlaps_trainer and not config.allow_overlap:
        raise ValueError(
            "Trainer and reward devices overlap but "
            "distributed.resources.allow_overlap=false: "
            f"trainer={list(trainer_devices)} reward={list(reward_devices)}",
        )

    # Asymmetric fsdp owns the whole training world with rollout/reward on separate
    # cards, so the trainer set must be disjoint from both regardless of
    # allow_overlap. Symmetric colocated fsdp (rollout.gpu_pool=trainer) is the
    # opposite by design — each rank's rollout shares its trainer GPU, exactly like
    # ddp — so the disjoint rule does not apply to it.
    if fsdp_asymmetric:
        _validate_fsdp_trainer_disjoint(trainer_devices, rollout_devices, reward_devices)

    reward_configured = local_reward_configured
    reward_runs_on_cpu = reward_num_workers == 1 and reward_gpus_per_worker == 0
    reward_uses_trainer_device = bool(
        reward_configured and not reward_devices and not reward_runs_on_cpu and trainer_devices
    )
    reward_execution_devices = (
        tuple(trainer_devices) if reward_uses_trainer_device else tuple(reward_devices)
    )
    reward_shared_with_rollout = bool(set(reward_execution_devices) & set(rollout_devices))
    reward_shared_with_trainer = bool(set(reward_execution_devices) & set(trainer_devices))

    # Release scheduling is derived entirely from the resolved GPU topology:
    # roles that share a GPU hand it over between phases; roles with dedicated
    # GPUs stay resident. The named handoff plan is the only behavior source.
    rollout_release_before_train = colocated
    rollout_release_before_reward_model = reward_shared_with_rollout
    trainer_release_before_reward_model = reward_shared_with_trainer
    # Rewards score in-process now (no Ray reward actors). A reward sharing
    # either rollout or trainer must prove it parked before the next phase can
    # reclaim the card; a dedicated reward stays resident.
    reward_release_after_score = reward_shared_with_rollout or reward_shared_with_trainer
    rollout_on_demand = rollout_release_before_train or rollout_release_before_reward_model

    # A role is on_demand when any handoff makes it yield, while the handoff plan
    # keeps the specific phase boundary explicit.
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(
            mode="on_demand" if rollout_on_demand else "resident",
        ),
        reward=ActorLeasePolicy(
            mode="on_demand" if reward_release_after_score else "resident",
        ),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=rollout_release_before_train,
            release_rollout_before_reward=rollout_release_before_reward_model,
            release_trainer_before_reward=trainer_release_before_reward_model,
            release_reward_after_score=reward_release_after_score,
        ),
    )
    return ResolvedDistributedResources(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_devices=rollout_devices,
        reward_devices=reward_devices,
        reward_uses_trainer_device=reward_uses_trainer_device,
        rollout_num_workers=rollout_num_workers,
        rollout_gpus_per_worker=rollout_gpus_per_worker,
        reward_num_workers=reward_num_workers,
        reward_gpus_per_worker=reward_gpus_per_worker,
        reward_cpus_per_worker=config.reward_cpus_per_worker,
        cross_node=config.cross_node,
        lifecycle=lifecycle,
    )


def _validate_trainer_device_count(
    strategy: str,
    trainer_devices: tuple[int, ...],
    world_size: int,
    *,
    symmetric_colocated: bool = False,
) -> None:
    """Trainer GPU-count rule, gated by ``distributed.training.strategy``.

    ``single_process`` is one process on one device (0 or 1 GPU).
    ASYMMETRIC ``fsdp`` runs one torchrun process per GPU under a single resolver,
    so the trainer device set must cover the whole training world
    (``num_nodes * gpus_per_node``). SYMMETRIC COLOCATED ``fsdp`` (and ``ddp``)
    resolve per-rank-local: each rank owns its 1 GPU, so they follow the
    single-device rule below, not the world-covering one.
    """

    if strategy == "fsdp" and not symmetric_colocated:
        if len(trainer_devices) != world_size:
            raise ValueError(
                "distributed.training.strategy=fsdp: trainer must own "
                f"num_nodes*gpus_per_node={world_size} GPUs, got "
                f"{list(trainer_devices)} (count {len(trainer_devices)}). Set "
                f"distributed.resources.trainer.num_gpus={world_size} (or matching "
                "devices).",
            )
        return
    if len(trainer_devices) > 1:
        raise ValueError(
            "distributed.resources.trainer.devices currently supports only "
            f"0 or 1 GPU for the single-process trainer, got {trainer_devices}. "
            "Set distributed.training.strategy=fsdp for multi-GPU training.",
        )


def _validate_fsdp_trainer_disjoint(
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_devices: tuple[int, ...],
) -> None:
    """fsdp trainer GPUs must not overlap rollout/reward (even with allow_overlap)."""

    overlap_rollout = sorted(set(trainer_devices) & set(rollout_devices))
    overlap_reward = sorted(set(trainer_devices) & set(reward_devices))
    if overlap_rollout or overlap_reward:
        raise ValueError(
            "distributed.training.strategy=fsdp requires trainer GPUs disjoint from "
            "rollout and reward (each FSDP rank owns its GPU; rollout actors and "
            "the in-process reward model need separate reserved devices). "
            f"trainer={list(trainer_devices)} rollout={list(rollout_devices)} "
            f"reward={list(reward_devices)} "
            f"(overlap rollout={overlap_rollout}, reward={overlap_reward}).",
        )


def trainer_torch_device(resolved: ResolvedDistributedResources) -> str:
    """Return the torch device string the single-process trainer should use."""

    devices = tuple(resolved.trainer_devices)
    if not devices:
        return "cpu"
    return f"cuda:{int(devices[0])}"


def reward_torch_device(
    resolved: ResolvedDistributedResources,
    *,
    trainer_device: Any | None = None,
) -> str:
    """Return the device for the local, in-process reward runtime.

    A resolved reward GPU is a local reservation, not a Ray worker placement.
    The in-process runtime can therefore consume exactly one execution slot.
    A CPU-only resource request selects CPU; cross-node reward ordinals are only
    Ray budget tokens and cannot name a device in the driver process. Multiple
    workers or remote reward inference need a real transport boundary instead.

    When no reward GPU is reserved, reward inference follows the trainer device.
    ``trainer_device`` lets torchrun callers provide their rank-local device
    instead of the resolver's rank-agnostic trainer ordinal.
    """

    devices = tuple(resolved.reward_devices)
    if len(devices) > 1:
        raise ValueError(
            "Local reward inference supports at most one resolved reward GPU, "
            f"got {list(devices)}. Rewards score in the driver process; split "
            "multi-GPU reward inference requires a remote runtime boundary.",
        )
    if resolved.reward_num_workers > 1:
        raise ValueError(
            "Local reward inference supports at most one resolved reward worker, "
            f"got {resolved.reward_num_workers}. Parallel reward workers require "
            "a remote runtime boundary.",
        )
    if resolved.reward_num_workers == 1 and resolved.reward_gpus_per_worker == 0:
        return "cpu"

    if resolved.cross_node and devices:
        raise ValueError(
            "distributed.resources.cross_node=true cannot place local reward "
            f"inference on reward devices {list(devices)}: cross-node device ids "
            "are Ray budget tokens, not driver-local CUDA ordinals. Remove the "
            "reward resource block to score on the trainer device, or implement "
            "a remote reward transport.",
        )
    if devices:
        return f"cuda:{int(devices[0])}"
    if resolved.reward_uses_trainer_device:
        if trainer_device is not None:
            return str(trainer_device)
        return trainer_torch_device(resolved)
    # Callers may ask for a device before a reward section is attached. Keep
    # the historical fallback, while active configured rewards are governed by
    # reward_uses_trainer_device above.
    if trainer_device is not None:
        return str(trainer_device)
    return trainer_torch_device(resolved)


def format_distributed_resource_plan(resolved: ResolvedDistributedResources) -> str:
    """Format a compact resource plan for logs and errors."""

    parts = [
        f"visible={list(resolved.visible_devices)}",
        f"trainer={list(resolved.trainer_devices)}",
        f"rollout={list(resolved.rollout_devices)}",
        f"reward={list(resolved.reward_devices)}",
        f"rollout_workers={resolved.rollout_num_workers}",
        f"rollout_gpus_per_worker={resolved.rollout_gpus_per_worker:g}",
        f"reward_workers={resolved.reward_num_workers}",
        f"reward_gpus_per_worker={resolved.reward_gpus_per_worker:g}",
        f"colocated={resolved.colocated}",
        f"cross_node={resolved.cross_node}",
        f"trainer_reservation={resolved.requires_trainer_reservation}",
        # Reading the plan at a glance: lease mode per role + which boundaries
        # release. resident=stays active, on_demand=parks at the handoff.
        f"lifecycle=rollout:{resolved.lifecycle.rollout.mode}"
        f"/reward:{resolved.lifecycle.reward.mode}",
        "handoff="
        f"before_train:{resolved.lifecycle.handoff.release_rollout_before_train}"
        f",before_reward:{resolved.lifecycle.handoff.release_rollout_before_reward}"
        f",trainer_before_reward:"
        f"{resolved.lifecycle.handoff.release_trainer_before_reward}"
        f",reward_after_score:{resolved.lifecycle.handoff.release_reward_after_score}",
    ]
    return "Distributed resources: " + " ".join(parts)


def _distributed_resource_config_from_cfg(cfg: Any) -> DistributedResourceConfig:
    distributed = cfg_get(cfg, "distributed", {})
    resources = cfg_get(distributed, "resources", {})
    trainer_node = cfg_get(resources, "trainer", {})
    rollout_node = cfg_get(resources, "rollout", {})
    reward_node = cfg_get(resources, "reward", _MISSING)
    rollout_runtime = cfg_get(distributed, "rollout", {})
    reward_runtime = cfg_get(distributed, "reward", {})
    _reject_removed_distributed_keys(rollout_runtime, reward_runtime)
    rollout_gpu_pool = _parse_rollout_gpu_pool(
        rollout_node=rollout_node,
    )

    trainer = RoleResourceConfig(
        num_gpus=cfg_get(trainer_node, "num_gpus", "auto"),
        devices=_parse_devices(cfg_get(trainer_node, "devices", "auto")),
    )
    rollout = RolloutResourceConfig(
        num_gpus=cfg_get(rollout_node, "num_gpus", "auto"),
        devices=_parse_devices(cfg_get(rollout_node, "devices", "auto")),
        gpus_per_worker=float(cfg_get(rollout_node, "gpus_per_worker", 1.0)),
        num_workers=cfg_get(rollout_node, "num_workers", "auto"),
        gpu_pool=rollout_gpu_pool,
    )
    if reward_node is _MISSING:
        reward = RewardResourceConfig(num_gpus=0, devices=[])
    else:
        reward = RewardResourceConfig(
            num_gpus=cfg_get(reward_node, "num_gpus", "auto"),
            devices=_parse_devices(cfg_get(reward_node, "devices", "auto")),
            gpus_per_worker=float(cfg_get(reward_node, "gpus_per_worker", 1.0)),
            num_workers=cfg_get(reward_node, "num_workers", "auto"),
            gpu_pool=_parse_reward_gpu_pool(reward_node),
        )
    return DistributedResourceConfig(
        visible_devices=_parse_devices(cfg_get(resources, "visible_devices", "auto")),
        trainer=trainer,
        rollout=rollout,
        reward=reward,
        allow_overlap=bool(cfg_get(resources, "allow_overlap", False)),
        reward_cpus_per_worker=float(
            cfg_get(reward_runtime, "cpus_per_worker", 0.5),
        ),
        cross_node=bool(cfg_get(resources, "cross_node", False)),
    )


def _resolve_cross_node_visible_devices(
    config: DistributedResourceConfig,
) -> tuple[int, ...]:
    """Synthesise a visible-device budget for cross-node runs.

    Resolution runs before ``ray.init()`` (see ``vrl/scripts/common/online.py``),
    so the live Ray cluster cannot be queried here. Under ``cross_node`` we instead
    size the visible pool from explicit per-role GPU counts. The resulting ordinals
    are budget tokens only: trainer keeps the head-local ordinal, rollout ordinals
    are never used as real remote device ids (placement is by Ray + node, and
    ``validate_actor_gpu_ids`` switches to a node-aware check).
    """

    explicit = _parse_devices(config.visible_devices)
    if explicit != "auto":
        return tuple(
            _dedupe_ints(explicit, field_name="distributed.resources.visible_devices"),
        )

    total = (
        _explicit_role_gpu_count("trainer", config.trainer)
        + _explicit_role_gpu_count("rollout", config.rollout)
        + _explicit_role_gpu_count("reward", config.reward)
    )
    return tuple(range(total))


def _explicit_role_gpu_count(role: str, role_config: RoleResourceConfig) -> int:
    """Return an explicit integer GPU count for a role under ``cross_node``."""

    devices = _parse_devices(role_config.devices)
    if devices != "auto":
        return len(_dedupe_ints(devices, field_name=f"distributed.resources.{role}.devices"))

    num_gpus = _parse_num_gpus(role_config.num_gpus, field_name=f"{role}.num_gpus")
    if num_gpus == "auto" or num_gpus is None:
        raise ValueError(
            "distributed.resources.cross_node=true requires an explicit integer "
            f"distributed.resources.{role}.num_gpus (got 'auto'/null): the Ray "
            "cluster is not queryable at resolution time, so the GPU budget must be "
            "declared up front.",
        )
    if int(num_gpus) < 0:
        raise ValueError(f"distributed.resources.{role}.num_gpus must be >= 0")
    return int(num_gpus)


def _resolve_role_devices(
    *,
    role: str,
    visible_devices: tuple[int, ...],
    role_config: RoleResourceConfig,
    default_auto_count: int,
) -> tuple[int, ...]:
    explicit_devices = _parse_devices(role_config.devices)
    num_gpus = _parse_num_gpus(role_config.num_gpus, field_name=f"{role}.num_gpus")

    if explicit_devices != "auto":
        devices = tuple(_dedupe_ints(explicit_devices, field_name=f"{role}.devices"))
        _validate_subset(devices, visible_devices, field_name=f"{role}.devices")
        if num_gpus != "auto" and num_gpus is not None and int(num_gpus) != len(devices):
            raise ValueError(
                f"distributed.resources.{role}.num_gpus={num_gpus} does not match "
                f"len(distributed.resources.{role}.devices)={len(devices)}",
            )
        return devices

    count = default_auto_count if num_gpus == "auto" or num_gpus is None else int(num_gpus)
    if count < 0:
        raise ValueError(f"distributed.resources.{role}.num_gpus must be >= 0")
    if count > len(visible_devices):
        raise ValueError(
            f"distributed.resources.{role}.num_gpus={count} exceeds visible devices "
            f"{list(visible_devices)}",
        )
    return tuple(visible_devices[:count])


def _resolve_rollout_devices(
    *,
    visible_devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_config: RolloutResourceConfig,
    allow_overlap: bool,
    gpu_pool: str,
) -> tuple[int, ...]:
    explicit_devices = _parse_devices(rollout_config.devices)
    num_gpus = _parse_num_gpus(
        rollout_config.num_gpus,
        field_name="rollout.num_gpus",
    )

    if explicit_devices != "auto":
        devices = tuple(
            _dedupe_ints(explicit_devices, field_name="distributed.resources.rollout.devices"),
        )
        _validate_subset(
            devices,
            visible_devices,
            field_name="distributed.resources.rollout.devices",
        )
        if num_gpus != "auto" and num_gpus is not None and int(num_gpus) != len(devices):
            raise ValueError(
                "distributed.resources.rollout.num_gpus does not match "
                f"len(distributed.resources.rollout.devices): {num_gpus} vs {len(devices)}",
            )
        trainer_pool = set(trainer_devices)
        if gpu_pool == "trainer" and not set(devices).issubset(trainer_pool):
            outside = sorted(set(devices) - trainer_pool)
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=trainer requires every "
                "rollout device to belong to the trainer pool, but "
                f"rollout.devices={list(devices)} includes {outside} outside "
                f"trainer={list(trainer_devices)}. Devices disjoint from trainer "
                "or mixing trainer and spare GPUs are invalid; drop the explicit "
                "devices (auto pins onto the trainer pool) or use "
                "gpu_pool=dedicated.",
            )
        trainer_overlap = sorted(set(devices) & trainer_pool)
        if gpu_pool == "dedicated" and trainer_overlap:
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=dedicated requires rollout "
                "devices disjoint from the trainer pool, but "
                f"rollout={list(devices)} trainer={list(trainer_devices)} "
                f"overlap={trainer_overlap}",
            )
        return devices

    if gpu_pool == "trainer":
        # Borrow the trainer GPU(s): pin rollout onto them even when spare GPUs exist
        # ("share the trainer card", not "find a free one").
        requested = _requested_role_gpu_count(
            role="rollout",
            num_gpus=rollout_config.num_gpus,
            num_workers=rollout_config.num_workers,
            gpus_per_worker=rollout_config.gpus_per_worker,
            available_count=len(trainer_devices),
        )
        if requested == 0:
            return ()
        if requested > len(trainer_devices):
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=trainer shares the trainer "
                f"GPU(s), but rollout needs {requested} GPU(s) and trainer owns "
                f"{list(trainer_devices)}. A shared rollout cannot exceed the trainer "
                "pool; use gpu_pool=dedicated (or expose more trainer GPUs).",
            )
        return tuple(trainer_devices[:requested])

    # auto / dedicated: a pool disjoint from the trainer GPU(s). `dedicated` forbids
    # the overlap fallback (a spare GPU is required); `auto` allows it under allow_overlap.
    excluded = set(trainer_devices)
    pool = tuple(device for device in visible_devices if device not in excluded)
    requested = _requested_role_gpu_count(
        role="rollout",
        num_gpus=rollout_config.num_gpus,
        num_workers=rollout_config.num_workers,
        gpus_per_worker=rollout_config.gpus_per_worker,
        available_count=len(pool),
    )
    return _slice_pool_with_overlap_fallback(
        requested=requested,
        pool=pool,
        excluded=excluded,
        visible_devices=visible_devices,
        allow_overlap=allow_overlap and gpu_pool != "dedicated",
        not_enough_pool_error=(
            "Not enough non-overlapping rollout GPUs: "
            f"requested={requested}, available={len(pool)}, "
            f"trainer={list(trainer_devices)}, visible={list(visible_devices)}. "
            "Expose more GPUs, or set distributed.resources.rollout.gpu_pool=trainer "
            "to time-share the trainer GPU."
        ),
        not_enough_visible_error=(
            "Not enough visible GPUs for rollout even with overlap allowed: "
            f"requested={requested}, visible={list(visible_devices)}"
        ),
    )


def _slice_pool_with_overlap_fallback(
    *,
    requested: int,
    pool: tuple[int, ...],
    excluded: set[int],
    visible_devices: tuple[int, ...],
    allow_overlap: bool,
    not_enough_pool_error: str,
    not_enough_visible_error: str,
) -> tuple[int, ...]:
    """Take ``requested`` devices from the non-overlapping ``pool``.

    Shared by the rollout and reward auto-allocation paths. When the pool is
    too small, fall back to the ``excluded`` devices (overlapping the trainer /
    rollout) only if ``allow_overlap`` is set; otherwise raise. Callers pass the
    role-specific error strings so each message keeps pointing at the right knob.
    """

    if requested == 0:
        return ()
    if requested <= len(pool):
        return tuple(pool[:requested])
    if not allow_overlap:
        raise ValueError(not_enough_pool_error)
    fallback = tuple(device for device in visible_devices if device in excluded)
    combined = pool + fallback
    if requested > len(combined):
        raise ValueError(not_enough_visible_error)
    return tuple(combined[:requested])


def _resolve_rollout_num_workers(
    *,
    rollout_config: RolloutResourceConfig,
    rollout_num_gpus: int,
    gpus_per_worker: float,
) -> int:
    requested = _parse_num_workers(
        rollout_config.num_workers,
        allow_zero=gpus_per_worker == 0 and rollout_num_gpus == 0,
    )
    if gpus_per_worker == 0:
        workers = 1 if requested == "auto" else int(requested)
        if workers < 0:
            raise ValueError("distributed.resources.rollout.num_workers must be >= 0")
        return workers

    if requested == "auto":
        workers_float = rollout_num_gpus / gpus_per_worker
        if int(workers_float) != workers_float:
            raise ValueError(
                "distributed.resources.rollout.num_gpus must be divisible by "
                "distributed.resources.rollout.gpus_per_worker",
            )
        workers = int(workers_float)
    else:
        workers = int(requested)
        expected_gpus = int(workers * gpus_per_worker)
        if expected_gpus != rollout_num_gpus:
            raise ValueError(
                "distributed.resources.rollout.num_workers * gpus_per_worker must "
                f"equal rollout GPU count: {workers} * {gpus_per_worker:g} "
                f"!= {rollout_num_gpus}",
            )

    if workers < 1:
        raise ValueError("distributed.resources.rollout.num_workers must be >= 1")
    return workers


def _resolve_reward_devices(
    *,
    visible_devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_config: RewardResourceConfig,
    allow_overlap: bool,
) -> tuple[int, ...]:
    explicit_devices = _parse_devices(reward_config.devices)
    num_gpus = _parse_num_gpus(
        reward_config.num_gpus,
        field_name="reward.num_gpus",
    )

    if explicit_devices != "auto":
        devices = tuple(
            _dedupe_ints(explicit_devices, field_name="distributed.resources.reward.devices"),
        )
        _validate_subset(
            devices,
            visible_devices,
            field_name="distributed.resources.reward.devices",
        )
        if num_gpus != "auto" and num_gpus is not None and int(num_gpus) != len(devices):
            raise ValueError(
                "distributed.resources.reward.num_gpus does not match "
                f"len(distributed.resources.reward.devices): {num_gpus} vs {len(devices)}",
            )
        _validate_reward_overlap(
            devices=devices,
            trainer_devices=trainer_devices,
            rollout_devices=rollout_devices,
            reward_config=reward_config,
            allow_overlap=allow_overlap,
        )
        return devices

    pool_source = reward_config.gpu_pool
    if pool_source == "auto":
        # Auto placement: prefer a dedicated spare GPU when the visible pool
        # can satisfy the request; otherwise fall back to sharing the rollout
        # pool. Removes the footgun where a spelled-out "rollout" kept forcing
        # shared single-GPU churn even on machines with spare GPUs.
        spare_excluded = set(trainer_devices) | set(rollout_devices)
        spare_pool = tuple(device for device in visible_devices if device not in spare_excluded)
        spare_requested = _requested_role_gpu_count(
            role="reward",
            num_gpus=reward_config.num_gpus,
            num_workers=reward_config.num_workers,
            gpus_per_worker=reward_config.gpus_per_worker,
            available_count=len(spare_pool),
        )
        if spare_requested > 0 and len(spare_pool) >= spare_requested:
            devices = tuple(spare_pool[:spare_requested])
            _validate_reward_overlap(
                devices=devices,
                trainer_devices=trainer_devices,
                rollout_devices=rollout_devices,
                reward_config=reward_config,
                allow_overlap=allow_overlap,
            )
            return devices
        pool_source = "rollout"

    if pool_source == "rollout":
        requested = _requested_role_gpu_count(
            role="reward",
            num_gpus=reward_config.num_gpus,
            num_workers=reward_config.num_workers,
            gpus_per_worker=reward_config.gpus_per_worker,
            available_count=len(rollout_devices),
        )
        if requested > len(rollout_devices):
            raise ValueError(
                "Not enough rollout GPUs for reward shared inference pool: "
                f"requested={requested}, rollout={list(rollout_devices)}",
            )
        devices = tuple(rollout_devices[:requested])
        _validate_reward_overlap(
            devices=devices,
            trainer_devices=trainer_devices,
            rollout_devices=rollout_devices,
            reward_config=reward_config,
            allow_overlap=allow_overlap,
        )
        return devices

    excluded = set(trainer_devices) | set(rollout_devices)
    pool = tuple(device for device in visible_devices if device not in excluded)
    requested = _requested_role_gpu_count(
        role="reward",
        num_gpus=reward_config.num_gpus,
        num_workers=reward_config.num_workers,
        gpus_per_worker=reward_config.gpus_per_worker,
        available_count=len(pool),
    )
    devices = _slice_pool_with_overlap_fallback(
        requested=requested,
        pool=pool,
        excluded=excluded,
        visible_devices=visible_devices,
        # This branch is the explicit dedicated pool. ``allow_overlap`` permits
        # auto placement to share, but cannot weaken a declared dedicated pool.
        allow_overlap=False,
        not_enough_pool_error=(
            "Not enough non-overlapping reward GPUs: "
            f"requested={requested}, available={len(pool)}, "
            f"trainer={list(trainer_devices)}, rollout={list(rollout_devices)}, "
            f"visible={list(visible_devices)}. Set "
            "distributed.resources.reward.gpu_pool=rollout for a shared "
            "inference pool (release is derived automatically), or expose a "
            "separate reward GPU."
        ),
        not_enough_visible_error=(
            "Not enough visible GPUs for reward even with overlap allowed: "
            f"requested={requested}, visible={list(visible_devices)}"
        ),
    )
    if not devices:
        return ()
    _validate_reward_overlap(
        devices=devices,
        trainer_devices=trainer_devices,
        rollout_devices=rollout_devices,
        reward_config=reward_config,
        allow_overlap=allow_overlap,
    )
    return devices


def _validate_reward_overlap(
    *,
    devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_config: RewardResourceConfig,
    allow_overlap: bool,
) -> None:
    device_set = set(devices)
    rollout_pool = set(rollout_devices)
    trainer_pool = set(trainer_devices)

    if reward_config.gpu_pool == "rollout" and not device_set.issubset(rollout_pool):
        outside = sorted(device_set - rollout_pool)
        raise ValueError(
            "distributed.resources.reward.gpu_pool=rollout requires every reward "
            f"device to belong to rollout={list(rollout_devices)}, but "
            f"reward={list(devices)} includes {outside} outside that pool",
        )

    rollout_overlap = sorted(device_set & rollout_pool)
    trainer_overlap = sorted(device_set & trainer_pool)
    if reward_config.gpu_pool == "dedicated" and (rollout_overlap or trainer_overlap):
        raise ValueError(
            "distributed.resources.reward.gpu_pool=dedicated requires reward "
            "devices disjoint from both trainer and rollout pools: "
            f"reward={list(devices)} trainer={list(trainer_devices)} "
            f"rollout={list(rollout_devices)}",
        )

    if trainer_overlap and not allow_overlap:
        raise ValueError(
            "Trainer and reward devices overlap but "
            "distributed.resources.allow_overlap=false: "
            f"trainer={list(trainer_devices)} reward={list(devices)}",
        )


def _requested_role_gpu_count(
    *,
    role: str,
    num_gpus: int | str | None,
    num_workers: int | str,
    gpus_per_worker: float,
    available_count: int,
) -> int:
    parsed_num_gpus = _parse_num_gpus(num_gpus, field_name=f"{role}.num_gpus")
    if parsed_num_gpus != "auto" and parsed_num_gpus is not None:
        count = int(parsed_num_gpus)
        if count < 0:
            raise ValueError(f"distributed.resources.{role}.num_gpus must be >= 0")
        return count
    if float(gpus_per_worker) == 0.0:
        return 0

    parsed_workers = _parse_num_workers(num_workers, role=role)
    if parsed_workers != "auto":
        return int(parsed_workers * float(gpus_per_worker))
    return int(available_count)


def _resolve_role_num_workers(
    *,
    role: str,
    num_workers: int | str,
    num_gpus: int,
    gpus_per_worker: float,
) -> int:
    requested = _parse_num_workers(num_workers, role=role)
    if gpus_per_worker == 0:
        workers = 1 if requested == "auto" else int(requested)
        if workers < 1:
            raise ValueError(f"distributed.resources.{role}.num_workers must be >= 1")
        return workers

    if num_gpus == 0 and requested == "auto":
        return 0

    if requested == "auto":
        workers_float = num_gpus / gpus_per_worker
        if int(workers_float) != workers_float:
            raise ValueError(
                f"distributed.resources.{role}.num_gpus must be divisible by "
                f"distributed.resources.{role}.gpus_per_worker",
            )
        workers = int(workers_float)
    else:
        workers = int(requested)
        expected_gpus = int(workers * gpus_per_worker)
        if expected_gpus != num_gpus:
            raise ValueError(
                f"distributed.resources.{role}.num_workers * gpus_per_worker must "
                f"equal {role} GPU count: {workers} * {gpus_per_worker:g} "
                f"!= {num_gpus}",
            )

    if workers < 1:
        raise ValueError(f"distributed.resources.{role}.num_workers must be >= 1")
    return workers


@dataclass(frozen=True, slots=True)
class BundleLayout:
    """Run-level mapping of execution roles to placement-group bundle indices.

    One GPU bundle per distinct device the run touches; one CPU bundle per
    CPU-only worker. This is the source of truth a single run-level placement
    group is built from, replacing the per-role private placement groups plus
    the reward ``gpu_reservation_count`` offset math: a role targets its own
    bundle index directly instead of pinning the slots beneath it.

    ``bundle_gpu_ids[i]`` is the GPU ordinal bundle ``i`` reserves, or ``None``
    for a CPU-only bundle. Trainer bundles are *reserved* (no actor runs in
    them) purely to keep the driver GPU out of Ray's scheduling pool. Two roles
    that resolve to the same physical GPU share one bundle index (coalescing) --
    that overlap *is* the "shared GPU" fact, read it off the indices rather than
    storing a separate flag.
    """

    bundle_gpu_ids: tuple[int | None, ...]
    rollout_bundle_indices: tuple[int, ...]
    reward_bundle_indices: tuple[int, ...]

    @property
    def total_bundles(self) -> int:
        return len(self.bundle_gpu_ids)


def build_bundle_layout(resolved: ResolvedDistributedResources) -> BundleLayout:
    """Derive a run-level role->bundle plan from a resolved resource plan.

    GPU roles share one bundle per physical device: a trainer-reserved GPU, a
    rollout GPU, and a reward GPU that lands on the same device all collapse to
    a single bundle. CPU-only roles (``gpus_per_worker == 0``) get one bundle
    per worker. The owner probes the live placement group to learn which bundle
    index maps to which GPU, so the ordinals here are the *requested* devices
    the probe is validated against.
    """

    bundle_gpu_ids: list[int | None] = []
    gpu_bundle_by_id: dict[int, int] = {}

    def _gpu_bundle(gpu_id: int) -> int:
        index = gpu_bundle_by_id.get(gpu_id)
        if index is None:
            index = len(bundle_gpu_ids)
            bundle_gpu_ids.append(gpu_id)
            gpu_bundle_by_id[gpu_id] = index
        return index

    def _cpu_bundles(count: int) -> tuple[int, ...]:
        indices: list[int] = []
        for _ in range(count):
            indices.append(len(bundle_gpu_ids))
            bundle_gpu_ids.append(None)
        return tuple(indices)

    # Trainer reserved bundles first so the driver GPU is protected before any
    # actor role can claim a bundle (single-node dedicated-trainer plans only;
    # colocated and cross-node set requires_trainer_reservation=False).
    if resolved.requires_trainer_reservation:
        for gpu_id in resolved.trainer_devices:
            _gpu_bundle(gpu_id)

    if resolved.rollout_gpus_per_worker > 0:
        rollout = tuple(_gpu_bundle(gpu_id) for gpu_id in resolved.rollout_devices)
    else:
        rollout = _cpu_bundles(resolved.rollout_num_workers)

    if resolved.reward_gpus_per_worker > 0:
        # Shared reward reuses the rollout GPU's existing bundle index; a
        # dedicated reward GPU appends a fresh bundle.
        reward = tuple(_gpu_bundle(gpu_id) for gpu_id in resolved.reward_devices)
    else:
        reward = _cpu_bundles(resolved.reward_num_workers)

    return BundleLayout(
        bundle_gpu_ids=tuple(bundle_gpu_ids),
        rollout_bundle_indices=rollout,
        reward_bundle_indices=reward,
    )


def _parse_rollout_gpu_pool(
    *,
    rollout_node: Any,
) -> str:
    """Resolve the rollout GPU pool and reject the removed resident-share knob.

    Single authoritative grammar (mirrors ``reward.gpu_pool``):
    ``distributed.resources.rollout.gpu_pool`` = ``auto|trainer|dedicated``.
    ``trainer`` always means on-demand phase handoff; sharing memory persistently is
    no longer a supported topology.
    """

    new_pool = cfg_get(rollout_node, "gpu_pool", _MISSING)
    if cfg_get(rollout_node, "memory_fraction", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.resources.rollout.memory_fraction was removed: shared "
            "trainer/rollout GPUs always use on-demand phase handoff. Remove the "
            "key and keep gpu_pool=trainer for time-shared colocation.",
        )

    pool = "auto"
    if new_pool is not _MISSING:
        pool = str(to_builtin(new_pool)).strip().lower()
        if pool not in {"auto", "trainer", "dedicated"}:
            raise ValueError(
                "distributed.resources.rollout.gpu_pool must be 'auto', 'trainer', "
                f"or 'dedicated', got {new_pool!r}",
            )
    return pool


def _parse_reward_gpu_pool(reward_node: Any) -> str:
    """Resolve the reward GPU-pool source from ``reward.gpu_pool`` (new) or the
    legacy ``reward.share_with_rollout`` bool.

    New configs use ``distributed.resources.reward.gpu_pool: auto|rollout|dedicated``.
    ``share_with_rollout`` is accepted as compat input (null -> auto, true ->
    rollout, false -> dedicated). Setting both is an error.
    """

    gpu_pool = cfg_get(reward_node, "gpu_pool", _MISSING)
    legacy = cfg_get(reward_node, "share_with_rollout", _MISSING)
    if gpu_pool is not _MISSING and legacy is not _MISSING:
        raise ValueError(
            "set either distributed.resources.reward.gpu_pool "
            "(auto|rollout|dedicated) or the legacy share_with_rollout, not both",
        )
    if gpu_pool is not _MISSING:
        value = str(to_builtin(gpu_pool)).strip().lower()
        if value not in {"auto", "rollout", "dedicated"}:
            raise ValueError(
                "distributed.resources.reward.gpu_pool must be 'auto', 'rollout', "
                f"or 'dedicated', got {gpu_pool!r}",
            )
        return value
    legacy_value = legacy if legacy is not _MISSING else None
    if legacy_value is None:
        return "auto"
    return "rollout" if bool(legacy_value) else "dedicated"


def _reject_removed_distributed_keys(rollout_runtime: Any, reward_runtime: Any) -> None:
    """Hard-fail on the public release/colocation keys removed by the config sweep.

    Release scheduling is derived from GPU topology. Shared trainer/rollout GPUs
    always hand ownership over between phases; a silent compatibility shim would
    re-introduce the multi-knob resident surface that was removed.
    """

    if cfg_get(rollout_runtime, "release_after_collect", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.rollout.release_after_collect was removed: rollout release is "
            "now derived from the GPU topology (shared GPU -> release, dedicated GPU -> "
            "resident). Remove the key; use "
            "distributed.resources.rollout.gpu_pool=trainer for time-shared colocation.",
        )
    if cfg_get(rollout_runtime, "release_before_reward_model", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.rollout.release_before_reward_model was removed: it is now "
            "derived from the GPU topology (reward sharing the rollout GPU -> release). "
            "Remove the key.",
        )
    if cfg_get(rollout_runtime, "persistent_colocated_workers", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.rollout.persistent_colocated_workers was removed: shared "
            "trainer/rollout GPUs always use on-demand phase handoff. Remove the key.",
        )
    if cfg_get(rollout_runtime, "gpu_memory_fraction", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.rollout.gpu_memory_fraction was removed with resident "
            "shared-GPU colocation. Remove the key.",
        )
    if cfg_get(rollout_runtime, "colocate", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.rollout.colocate was removed: use "
            "distributed.resources.rollout.gpu_pool=trainer for time-shared "
            "colocation. Resident shared-GPU colocation is no longer supported.",
        )
    if cfg_get(reward_runtime, "release_after_score", _MISSING) is not _MISSING:
        raise ValueError(
            "distributed.reward.release_after_score was removed: reward release is now "
            "derived from the GPU topology (shared / multi-reward pool -> release, "
            "dedicated reward GPU -> resident). Remove the key.",
        )
    for stale_pool_key in ("placement_strategy", "max_inflight_batches"):
        if cfg_get(reward_runtime, stale_pool_key, _MISSING) is not _MISSING:
            raise ValueError(
                f"distributed.reward.{stale_pool_key} was removed with the Ray reward "
                "actor pool: rewards score in-process now. Remove the key and "
                "select shared or dedicated ownership under "
                "distributed.resources.reward; parking follows that topology.",
            )


def _parse_devices(value: Any) -> list[int] | str:
    value = to_builtin(value)
    if _is_auto(value):
        return "auto"
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid device list: {value!r}") from exc
        return _parse_devices(parsed)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"invalid device list: {value!r}")


def _parse_num_gpus(value: Any, *, field_name: str) -> int | str | None:
    value = to_builtin(value)
    if _is_auto(value):
        return "auto"
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"distributed.resources.{field_name} must be int, auto, or null") from exc
    return parsed


def _parse_num_workers(
    value: Any,
    *,
    role: str = "rollout",
    allow_zero: bool = False,
) -> int | str:
    value = to_builtin(value)
    if _is_auto(value):
        return "auto"
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"distributed.resources.{role}.num_workers must be int or auto",
        ) from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        minimum = 0 if allow_zero else 1
        raise ValueError(f"distributed.resources.{role}.num_workers must be >= {minimum}")
    return parsed


def _dedupe_ints(values: list[int], *, field_name: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item < 0:
            raise ValueError(f"{field_name} cannot contain negative device ids")
        if item in seen:
            raise ValueError(f"{field_name} contains duplicate device id {item}")
        seen.add(item)
        out.append(item)
    return out


def _validate_subset(
    devices: tuple[int, ...],
    visible_devices: tuple[int, ...],
    *,
    field_name: str,
) -> None:
    missing = sorted(set(devices) - set(visible_devices))
    if missing:
        raise ValueError(
            f"{field_name} contains devices outside distributed.resources.visible_devices: "
            f"{missing} not in {list(visible_devices)}",
        )


def _auto_visible_cuda_devices() -> tuple[int, ...]:
    try:
        import torch
    except Exception:
        return ()
    try:
        if not torch.cuda.is_available():
            return ()
        return tuple(range(int(torch.cuda.device_count())))
    except Exception:
        return ()


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


__all__ = [
    "ActorLeasePolicy",
    "BundleLayout",
    "DistributedResourceConfig",
    "PhaseHandoffPolicy",
    "RayLifecyclePlan",
    "ResolvedDistributedResources",
    "RewardResourceConfig",
    "RoleResourceConfig",
    "RolloutResourceConfig",
    "build_bundle_layout",
    "format_distributed_resource_plan",
    "resolve_distributed_resources",
    "reward_torch_device",
    "trainer_torch_device",
]
