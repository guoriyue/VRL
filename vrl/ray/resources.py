"""Role-level resource resolution for distributed VRL runs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.utils.config import cfg_get


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


@dataclass(frozen=True, slots=True)
class RewardResourceConfig(RoleResourceConfig):
    """GPU ownership request for reward inference workers."""

    gpus_per_worker: float = 1.0
    num_workers: int | str = "auto"
    # Tri-state placement preference: None derives from GPU topology (use a
    # dedicated spare GPU when one exists, otherwise share the rollout pool);
    # explicit true forces sharing, explicit false forces a dedicated GPU.
    share_with_rollout: bool | None = None


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
    # Release lifecycle flags are tri-state: None means "derive from the
    # resolved GPU topology" (overlap -> release, dedicated GPUs -> resident).
    # Explicit values that contradict a shared topology fail in validation.
    rollout_release_after_collect: bool | None = None
    rollout_release_before_reward_model: bool | None = None
    rollout_persistent_colocated_workers: bool = False
    # Hard cap on the rollout worker's share of its GPU, in (0, 1]. None means
    # "no cap" (the worker owns its whole dedicated GPU). Required when
    # persistent_colocated_workers is set: the worker then shares the trainer GPU,
    # so its allocator must be bounded to leave the trainer room.
    rollout_gpu_memory_fraction: float | None = None
    reward_release_after_score: bool | None = None
    reward_placement_strategy: str = "SPREAD"
    reward_cpus_per_worker: float = 0.5
    reward_max_inflight_batches: int = 1
    cross_node: bool = False


@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    """How long a role's Ray actors stay alive across phases.

    ``resident`` actors stay up across phases (the role owns a dedicated GPU,
    or is the persistent colocated debug worker). ``on_demand`` actors are
    released at a phase handoff and reacquired on next use, because the role
    shares a GPU it must hand back.
    """

    mode: Literal["resident", "on_demand"]


@dataclass(frozen=True, slots=True)
class PhaseHandoffPolicy:
    """Which resident-vs-shared actors must step off their GPU at each boundary.

    A flag is True only when two roles share a GPU and the next phase needs the
    first to release it. Derived once from topology so no runtime re-decides it
    per call.
    """

    release_rollout_before_train: bool
    release_rollout_before_reward: bool
    release_reward_after_score: bool


@dataclass(frozen=True, slots=True)
class RayLifecyclePlan:
    """Single topology-derived answer to "which Ray actors release when".

    Built by :func:`resolve_distributed_resources` from GPU ownership so the
    launcher, collector, and reward runtime read one declarative plan instead of
    each re-deriving ``release_after_*`` from raw device sets. The flat
    ``rollout_release_* / reward_release_*`` fields on
    :class:`ResolvedDistributedResources` are compatibility views derived beside
    this plan for existing consumers.
    """

    rollout: ActorLeasePolicy
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
    rollout_num_gpus: int
    rollout_num_workers: int
    rollout_gpus_per_worker: float
    reward_num_workers: int
    reward_gpus_per_worker: float
    reward_shared_with_rollout: bool
    rollout_release_after_collect: bool
    rollout_release_before_reward_model: bool
    rollout_persistent_colocated_workers: bool
    rollout_gpu_memory_fraction: float | None
    reward_release_after_score: bool
    reward_placement_strategy: str
    reward_cpus_per_worker: float
    reward_max_inflight_batches: int
    requires_trainer_reservation: bool
    colocated: bool
    cross_node: bool
    # Named view over the old release flags: lease mode per role plus the
    # per-boundary handoff. The launcher/collector/reward read this instead of
    # re-deriving release decisions from device sets (the flat flags stay as a
    # compatibility view for existing consumers).
    lifecycle: RayLifecyclePlan


_MISSING = object()


def resolve_distributed_resources(cfg: Any) -> ResolvedDistributedResources:
    """Resolve role-level resource config into concrete CUDA ordinals.

    This is the single source of truth for trainer/rollout/reward GPU
    ownership. It intentionally does static ownership checks only; memory
    pressure is still a runtime concern.
    """

    config = _distributed_resource_config_from_cfg(cfg)
    training = cfg_get(cfg_get(cfg, "distributed", {}), "training", {})
    training_strategy = str(cfg_get(training, "strategy", "single_process"))
    training_world_size = int(cfg_get(training, "num_nodes", 1)) * int(
        cfg_get(training, "gpus_per_node", 1),
    )
    if config.cross_node:
        visible_devices = _resolve_cross_node_visible_devices(config)
    else:
        visible_devices = _resolve_visible_devices(config.visible_devices)

    # fsdp runs one torchrun process per GPU, so an unset trainer GPU count
    # defaults to the whole training world; single_process stays one device.
    trainer_default_auto = (
        training_world_size
        if training_strategy == "fsdp"
        else (1 if visible_devices else 0)
    )
    trainer_devices = _resolve_role_devices(
        role="trainer",
        visible_devices=visible_devices,
        role_config=config.trainer,
        default_auto_count=trainer_default_auto,
    )
    _validate_trainer_device_count(training_strategy, trainer_devices, training_world_size)

    rollout_gpus_per_worker = float(config.rollout.gpus_per_worker)
    if rollout_gpus_per_worker not in {0.0, 1.0}:
        raise ValueError(
            "distributed.resources.rollout.gpus_per_worker currently supports "
            f"0 or 1, got {rollout_gpus_per_worker}",
        )

    rollout_devices = _resolve_rollout_devices(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_config=config.rollout,
        allow_overlap=config.allow_overlap,
    )
    rollout_num_gpus = len(rollout_devices)

    if rollout_gpus_per_worker > 0 and rollout_num_gpus == 0:
        raise ValueError(
            "No rollout GPUs are available after reserving trainer devices "
            f"{list(trainer_devices)} with distributed.resources.allow_overlap=false. "
            "Expose more GPUs or set distributed.resources.allow_overlap=true with "
            "distributed.rollout.release_after_collect=true for single-GPU debug.",
        )

    rollout_num_workers = _resolve_rollout_num_workers(
        rollout_config=config.rollout,
        rollout_num_gpus=rollout_num_gpus,
        gpus_per_worker=rollout_gpus_per_worker,
    )

    colocated = bool(set(trainer_devices) & set(rollout_devices))
    if colocated and not config.allow_overlap:
        raise ValueError(
            "Trainer and rollout devices overlap but "
            "distributed.resources.allow_overlap=false: "
            f"trainer={list(trainer_devices)} rollout={list(rollout_devices)}",
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
        rollout_release_after_collect=config.rollout_release_after_collect,
        rollout_release_before_reward_model=config.rollout_release_before_reward_model,
        reward_release_after_score=config.reward_release_after_score,
    )
    reward_num_gpus = len(reward_devices)
    ray_reward_count = _count_ray_rewards(cfg)
    if ray_reward_count > 0 and reward_gpus_per_worker > 0 and reward_num_gpus == 0:
        raise ValueError(
            "reward component with execution=pool requires "
            "distributed.resources.reward.num_gpus > 0",
        )
    reward_num_workers = _resolve_role_num_workers(
        role="reward",
        num_workers=config.reward.num_workers,
        num_gpus=reward_num_gpus,
        gpus_per_worker=reward_gpus_per_worker,
    )
    if reward_gpus_per_worker > 0 and reward_num_gpus == 0 and reward_num_workers > 0:
        raise ValueError(
            "distributed.resources.reward requested GPU workers but no reward GPUs "
            "were resolved",
        )

    reward_shared_with_rollout = bool(set(reward_devices) & set(rollout_devices))
    reward_overlaps_trainer = bool(set(reward_devices) & set(trainer_devices))
    if reward_overlaps_trainer and not config.allow_overlap:
        raise ValueError(
            "Trainer and reward devices overlap but "
            "distributed.resources.allow_overlap=false: "
            f"trainer={list(trainer_devices)} reward={list(reward_devices)}",
        )

    # Under fsdp every rank owns its GPU; rollout/reward run as separate Ray
    # actors. The trainer GPU set must be disjoint from both regardless of
    # allow_overlap (the single-GPU colocated debug path is single_process only).
    if training_strategy == "fsdp":
        _validate_fsdp_trainer_disjoint(trainer_devices, rollout_devices, reward_devices)

    if config.rollout_persistent_colocated_workers and not colocated:
        raise ValueError(
            "distributed.rollout.persistent_colocated_workers=true requires "
            "trainer and rollout devices to overlap",
        )
    if config.rollout_persistent_colocated_workers and reward_shared_with_rollout:
        raise ValueError(
            "distributed.rollout.persistent_colocated_workers=true cannot share "
            "the rollout GPU with a Ray reward worker",
        )
    if (
        config.rollout_persistent_colocated_workers
        and config.rollout_release_after_collect is True
    ):
        raise ValueError(
            "distributed.rollout.persistent_colocated_workers=true requires "
            "distributed.rollout.release_after_collect=false",
        )
    gpu_memory_fraction = config.rollout_gpu_memory_fraction
    if gpu_memory_fraction is not None and not 0.0 < gpu_memory_fraction <= 1.0:
        raise ValueError(
            "distributed.rollout.gpu_memory_fraction must be in (0, 1] when set, got "
            f"{gpu_memory_fraction}",
        )
    if config.rollout_persistent_colocated_workers and gpu_memory_fraction is None:
        # Resident worker + resident trainer on one GPU: without a cap the worker's
        # allocator can grow until the trainer's backward OOMs. Force an explicit
        # budget so the split is declared, not hoped for (cf. vLLM/cosmos-rl
        # gpu_memory_utilization on the colocated rollout backend).
        raise ValueError(
            "distributed.rollout.persistent_colocated_workers=true requires an "
            "explicit distributed.rollout.gpu_memory_fraction in (0, 1): the rollout "
            "worker shares the trainer GPU, so cap its share to leave the trainer room",
        )

    # Unset release flags follow the resolved topology: roles that share a GPU
    # must hand it over between phases; roles with dedicated GPUs stay resident.
    # Keep the old flat "release_after_collect" field as a compatibility view,
    # but derive the named lifecycle handoffs per boundary so reward/rollout
    # sharing is not mislabeled as a trainer handoff.
    rollout_release_before_train = _derived_release_flag(
        config.rollout_release_after_collect,
        derived=colocated
        and not config.rollout_persistent_colocated_workers,
    )
    rollout_release_before_reward_model = _derived_release_flag(
        config.rollout_release_before_reward_model,
        derived=reward_shared_with_rollout,
    )
    reward_release_after_score = _derived_release_flag(
        config.reward_release_after_score,
        derived=reward_shared_with_rollout
        or (ray_reward_count > 1 and reward_gpus_per_worker > 0),
    )
    if (
        ray_reward_count > 1
        and reward_gpus_per_worker > 0
        and not reward_release_after_score
    ):
        raise ValueError(
            "multiple active reward components with execution=pool share "
            "the reward role placement; set distributed.reward.release_after_score "
            "to true (or leave it unset) until per-reward placement is supported",
        )
    rollout_release_after_collect = (
        rollout_release_before_train or rollout_release_before_reward_model
    )

    requires_trainer_reservation = (
        bool(trainer_devices)
        and rollout_gpus_per_worker > 0
        and not colocated
        and rollout_num_workers > 0
        and not config.cross_node
    )
    # One derivation feeds both the named plan and the flat compatibility
    # fields: a role is on_demand when any handoff can drop it, while the plan
    # keeps the specific phase boundary explicit.
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(
            mode="on_demand" if rollout_release_after_collect else "resident",
        ),
        reward=ActorLeasePolicy(
            mode="on_demand" if reward_release_after_score else "resident",
        ),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=rollout_release_before_train,
            release_rollout_before_reward=rollout_release_before_reward_model,
            release_reward_after_score=reward_release_after_score,
        ),
    )
    return ResolvedDistributedResources(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_devices=rollout_devices,
        reward_devices=reward_devices,
        rollout_num_gpus=rollout_num_gpus,
        rollout_num_workers=rollout_num_workers,
        rollout_gpus_per_worker=rollout_gpus_per_worker,
        reward_num_workers=reward_num_workers,
        reward_gpus_per_worker=reward_gpus_per_worker,
        reward_shared_with_rollout=reward_shared_with_rollout,
        rollout_release_after_collect=rollout_release_after_collect,
        rollout_release_before_reward_model=rollout_release_before_reward_model,
        rollout_persistent_colocated_workers=config.rollout_persistent_colocated_workers,
        rollout_gpu_memory_fraction=gpu_memory_fraction,
        reward_release_after_score=reward_release_after_score,
        reward_placement_strategy=config.reward_placement_strategy,
        reward_cpus_per_worker=config.reward_cpus_per_worker,
        reward_max_inflight_batches=config.reward_max_inflight_batches,
        requires_trainer_reservation=requires_trainer_reservation,
        colocated=colocated,
        cross_node=config.cross_node,
        lifecycle=lifecycle,
    )


def _validate_trainer_device_count(
    strategy: str,
    trainer_devices: tuple[int, ...],
    world_size: int,
) -> None:
    """Trainer GPU-count rule, gated by ``distributed.training.strategy``.

    ``single_process`` is one process on one device (0 or 1 GPU).
    ``fsdp`` runs one torchrun process per GPU, so the trainer device set must
    cover the whole training world (``num_nodes * gpus_per_node``).
    """

    if strategy == "fsdp":
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
            "rollout and reward (each FSDP rank owns its GPU; rollout/reward run as "
            "separate Ray actors). "
            f"trainer={list(trainer_devices)} rollout={list(rollout_devices)} "
            f"reward={list(reward_devices)} "
            f"(overlap rollout={overlap_rollout}, reward={overlap_reward}).",
        )


def trainer_torch_device(
    resolved: ResolvedDistributedResources,
    *,
    actual_trainer_devices: tuple[int, ...] | list[int] | None = None,
) -> str:
    """Return the torch device string the single-process trainer should use."""

    devices = tuple(actual_trainer_devices or resolved.trainer_devices)
    if not devices:
        return "cpu"
    return f"cuda:{int(devices[0])}"


def format_distributed_resource_plan(
    resolved: ResolvedDistributedResources,
    *,
    actual_placement: Any | None = None,
) -> str:
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
        f"reward_shared_with_rollout={resolved.reward_shared_with_rollout}",
        f"rollout_release_after_collect={resolved.rollout_release_after_collect}",
        "rollout_persistent_colocated_workers="
        f"{resolved.rollout_persistent_colocated_workers}",
        f"rollout_gpu_memory_fraction={resolved.rollout_gpu_memory_fraction}",
        "rollout_release_before_reward_model="
        f"{resolved.rollout_release_before_reward_model}",
        f"reward_release_after_score={resolved.reward_release_after_score}",
        f"colocated={resolved.colocated}",
        f"cross_node={resolved.cross_node}",
        f"trainer_reservation={resolved.requires_trainer_reservation}",
        # Reading the plan at a glance: lease mode per role + which boundaries
        # release. resident=stays up, on_demand=released at the handoff.
        f"lifecycle=rollout:{resolved.lifecycle.rollout.mode}"
        f"/reward:{resolved.lifecycle.reward.mode}",
        "handoff="
        f"before_train:{resolved.lifecycle.handoff.release_rollout_before_train}"
        f",before_reward:{resolved.lifecycle.handoff.release_rollout_before_reward}"
        f",reward_after_score:{resolved.lifecycle.handoff.release_reward_after_score}",
    ]
    if actual_placement is not None:
        parts.append(f"actual={actual_placement}")
    return "Distributed resources: " + " ".join(parts)


def _distributed_resource_config_from_cfg(cfg: Any) -> DistributedResourceConfig:
    distributed = cfg_get(cfg, "distributed", {})
    resources = cfg_get(distributed, "resources", {})
    trainer_node = cfg_get(resources, "trainer", {})
    rollout_node = cfg_get(resources, "rollout", {})
    reward_node = cfg_get(resources, "reward", _MISSING)
    rollout_runtime = cfg_get(distributed, "rollout", {})
    reward_runtime = cfg_get(distributed, "reward", {})

    trainer = RoleResourceConfig(
        num_gpus=cfg_get(trainer_node, "num_gpus", "auto"),
        devices=_parse_devices(cfg_get(trainer_node, "devices", "auto")),
    )
    rollout = RolloutResourceConfig(
        num_gpus=cfg_get(rollout_node, "num_gpus", "auto"),
        devices=_parse_devices(cfg_get(rollout_node, "devices", "auto")),
        gpus_per_worker=float(cfg_get(rollout_node, "gpus_per_worker", 1.0)),
        num_workers=cfg_get(rollout_node, "num_workers", "auto"),
    )
    if reward_node is _MISSING:
        reward = RewardResourceConfig(num_gpus=0, devices=[])
    else:
        reward = RewardResourceConfig(
            num_gpus=cfg_get(reward_node, "num_gpus", "auto"),
            devices=_parse_devices(cfg_get(reward_node, "devices", "auto")),
            gpus_per_worker=float(cfg_get(reward_node, "gpus_per_worker", 1.0)),
            num_workers=cfg_get(reward_node, "num_workers", "auto"),
            share_with_rollout=_parse_optional_bool(
                cfg_get(reward_node, "share_with_rollout", None),
            ),
        )
    return DistributedResourceConfig(
        visible_devices=_parse_devices(cfg_get(resources, "visible_devices", "auto")),
        trainer=trainer,
        rollout=rollout,
        reward=reward,
        allow_overlap=bool(cfg_get(resources, "allow_overlap", False)),
        rollout_release_after_collect=_parse_optional_bool(
            cfg_get(rollout_runtime, "release_after_collect", None),
        ),
        rollout_release_before_reward_model=_parse_optional_bool(
            cfg_get(rollout_runtime, "release_before_reward_model", None),
        ),
        rollout_persistent_colocated_workers=bool(
            cfg_get(rollout_runtime, "persistent_colocated_workers", False),
        ),
        rollout_gpu_memory_fraction=_parse_optional_float(
            cfg_get(rollout_runtime, "gpu_memory_fraction", None),
        ),
        reward_release_after_score=_parse_optional_bool(
            cfg_get(reward_runtime, "release_after_score", None),
        ),
        reward_placement_strategy=str(
            cfg_get(reward_runtime, "placement_strategy", "SPREAD"),
        ),
        reward_cpus_per_worker=float(
            cfg_get(reward_runtime, "cpus_per_worker", 0.5),
        ),
        reward_max_inflight_batches=int(
            cfg_get(reward_runtime, "max_inflight_batches", 1),
        ),
        cross_node=bool(cfg_get(resources, "cross_node", False)),
    )


def _resolve_visible_devices(value: list[int] | str) -> tuple[int, ...]:
    devices = _parse_devices(value)
    if devices == "auto":
        return _auto_visible_cuda_devices()
    return tuple(_dedupe_ints(devices, field_name="distributed.resources.visible_devices"))


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
        return devices

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
        allow_overlap=allow_overlap,
        not_enough_pool_error=(
            "Not enough non-overlapping rollout GPUs: "
            f"requested={requested}, available={len(pool)}, "
            f"trainer={list(trainer_devices)}, visible={list(visible_devices)}. "
            "Expose more GPUs or set distributed.resources.allow_overlap=true with "
            "distributed.rollout.release_after_collect=true for single-GPU debug."
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
    rollout_release_after_collect: bool | None,
    rollout_release_before_reward_model: bool | None,
    reward_release_after_score: bool | None,
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
            rollout_release_after_collect=rollout_release_after_collect,
            rollout_release_before_reward_model=rollout_release_before_reward_model,
            reward_release_after_score=reward_release_after_score,
        )
        return devices

    share = reward_config.share_with_rollout
    if share is None:
        # Auto placement: prefer a dedicated spare GPU when the visible pool
        # can satisfy the request; otherwise fall back to sharing the rollout
        # pool. Removes the footgun where a spelled-out true kept forcing
        # shared single-GPU churn even on machines with spare GPUs.
        spare_excluded = set(trainer_devices) | set(rollout_devices)
        spare_pool = tuple(
            device for device in visible_devices if device not in spare_excluded
        )
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
                rollout_release_after_collect=rollout_release_after_collect,
                rollout_release_before_reward_model=rollout_release_before_reward_model,
                reward_release_after_score=reward_release_after_score,
            )
            return devices
        share = True

    if share:
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
            rollout_release_after_collect=rollout_release_after_collect,
            rollout_release_before_reward_model=rollout_release_before_reward_model,
            reward_release_after_score=reward_release_after_score,
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
        allow_overlap=allow_overlap,
        not_enough_pool_error=(
            "Not enough non-overlapping reward GPUs: "
            f"requested={requested}, available={len(pool)}, "
            f"trainer={list(trainer_devices)}, rollout={list(rollout_devices)}, "
            f"visible={list(visible_devices)}. Use "
            "distributed.resources.reward.share_with_rollout=true with "
            "release_after_collect/release_after_score for a shared inference pool, "
            "or expose a separate reward GPU."
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
        rollout_release_after_collect=rollout_release_after_collect,
        rollout_release_before_reward_model=rollout_release_before_reward_model,
        reward_release_after_score=reward_release_after_score,
    )
    return devices


def _validate_reward_overlap(
    *,
    devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_config: RewardResourceConfig,
    allow_overlap: bool,
    rollout_release_after_collect: bool | None,
    rollout_release_before_reward_model: bool | None,
    reward_release_after_score: bool | None,
) -> None:
    rollout_overlap = sorted(set(devices) & set(rollout_devices))
    if rollout_overlap:
        # None means auto placement chose sharing; only an explicit false
        # contradicts an overlapping topology.
        if reward_config.share_with_rollout is False:
            raise ValueError(
                "Reward and rollout devices overlap but "
                "distributed.resources.reward.share_with_rollout=false: "
                f"reward={list(devices)} rollout={list(rollout_devices)}",
            )
        # Unset flags derive to true for a shared pool; only an explicit
        # false contradicts the topology.
        if (
            rollout_release_after_collect is False
            or rollout_release_before_reward_model is False
            or reward_release_after_score is False
        ):
            raise ValueError(
                "Reward/rollout shared inference pool requires "
                "distributed.rollout.release_after_collect, "
                "distributed.rollout.release_before_reward_model, and "
                "distributed.reward.release_after_score to be true (or unset, "
                "in which case they derive to true)",
            )

    trainer_overlap = sorted(set(devices) & set(trainer_devices))
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


def reward_runtime_resource_kwargs(
    resolved: ResolvedDistributedResources,
) -> dict[str, Any]:
    """Return flat reward Ray runtime kwargs derived from the shared plan."""

    return {
        "num_workers": resolved.reward_num_workers,
        "gpus_per_worker": resolved.reward_gpus_per_worker,
        "cpus_per_worker": resolved.reward_cpus_per_worker,
        "max_inflight_batches": resolved.reward_max_inflight_batches,
        # Sourced from the lifecycle plan: an on_demand reward lease drops its
        # actors after each score so a shared placement can be reused.
        "release_after_score": resolved.lifecycle.reward.mode == "on_demand",
        "placement_strategy": resolved.reward_placement_strategy,
        "expected_gpu_ids": tuple(resolved.reward_devices),
    }


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
    trainer_bundle_indices: tuple[int, ...]
    rollout_bundle_indices: tuple[int, ...]
    reward_bundle_indices: tuple[int, ...]

    @property
    def total_bundles(self) -> int:
        return len(self.bundle_gpu_ids)

    def gpu_id_for_bundle(self, bundle_index: int) -> int | None:
        return self.bundle_gpu_ids[bundle_index]


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
    trainer = (
        tuple(_gpu_bundle(gpu_id) for gpu_id in resolved.trainer_devices)
        if resolved.requires_trainer_reservation
        else ()
    )

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
        trainer_bundle_indices=trainer,
        rollout_bundle_indices=rollout,
        reward_bundle_indices=reward,
    )


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _parse_optional_float(value: Any) -> float | None:
    value = _to_plain(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"distributed.rollout.gpu_memory_fraction must be a float or null, got {value!r}",
        ) from exc


def _derived_release_flag(explicit: bool | None, *, derived: bool) -> bool:
    if explicit is None:
        return derived
    return explicit


def _parse_devices(value: Any) -> list[int] | str:
    value = _to_plain(value)
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
    value = _to_plain(value)
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
    value = _to_plain(value)
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


def _count_ray_rewards(cfg: Any) -> int:
    reward = cfg_get(cfg, "reward", {})
    components = cfg_get(reward, "components", {})
    kwargs = cfg_get(reward, "kwargs", {})
    count = 0
    for reward_key in components or {}:
        try:
            reward_weight = float(cfg_get(components, str(reward_key), 0.0))
        except (TypeError, ValueError):
            reward_weight = 0.0
        if reward_weight <= 0:
            continue
        reward_kwargs = cfg_get(kwargs, str(reward_key), {})
        if str(cfg_get(reward_kwargs, "execution", "")) == "pool":
            count += 1
    return count


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


def _to_plain(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except Exception:
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


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
    "reward_runtime_resource_kwargs",
    "trainer_torch_device",
]
