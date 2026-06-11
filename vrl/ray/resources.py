"""Role-level resource resolution for distributed VRL runs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

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
    reward_release_after_score: bool | None = None
    reward_placement_strategy: str = "SPREAD"
    reward_cpus_per_worker: float = 0.5
    reward_max_inflight_batches: int = 1
    cross_node: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedDistributedResources:
    """Concrete resource plan consumed by trainer and Ray role launchers."""

    visible_devices: tuple[int, ...]
    trainer_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]
    reward_devices: tuple[int, ...]
    rollout_num_gpus: int
    rollout_num_workers: int
    rollout_gpus_per_worker: float
    reward_num_gpus: int
    reward_num_workers: int
    reward_gpus_per_worker: float
    reward_shared_with_rollout: bool
    rollout_release_after_collect: bool
    rollout_release_before_reward_model: bool
    reward_release_after_score: bool
    reward_placement_strategy: str
    reward_cpus_per_worker: float
    reward_max_inflight_batches: int
    reward_gpu_reservation_count: int
    total_gpu_slots: int
    ray_total_bundles: int
    requires_trainer_reservation: bool
    colocated: bool
    cross_node: bool


_MISSING = object()


def resolve_distributed_resources(cfg: Any) -> ResolvedDistributedResources:
    """Resolve role-level resource config into concrete CUDA ordinals.

    This is the single source of truth for trainer/rollout/reward GPU
    ownership. It intentionally does static ownership checks only; memory
    pressure is still a runtime concern.
    """

    config = _distributed_resource_config_from_cfg(cfg)
    if config.cross_node:
        visible_devices = _resolve_cross_node_visible_devices(config)
    else:
        visible_devices = _resolve_visible_devices(config.visible_devices)

    trainer_devices = _resolve_role_devices(
        role="trainer",
        visible_devices=visible_devices,
        role_config=config.trainer,
        default_auto_count=1 if visible_devices else 0,
    )
    if len(trainer_devices) > 1:
        raise ValueError(
            "distributed.resources.trainer.devices currently supports only "
            f"0 or 1 GPU for the single-process trainer, got {trainer_devices}",
        )

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
    if _uses_ray_video_reward(cfg) and reward_gpus_per_worker > 0 and reward_num_gpus == 0:
        raise ValueError(
            "reward.kwargs.kling_video_reward.inference_runtime=ray requires "
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

    # Unset release flags follow the resolved topology: roles that share a GPU
    # must hand it over between phases; roles with dedicated GPUs stay resident.
    rollout_release_after_collect = _derived_release_flag(
        config.rollout_release_after_collect,
        derived=colocated or reward_shared_with_rollout,
    )
    rollout_release_before_reward_model = _derived_release_flag(
        config.rollout_release_before_reward_model,
        derived=reward_shared_with_rollout,
    )
    reward_release_after_score = _derived_release_flag(
        config.reward_release_after_score,
        derived=reward_shared_with_rollout,
    )

    requires_trainer_reservation = (
        bool(trainer_devices)
        and rollout_gpus_per_worker > 0
        and not colocated
        and rollout_num_workers > 0
        and not config.cross_node
    )
    reward_gpu_reservation_count = _reward_gpu_reservation_count(
        visible_devices=visible_devices,
        reward_devices=reward_devices,
        reward_gpus_per_worker=reward_gpus_per_worker,
    )
    total_gpu_slots = len(set(trainer_devices) | set(rollout_devices) | set(reward_devices))
    ray_total_bundles = rollout_num_workers + reward_num_workers + (
        len(trainer_devices) if requires_trainer_reservation else 0
    )

    return ResolvedDistributedResources(
        visible_devices=visible_devices,
        trainer_devices=trainer_devices,
        rollout_devices=rollout_devices,
        reward_devices=reward_devices,
        rollout_num_gpus=rollout_num_gpus,
        rollout_num_workers=rollout_num_workers,
        rollout_gpus_per_worker=rollout_gpus_per_worker,
        reward_num_gpus=reward_num_gpus,
        reward_num_workers=reward_num_workers,
        reward_gpus_per_worker=reward_gpus_per_worker,
        reward_shared_with_rollout=reward_shared_with_rollout,
        rollout_release_after_collect=rollout_release_after_collect,
        rollout_release_before_reward_model=rollout_release_before_reward_model,
        reward_release_after_score=reward_release_after_score,
        reward_placement_strategy=config.reward_placement_strategy,
        reward_cpus_per_worker=config.reward_cpus_per_worker,
        reward_max_inflight_batches=config.reward_max_inflight_batches,
        reward_gpu_reservation_count=reward_gpu_reservation_count,
        total_gpu_slots=total_gpu_slots,
        ray_total_bundles=ray_total_bundles,
        requires_trainer_reservation=requires_trainer_reservation,
        colocated=colocated,
        cross_node=config.cross_node,
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
        "rollout_release_before_reward_model="
        f"{resolved.rollout_release_before_reward_model}",
        f"reward_release_after_score={resolved.reward_release_after_score}",
        f"colocated={resolved.colocated}",
        f"cross_node={resolved.cross_node}",
        f"trainer_reservation={resolved.requires_trainer_reservation}",
        f"ray_bundles={resolved.ray_total_bundles}",
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
        "release_after_score": resolved.reward_release_after_score,
        "placement_strategy": resolved.reward_placement_strategy,
        "expected_gpu_ids": tuple(resolved.reward_devices),
        "gpu_reservation_count": resolved.reward_gpu_reservation_count,
    }


def _reward_gpu_reservation_count(
    *,
    visible_devices: tuple[int, ...],
    reward_devices: tuple[int, ...],
    reward_gpus_per_worker: float,
) -> int:
    if reward_gpus_per_worker <= 0 or not reward_devices:
        return 0
    positions = [visible_devices.index(device) for device in reward_devices if device in visible_devices]
    return min(positions) if positions else 0


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


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


def _uses_ray_video_reward(cfg: Any) -> bool:
    reward = cfg_get(cfg, "reward", {})
    components = cfg_get(reward, "components", {})
    kwargs = cfg_get(reward, "kwargs", {})
    for reward_key in ("kling_video_reward", "video_reward"):
        try:
            video_weight = float(cfg_get(components, reward_key, 0.0))
        except (TypeError, ValueError):
            video_weight = 0.0
        if video_weight <= 0:
            continue
        video_kwargs = cfg_get(kwargs, reward_key, {})
        return str(cfg_get(video_kwargs, "inference_runtime", "")) == "ray"
    return False


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
    "DistributedResourceConfig",
    "ResolvedDistributedResources",
    "RewardResourceConfig",
    "RoleResourceConfig",
    "RolloutResourceConfig",
    "format_distributed_resource_plan",
    "resolve_distributed_resources",
    "reward_runtime_resource_kwargs",
    "trainer_torch_device",
]
