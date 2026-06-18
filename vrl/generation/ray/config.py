"""Ray generation config and driver-side validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from vrl.ray.resources import (
    ResolvedDistributedResources,
    resolve_distributed_resources,
)
from vrl.utils.config import cfg_get

DRIVER_CUDA_OWNERSHIP_ERROR = (
    "Driver CUDA device overlaps rollout devices without an explicit colocate "
    "configuration."
)


@dataclass(slots=True)
class RayGenerationConfig:
    """Ray generation execution config plus resolved worker resources."""

    num_workers: int = 1
    gpus_per_worker: float = 1.0
    cpus_per_worker: float = 1.0
    allow_driver_gpu_overlap: bool = False
    max_inflight_chunks_per_worker: int = 1
    # Chunk->worker binding: "round_robin" binds at plan time (baseline);
    # "dynamic" binds at dispatch time (pull + LPT). Equivalent for 1 worker.
    chunk_placement_strategy: str = "round_robin"
    # Binary on/off (consumers only check != "disabled"; the "lora_only" name is
    # legacy — the syncer flattens whatever trainable modules exist, lora or full).
    # Defaults ON: online runs train the policy the rollout workers must resync, so
    # an omitted value previously meant silent stale-policy training. The weight
    # syncer is only built on the online launch path, so this never affects eval.
    sync_trainable_state: str = "lora_only"
    release_after_collect: bool = False
    persistent_colocated_workers: bool = False
    # Hard cap on this worker's CUDA allocator share, in (0, 1]; None = no cap.
    # Applied in the worker process via torch.cuda.set_per_process_memory_fraction.
    gpu_memory_fraction: float | None = None
    resources: ResolvedDistributedResources | None = None

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.gpus_per_worker < 0:
            raise ValueError("gpus_per_worker must be >= 0")
        if self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        if self.max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        if self.sync_trainable_state not in {"disabled", "lora_only"}:
            raise ValueError("sync_trainable_state must be 'disabled' or 'lora_only'")
        if self.chunk_placement_strategy not in {"round_robin", "dynamic"}:
            raise ValueError(
                "chunk_placement_strategy must be 'round_robin' or 'dynamic'",
            )
        if self.persistent_colocated_workers and not self.allow_driver_gpu_overlap:
            raise ValueError(
                "persistent_colocated_workers=true requires trainer/rollout GPU overlap",
            )
        if self.persistent_colocated_workers and self.release_after_collect:
            raise ValueError(
                "persistent_colocated_workers=true requires release_after_collect=false",
            )
        if self.gpu_memory_fraction is not None and not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be in (0, 1] when set")

    @classmethod
    def from_cfg(cls, cfg: Any) -> RayGenerationConfig:
        """Build Ray generation config from a full training cfg."""
        if isinstance(cfg, cls):
            return cfg

        distributed = cfg_get(cfg, "distributed", _MISSING)
        if distributed is _MISSING:
            raise ValueError("distributed.resources is required")

        resources_node = cfg_get(distributed, "resources", _MISSING)
        if resources_node is _MISSING:
            raise ValueError("distributed.resources is required")
        resources = resolve_distributed_resources(cfg)
        rollout = cfg_get(distributed, "rollout", {})

        return cls(
            num_workers=resources.rollout_num_workers,
            gpus_per_worker=resources.rollout_gpus_per_worker,
            cpus_per_worker=float(
                cfg_get(rollout, "cpus_per_worker", 1.0),
            ),
            allow_driver_gpu_overlap=bool(resources.colocated),
            max_inflight_chunks_per_worker=int(
                cfg_get(
                    rollout,
                    "max_inflight_chunks_per_worker",
                    1,
                ),
            ),
            sync_trainable_state=str(
                cfg_get(rollout, "sync_trainable_state", "lora_only"),
            ),
            chunk_placement_strategy=str(
                cfg_get(rollout, "chunk_placement_strategy", "round_robin"),
            ),
            # Resolved values: unset YAML flags derive from the GPU topology
            # (resolve_distributed_resources is the single source of truth).
            release_after_collect=resources.rollout_release_after_collect,
            persistent_colocated_workers=resources.rollout_persistent_colocated_workers,
            gpu_memory_fraction=resources.rollout_gpu_memory_fraction,
            resources=resources,
        )

    def validate_driver_state(
        self,
        *,
        driver_bundle: Any | None = None,
        driver_policy: Any | None = None,
        trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
    ) -> RayGenerationConfig:
        """Validate driver CUDA ownership before Ray rollout actors are launched."""

        driver_cuda_devices = _driver_cuda_devices(
            driver_bundle=driver_bundle,
            driver_policy=driver_policy,
            trainable_modules=trainable_modules,
        )
        _validate_driver_cuda_ownership(self, driver_cuda_devices)
        if driver_bundle is not None:
            from vrl.utils.memory import validate_colocated_replay_memory

            validate_colocated_replay_memory(
                bundle=driver_bundle,
                rollout_config=self,
            )
        return self


def _validate_driver_cuda_ownership(
    config: RayGenerationConfig,
    driver_cuda_devices: set[int],
) -> None:
    if not driver_cuda_devices:
        return

    resources = config.resources
    if resources is not None and resources.cross_node:
        # Cross-node: the driver's head-local cuda ordinal and a remote rollout
        # GPU live in different ordinal spaces, so a set-intersection overlap
        # check is meaningless. Node-level isolation is enforced by the launcher
        # preflight (head --num-gpus=0) and validate_actor_gpu_ids node check.
        return
    if resources is None:
        raise ValueError(
            "Driver loaded rollout policy on CUDA, but no distributed.resources "
            "plan is available to prove rollout devices do not overlap. "
            "Provide distributed.resources for split runs, or for resident "
            "single-GPU debug set distributed.rollout.colocate_with_trainer: "
            "{memory_fraction: <0..1>}.",
        )

    overlap = driver_cuda_devices & set(resources.rollout_devices)
    if not overlap:
        return

    overlap_list = sorted(overlap)
    rollout_devices = list(resources.rollout_devices)
    if not config.allow_driver_gpu_overlap:
        raise ValueError(
            f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
            f"{rollout_devices}, but resources.allow_overlap=false. "
            "Use CUDA_VISIBLE_DEVICES=0,1,2,3 with auto split for throughput, or "
            "for resident single-GPU debug set "
            "distributed.rollout.colocate_with_trainer: {memory_fraction: <0..1>}.",
        )

    if not config.release_after_collect and not config.persistent_colocated_workers:
        raise ValueError(
            f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
            f"{rollout_devices}, but the rollout worker is neither released after "
            "collect nor a resident colocated worker. Release is derived "
            "automatically from a shared-GPU topology; for an intentionally "
            "resident colocated debug worker set "
            "distributed.rollout.colocate_with_trainer: {memory_fraction: <0..1>}.",
        )


def _driver_cuda_devices(
    *,
    driver_bundle: Any | None,
    driver_policy: Any | None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None,
) -> set[int]:
    policy = driver_policy
    if policy is None and driver_bundle is not None:
        policy = getattr(driver_bundle, "model", None)

    has_policy_device, device = _get_device(policy)
    if has_policy_device:
        parsed = _cuda_device_index(device)
        return set() if parsed is None else {parsed}

    modules = trainable_modules
    if modules is None and driver_bundle is not None:
        modules = getattr(driver_bundle, "trainable_modules", None)

    devices: set[int] = set()
    for parameter_device in _iter_parameter_devices(modules):
        parsed = _cuda_device_index(parameter_device)
        if parsed is not None:
            devices.add(parsed)
    return devices


def _get_device(obj: Any) -> tuple[bool, Any]:
    if obj is None:
        return False, None
    try:
        device = obj.device
    except Exception:
        return False, None
    if device is None:
        return False, None
    return True, device


def _iter_parameter_devices(obj: Any, seen: set[int] | None = None) -> Iterable[Any]:
    if obj is None or isinstance(obj, (str, bytes)):
        return
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_parameter_devices(value, seen)
        return

    has_device, device = _get_device(obj)
    if has_device:
        yield device
        return

    parameters = getattr(obj, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            device = getattr(parameter, "device", None)
            if device is not None:
                yield device
        return

    if isinstance(obj, Iterable):
        for value in obj:
            yield from _iter_parameter_devices(value, seen)


def _cuda_device_index(device: Any) -> int | None:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        if str(device_type).lower() != "cuda":
            return None
        index = getattr(device, "index", None)
        return 0 if index is None else int(index)

    text = str(device).lower()
    if not text.startswith("cuda"):
        return None
    if ":" not in text:
        return 0
    _, raw_index = text.split(":", 1)
    try:
        return int(raw_index)
    except ValueError:
        return 0


_MISSING = object()


__all__ = ["DRIVER_CUDA_OWNERSHIP_ERROR", "RayGenerationConfig"]
