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
    placement_strategy: str = "SPREAD"
    allow_driver_gpu_overlap: bool = False
    max_inflight_chunks_per_worker: int = 1
    sync_trainable_state: str = "disabled"
    release_after_collect: bool = False
    release_before_reward_model: bool = False
    resources: ResolvedDistributedResources | None = None

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.gpus_per_worker < 0:
            raise ValueError("gpus_per_worker must be >= 0")
        if self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        if not self.placement_strategy:
            raise ValueError("placement_strategy must be non-empty")
        if self.max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        if self.sync_trainable_state not in {"disabled", "lora_only"}:
            raise ValueError("sync_trainable_state must be 'disabled' or 'lora_only'")

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
            placement_strategy=str(
                cfg_get(rollout, "placement_strategy", "SPREAD"),
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
                cfg_get(rollout, "sync_trainable_state", "disabled"),
            ),
            # Resolved values: unset YAML flags derive from the GPU topology
            # (resolve_distributed_resources is the single source of truth).
            release_after_collect=resources.rollout_release_after_collect,
            release_before_reward_model=resources.rollout_release_before_reward_model,
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

    def to_dict(self) -> dict[str, Any]:
        """Return the flat config shape accepted by Ray generation runtimes."""
        return {
            "num_workers": self.num_workers,
            "gpus_per_worker": self.gpus_per_worker,
            "cpus_per_worker": self.cpus_per_worker,
            "placement_strategy": self.placement_strategy,
            "allow_driver_gpu_overlap": self.allow_driver_gpu_overlap,
            "max_inflight_chunks_per_worker": self.max_inflight_chunks_per_worker,
            "sync_trainable_state": self.sync_trainable_state,
            "release_after_collect": self.release_after_collect,
            "release_before_reward_model": self.release_before_reward_model,
        }


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
            "Use distributed.resources for split runs or enable overlap with "
            "distributed.rollout.release_after_collect=true for single-GPU debug.",
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
            "Use CUDA_VISIBLE_DEVICES=0,1,2,3 with auto split for throughput, "
            "or set allow_overlap=true with rollout.release_after_collect=true "
            "for single-GPU debug.",
        )

    if not config.release_after_collect:
        raise ValueError(
            f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
            f"{rollout_devices}, but distributed.rollout.release_after_collect=false. "
            "Set release_after_collect=true for single-GPU Ray debug.",
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
