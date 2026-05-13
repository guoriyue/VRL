"""Rollout backend selection and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vrl.engine.execution.runtime import RolloutBackend
from vrl.rollouts.runtime.config import RolloutBackendConfig

DRIVER_CUDA_OWNERSHIP_ERROR = (
    "Driver CUDA device overlaps rollout devices without an explicit colocate "
    "configuration."
)

_MISSING = object()


def validate_rollout_backend_config(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
) -> RolloutBackendConfig:
    """Validate rollout backend config before a collector touches the runtime."""
    config = RolloutBackendConfig.from_cfg(cfg)

    driver_cuda_devices = _driver_cuda_devices(
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )
    _validate_driver_cuda_ownership(config, driver_cuda_devices)

    return config


def build_rollout_backend_from_cfg(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
    runtime_spec: Any | None = None,
    gatherer: Any | None = None,
) -> RolloutBackend:
    """Build the Ray rollout backend selected by config.

    Ray launch requires a serializable ``runtime_spec`` plus pure chunk
    ``gatherer`` so the launcher can create rollout workers without receiving
    live model objects.
    """
    config = validate_rollout_backend_config(
        cfg,
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )

    runtime_spec = (
        runtime_spec
        if runtime_spec is not None
        else _cfg_path(
            cfg,
            "distributed.rollout.runtime_spec",
            None,
        )
    )
    if runtime_spec is not None and gatherer is not None:
        from vrl.distributed.ray.rollout.launcher import RayRolloutLauncher

        if config.release_after_collect:
            return ReleasableRayRolloutBackend(config, runtime_spec, gatherer)
        return RayRolloutLauncher().launch(config, runtime_spec, gatherer)

    raise ValueError(
        "Ray-only rollout backend requires runtime_spec plus gatherer so "
        "RayRolloutLauncher can construct workers through the "
        "runtime_builder+executor_cls path.",
    )


def _validate_driver_cuda_ownership(
    config: RolloutBackendConfig,
    driver_cuda_devices: set[int],
) -> None:
    if not driver_cuda_devices:
        return

    resources = config.resources
    if resources is None:
        if config.allow_driver_gpu_overlap:
            if config.release_after_collect:
                return
            raise ValueError(
                "Driver CUDA placement uses legacy overlap config but "
                "distributed.rollout.release_after_collect=false. "
                "Set release_after_collect=true for single-GPU Ray debug.",
            )
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
        policy = getattr(driver_bundle, "policy", None)

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


class ReleasableRayRolloutBackend(RolloutBackend):
    """Ray backend wrapper that can drop rollout actors between train phases.

    Single-GPU Ray debugging colocates the trainer and one rollout worker on the
    same CUDA device. Keeping the rollout worker alive after collection leaves
    the full generation pipeline resident while the trainer replays the batch,
    which is too much for large diffusion models. This wrapper keeps the public
    runtime object stable for the trainer/weight-syncer, but tears down and
    recreates the underlying Ray actors when ``release_memory()`` is called.
    """

    def __init__(
        self,
        config: RolloutBackendConfig,
        runtime_spec: Any,
        gatherer: Any,
    ) -> None:
        self.config = config
        self.runtime_spec = runtime_spec
        self.gatherer = gatherer
        self.weight_sync = object() if config.sync_trainable_state != "disabled" else None
        self.requires_driver_model_offload = config.gpus_per_worker > 0
        self.current_policy_version = _runtime_spec_policy_version(runtime_spec)
        self._runtime: Any | None = None
        self._last_state: Any | None = None

    async def generate(self, request: Any) -> Any:
        runtime = await self._ensure_runtime()
        return await runtime.generate(request)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        if self.weight_sync is None:
            raise RuntimeError("ReleasableRayRolloutBackend has no rollout weight sync")
        self._last_state = state_ref
        self.current_policy_version = int(policy_version)
        if self._runtime is not None:
            await self._runtime.update_weights(state_ref, self.current_policy_version)

    async def release_memory(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        self._runtime = None
        await runtime.shutdown()

    async def shutdown(self) -> None:
        await self.release_memory()

    async def _ensure_runtime(self) -> Any:
        if self._runtime is None:
            from vrl.distributed.ray.rollout.launcher import RayRolloutLauncher

            runtime = RayRolloutLauncher().launch(
                self.config,
                self.runtime_spec,
                self.gatherer,
            )
            self._runtime = runtime
            if self._last_state is not None:
                await runtime.update_weights(
                    self._last_state,
                    int(self.current_policy_version),
                )
        return self._runtime


def _runtime_spec_policy_version(runtime_spec: Any) -> int | None:
    try:
        from vrl.engine.core.runtime_spec import GenerationRuntimeSpec

        spec = GenerationRuntimeSpec.from_value(runtime_spec)
    except Exception:
        return None
    if spec.policy_version is None:
        return None
    return int(spec.policy_version)


def _cfg_path(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "ReleasableRayRolloutBackend",
    "build_rollout_backend_from_cfg",
    "validate_rollout_backend_config",
]
