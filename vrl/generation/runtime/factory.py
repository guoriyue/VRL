"""Generation runtime selection and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vrl.generation.protocols import GenerationRuntime
from vrl.generation.runtime.config import GenerationRuntimeConfig

DRIVER_CUDA_OWNERSHIP_ERROR = (
    "Driver CUDA device overlaps rollout devices without an explicit colocate "
    "configuration."
)

_MISSING = object()


def validate_generation_runtime_config(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
) -> GenerationRuntimeConfig:
    """Validate generation runtime config before a collector touches the runtime."""
    config = GenerationRuntimeConfig.from_cfg(cfg)

    driver_cuda_devices = _driver_cuda_devices(
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )
    _validate_driver_cuda_ownership(config, driver_cuda_devices)
    if driver_bundle is not None:
        from vrl.utils.memory import validate_colocated_replay_memory

        validate_colocated_replay_memory(
            bundle=driver_bundle,
            rollout_config=config,
        )

    return config


def build_generation_runtime_from_cfg(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
    launch_contract: Any | None = None,
    gatherer: Any | None = None,
) -> GenerationRuntime:
    """Build the Ray generation runtime selected by config.

    Ray launch requires a serializable ``launch_contract`` plus pure chunk
    ``gatherer`` so the launcher can create generation workers without receiving
    live model objects.
    """
    config = validate_generation_runtime_config(
        cfg,
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )

    launch_contract = (
        launch_contract
        if launch_contract is not None
        else _cfg_path(
            cfg,
            "distributed.rollout.launch_contract",
            None,
        )
    )
    if launch_contract is not None and gatherer is not None:
        from vrl.generation.ray.launcher import RayGenerationLauncher

        if config.release_after_collect:
            return ReleasableRayGenerationRuntime(config, launch_contract, gatherer)
        return RayGenerationLauncher().launch(config, launch_contract, gatherer)

    raise ValueError(
        "Ray-only generation runtime requires launch_contract plus gatherer so "
        "RayGenerationLauncher can construct workers through the "
        "runtime_builder+executor_cls path.",
    )


def _validate_driver_cuda_ownership(
    config: GenerationRuntimeConfig,
    driver_cuda_devices: set[int],
) -> None:
    if not driver_cuda_devices:
        return

    resources = config.resources
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


class ReleasableRayGenerationRuntime(GenerationRuntime):
    """Ray runtime wrapper that can drop generation actors between train phases.

    Single-GPU Ray debugging colocates the trainer and one generation worker on the
    same CUDA device. Keeping the generation worker alive after collection leaves
    the full generation pipeline resident while the trainer replays the batch,
    which is too much for large diffusion models. This wrapper keeps the public
    runtime object stable for the trainer/weight-syncer, but tears down and
    recreates the underlying Ray actors when ``release_memory()`` is called.
    """

    def __init__(
        self,
        config: GenerationRuntimeConfig,
        launch_contract: Any,
        gatherer: Any,
    ) -> None:
        self.config = config
        self.launch_contract = launch_contract
        self.gatherer = gatherer
        self.weight_sync = object() if config.sync_trainable_state != "disabled" else None
        self.requires_driver_model_offload = config.gpus_per_worker > 0
        self.current_policy_version = _launch_contract_policy_version(launch_contract)
        self._runtime: Any | None = None
        self._last_state: Any | None = None

    async def generate(self, request: Any) -> Any:
        runtime = await self._ensure_runtime()
        return await runtime.generate(request)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        if self.weight_sync is None:
            raise RuntimeError("ReleasableRayGenerationRuntime has no generation weight sync")
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
            from vrl.generation.ray.launcher import RayGenerationLauncher

            runtime = RayGenerationLauncher().launch(
                self.config,
                self.launch_contract,
                self.gatherer,
            )
            self._runtime = runtime
            if self._last_state is not None:
                await runtime.update_weights(
                    self._last_state,
                    int(self.current_policy_version),
                )
        return self._runtime


def _launch_contract_policy_version(launch_contract: Any) -> int | None:
    try:
        from vrl.generation.launch_contract import GenerationRuntimeLaunchContract

        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
    except Exception:
        return None
    if contract.policy_version is None:
        return None
    return int(contract.policy_version)


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


ReleasableRayRolloutBackend = ReleasableRayGenerationRuntime
build_rollout_backend_from_cfg = build_generation_runtime_from_cfg
validate_rollout_backend_config = validate_generation_runtime_config


__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "ReleasableRayGenerationRuntime",
    "ReleasableRayRolloutBackend",
    "build_generation_runtime_from_cfg",
    "build_rollout_backend_from_cfg",
    "validate_generation_runtime_config",
    "validate_rollout_backend_config",
]
