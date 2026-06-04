"""Runtime metadata helpers for replay-only model loading.

The trainer and Ray rollout worker intentionally load different runtime
surfaces. Rollout workers own full generation state for sampling and decoding;
trainers should eventually own only the modules needed to replay recorded
trajectory actions under the current trainable weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from vrl.models.dtypes import resolve_torch_dtype

RuntimeRole = Literal["full_generation_model", "minimal_replay_model"]

FULL_GENERATION_RUNTIME_ROLE: RuntimeRole = "full_generation_model"
MINIMAL_REPLAY_RUNTIME_ROLE: RuntimeRole = "minimal_replay_model"

RUNTIME_ROLE_KEY = "runtime_role"
LOADS_FULL_GENERATION_MODULES_KEY = "loads_full_generation_modules"
REQUIRES_MINIMAL_REPLAY_LOADER_KEY = "requires_minimal_replay_loader"
REPLAY_MODULES_KEY = "replay_modules"
GENERATION_ONLY_MODULES_KEY = "generation_only_modules"


@dataclass(frozen=True, slots=True)
class ReplayModuleLoadingProfile:
    """Declared module-loading shape for one runtime bundle."""

    runtime_role: RuntimeRole
    loads_full_generation_modules: bool
    requires_minimal_replay_loader: bool
    replay_modules: tuple[str, ...] = ()
    generation_only_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_runtime_role(self.runtime_role)
        object.__setattr__(
            self,
            "replay_modules",
            _normalize_names(self.replay_modules, REPLAY_MODULES_KEY),
        )
        object.__setattr__(
            self,
            "generation_only_modules",
            _normalize_names(self.generation_only_modules, GENERATION_ONLY_MODULES_KEY),
        )
        if self.runtime_role == FULL_GENERATION_RUNTIME_ROLE:
            if not self.loads_full_generation_modules:
                raise ValueError(
                    "full_generation_model must declare "
                    "loads_full_generation_modules=true",
                )
            return
        if self.loads_full_generation_modules:
            raise ValueError(
                "minimal_replay_model cannot declare "
                "loads_full_generation_modules=true",
            )
        if self.requires_minimal_replay_loader:
            raise ValueError(
                "minimal_replay_model cannot require another minimal replay loader",
            )

    def as_metadata(self) -> dict[str, Any]:
        """Serialize the profile into ``RuntimeBundle.metadata`` fields."""

        metadata: dict[str, Any] = {
            RUNTIME_ROLE_KEY: self.runtime_role,
            LOADS_FULL_GENERATION_MODULES_KEY: self.loads_full_generation_modules,
            REQUIRES_MINIMAL_REPLAY_LOADER_KEY: self.requires_minimal_replay_loader,
        }
        if self.replay_modules:
            metadata[REPLAY_MODULES_KEY] = list(self.replay_modules)
        if self.generation_only_modules:
            metadata[GENERATION_ONLY_MODULES_KEY] = list(self.generation_only_modules)
        return metadata


def full_generation_bundle_metadata(
    *,
    requires_minimal_replay_loader: bool = True,
    replay_modules: tuple[str, ...] = (),
    generation_only_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return metadata for a runtime bundle that owns generation-only modules."""

    return ReplayModuleLoadingProfile(
        runtime_role=FULL_GENERATION_RUNTIME_ROLE,
        loads_full_generation_modules=True,
        requires_minimal_replay_loader=bool(requires_minimal_replay_loader),
        replay_modules=replay_modules,
        generation_only_modules=generation_only_modules,
    ).as_metadata()


def minimal_replay_bundle_metadata(
    *,
    replay_modules: tuple[str, ...] = (),
    generation_only_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return metadata for a trainer bundle that only owns replay modules."""

    return ReplayModuleLoadingProfile(
        runtime_role=MINIMAL_REPLAY_RUNTIME_ROLE,
        loads_full_generation_modules=False,
        requires_minimal_replay_loader=False,
        replay_modules=replay_modules,
        generation_only_modules=generation_only_modules,
    ).as_metadata()


def module_loading_profile_from_metadata(
    metadata: Mapping[str, Any],
) -> ReplayModuleLoadingProfile:
    """Parse and validate replay module-loading metadata."""

    runtime_role = _runtime_role_from_metadata(metadata)
    return ReplayModuleLoadingProfile(
        runtime_role=runtime_role,
        loads_full_generation_modules=_required_bool(
            metadata,
            LOADS_FULL_GENERATION_MODULES_KEY,
        ),
        requires_minimal_replay_loader=_required_bool(
            metadata,
            REQUIRES_MINIMAL_REPLAY_LOADER_KEY,
        ),
        replay_modules=_tuple_metadata(metadata, REPLAY_MODULES_KEY),
        generation_only_modules=_tuple_metadata(metadata, GENERATION_ONLY_MODULES_KEY),
    )


def bundle_loads_full_generation_modules(bundle: Any) -> bool:
    """Return whether a runtime bundle declares full generation module ownership."""

    metadata = getattr(bundle, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get(LOADS_FULL_GENERATION_MODULES_KEY, False))


def require_minimal_replay_bundle(bundle: Any, *, owner: str = "RuntimeBundle") -> None:
    """Fail if ``bundle`` is not declared as a minimal replay runtime."""

    metadata = getattr(bundle, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{owner}.metadata must be a mapping")
    profile = module_loading_profile_from_metadata(metadata)
    if profile.runtime_role != MINIMAL_REPLAY_RUNTIME_ROLE:
        raise ValueError(
            f"{owner} must declare runtime_role='minimal_replay_model'; "
            f"got {profile.runtime_role!r}",
        )


def load_diffusers_transformer_component(
    spec: Any,
    class_name: str,
    *,
    subfolder: str = "transformer",
) -> Any:
    """Load only a diffusers transformer component from a model repository."""

    import diffusers

    load_kwargs: dict[str, Any] = {}
    revision = (getattr(spec, "model_config", None) or {}).get("revision")
    if revision:
        load_kwargs["revision"] = revision
    transformer_cls = getattr(diffusers, class_name)
    return transformer_cls.from_pretrained(
        spec.model_name_or_path,
        subfolder=subfolder,
        torch_dtype=resolve_torch_dtype(spec.dtype),
        **load_kwargs,
    )


def load_diffusers_scheduler_component(
    spec: Any,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    import diffusers

    load_kwargs: dict[str, Any] = {}
    revision = (getattr(spec, "model_config", None) or {}).get("revision")
    if revision:
        load_kwargs["revision"] = revision
    scheduler_cls = getattr(diffusers, class_name)
    scheduler = scheduler_cls.from_pretrained(
        spec.model_name_or_path,
        subfolder=subfolder,
        **load_kwargs,
    )
    num_steps = spec.num_steps
    if num_steps is not None:
        scheduler.set_timesteps(int(num_steps), device=getattr(spec, "device", None))
    return scheduler


def load_flow_match_scheduler_component(
    spec: Any,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load the lightweight FlowMatch scheduler needed for replay log-prob math."""

    return load_diffusers_scheduler_component(
        spec,
        "FlowMatchEulerDiscreteScheduler",
        subfolder=subfolder,
    )


def apply_lora_to_transformer(model: Any, spec: Any) -> None:
    """Attach or load a PEFT LoRA adapter on ``model.transformer``."""

    from peft import LoraConfig, PeftModel, get_peft_model

    transformer = model.transformer
    transformer.requires_grad_(False)
    to = getattr(transformer, "to", None)
    if callable(to):
        to(model.device, dtype=resolve_torch_dtype(spec.dtype))

    lora_path = spec.lora_path
    if lora_path:
        wrapped = PeftModel.from_pretrained(
            transformer,
            lora_path,
            is_trainable=True,
        )
        wrapped.set_adapter("default")
        model._set_transformer(wrapped)
        return

    lora_config = spec.lora
    if lora_config is None:
        raise ValueError("LoRA runtime spec requires lora_config when lora_path is empty")
    cfg = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        init_lora_weights=lora_config.get("init_lora_weights", "gaussian"),
        target_modules=lora_config["target_modules"],
    )
    model._set_transformer(get_peft_model(transformer, cfg))


def enable_transformer_full_finetune(model: Any) -> None:
    """Mark the replay transformer fully trainable."""

    model.transformer.requires_grad_(True)
    to = getattr(model.transformer, "to", None)
    if callable(to):
        to(model.device)


def compile_transformer(model: Any, mode: str) -> None:
    """Apply ``torch.compile`` to the replay transformer."""

    import torch

    model._set_transformer(
        torch.compile(model.transformer, mode=mode, fullgraph=False),
    )


def _runtime_role_from_metadata(metadata: Mapping[str, Any]) -> RuntimeRole:
    value = metadata.get(RUNTIME_ROLE_KEY)
    if not isinstance(value, str):
        raise ValueError(f"RuntimeBundle.metadata[{RUNTIME_ROLE_KEY!r}] is required")
    return _validate_runtime_role(value)


def _validate_runtime_role(value: str) -> RuntimeRole:
    if value == FULL_GENERATION_RUNTIME_ROLE:
        return FULL_GENERATION_RUNTIME_ROLE
    if value == MINIMAL_REPLAY_RUNTIME_ROLE:
        return MINIMAL_REPLAY_RUNTIME_ROLE
    raise ValueError(
        f"unknown runtime_role={value!r}; expected "
        f"{FULL_GENERATION_RUNTIME_ROLE!r} or {MINIMAL_REPLAY_RUNTIME_ROLE!r}",
    )


def _required_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"RuntimeBundle.metadata[{key!r}] must be a bool")
    return value


def _tuple_metadata(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key, ())
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"RuntimeBundle.metadata[{key!r}] must be a sequence of strings")
    return _normalize_names(tuple(value), key)


def _normalize_names(values: tuple[str, ...], key: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must contain non-empty strings")
        normalized.append(value)
    return tuple(normalized)


__all__ = [
    "FULL_GENERATION_RUNTIME_ROLE",
    "GENERATION_ONLY_MODULES_KEY",
    "LOADS_FULL_GENERATION_MODULES_KEY",
    "MINIMAL_REPLAY_RUNTIME_ROLE",
    "REPLAY_MODULES_KEY",
    "REQUIRES_MINIMAL_REPLAY_LOADER_KEY",
    "RUNTIME_ROLE_KEY",
    "ReplayModuleLoadingProfile",
    "RuntimeRole",
    "apply_lora_to_transformer",
    "bundle_loads_full_generation_modules",
    "compile_transformer",
    "enable_transformer_full_finetune",
    "full_generation_bundle_metadata",
    "load_diffusers_scheduler_component",
    "load_diffusers_transformer_component",
    "load_flow_match_scheduler_component",
    "minimal_replay_bundle_metadata",
    "module_loading_profile_from_metadata",
    "require_minimal_replay_bundle",
    "resolve_torch_dtype",
]
