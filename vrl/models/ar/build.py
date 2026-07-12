"""Shared descriptor-driven AR runtime construction and bundle assembly.

Every AR family (janus_pro, nextstep_1, ...) assembled its rollout and replay
``RuntimeBundle`` with the same model/config/import sequence. The registry now
owns those construction inputs and this module owns the sequence once.

Like the diffusion counterpart, the registry records model/config import paths
and points every AR family at this module. Family runtime modules keep only
their request executor and config projection; repeated rollout/replay builder
and model-build resolver facades are gone.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.capabilities import FamilyCapability
from vrl.models.dtypes import dtype_to_wire_name
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RuntimeBundle,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.model_build import resolve_model_build


def resolve_ar_model_build(
    cfg: Any,
    device: Any,
    *,
    family: str,
    default_model_path: str,
    for_rollout: bool = True,
    parameter_dtype_override: Any | None = None,
) -> ModelBuild:
    """Resolve AR model-construction fields from a whole RL cfg.

    The AR families share this exact preamble: read the model block, take
    parameter precision from the unified top-level precision bridge, and fall
    back to the family's canonical checkpoint when ``model.path`` is unset. A family
    stub supplies only its family and default path; family-specific
    post-processing (NextStep's gradient-checkpointing fold-in) stays in the
    stub.
    """

    model_config = cfg.get("model") if hasattr(cfg, "get") else None
    model_path = (model_config or {}).get("path") if model_config is not None else None
    configured_dtype = (model_config or {}).get("dtype") if model_config is not None else None
    if configured_dtype is not None:
        raise ValueError(
            "model.dtype is not configurable for AR families; remove it and use "
            "the top-level precision block so rollout and replay cannot diverge",
        )
    return resolve_model_build(
        cfg,
        device,
        family=family,
        model_name_or_path=model_path or default_model_path,
        for_rollout=for_rollout,
        parameter_dtype_override=parameter_dtype_override,
    )


def resolve_family_ar_model_build(
    cfg: Any,
    device: Any,
    *,
    for_rollout: bool = True,
    parameter_dtype_override: Any | None = None,
) -> ModelBuild:
    """Resolve one AR family build from its declarative registry recipe."""

    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )
    from vrl.utils.config import import_from_path

    model = cfg.get("model") if hasattr(cfg, "get") else None
    family = normalize_rollout_family(str((model or {}).get("family") or ""))
    if not family:
        raise ValueError("AR model-build resolution requires model.family")
    entry = get_rollout_family_entry(family)
    recipe = entry.ar_build
    if recipe is None:
        raise ValueError(f"rollout family {family!r} has no AR build descriptor")
    build = resolve_ar_model_build(
        cfg,
        device,
        family=entry.family,
        default_model_path=recipe.default_model_path,
        for_rollout=for_rollout,
        parameter_dtype_override=parameter_dtype_override,
    )
    if recipe.build_enricher is not None:
        import_from_path(recipe.build_enricher)(build, cfg)
    return build


def ar_model_config_base(
    build: ModelBuild,
    lora_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Family-shared model_config head: identity keys + typed LoRA block.

    Merges the carried ``model.lora`` block over the family's LoRA defaults so
    the yaml only needs the values it overrides. The caller appends its
    family-specific sampling / checkpoint keys to the returned dict.
    """

    config: dict[str, Any] = {
        "model_path": build.model_name_or_path,
        "dtype": dtype_to_wire_name(build.parameter_dtype),
        "device": str(build.device),
        "use_lora": build.use_lora,
    }
    if build.use_lora:
        lora = dict(lora_defaults)
        lora.update((build.model_config or {}).get("lora") or {})
        config.update(
            {
                "lora_rank": int(lora["rank"]),
                "lora_alpha": int(lora["alpha"]),
                "lora_target_modules": tuple(lora["target_modules"]),
                "lora_dropout": float(lora["dropout"]),
                "lora_init": str(lora["init"]),
            },
        )
    return config


def build_ar_runtime_bundle(
    build: ModelBuild,
    *,
    model: Any,
    capability: FamilyCapability,
    replay: bool = False,
) -> RuntimeBundle:
    """Assemble the canonical AR bundle around a family-constructed model.

    ``replay=True`` selects the trainer-side shape: no raw handle, chunked
    execution off, and minimal bundle metadata (the replay model loads no
    generation-only modules). The rollout shape exposes the model as its own
    raw handle and advertises chunked execution.

    Rollout-only quantization (FP8 attention+MLP or NVFP4 MLP-only) applies
    here under the same contract as the diffusion builder; the replay core
    keeps its configured base-precision parameters.
    """

    if replay:
        build.require_replay()
    else:
        build.require_rollout()
        from vrl.models.loader import (
            apply_rollout_quantization,
            validate_rollout_quantization_support,
        )

        validate_rollout_quantization_support(build)
        apply_rollout_quantization(model, build)

    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=None if replay else model,
        metadata={
            "model_path": build.model_name_or_path,
            "family": capability.family,
            "ar_task": capability.task,
            "use_lora": build.use_lora,
            **(minimal_replay_bundle_metadata() if replay else full_generation_bundle_metadata()),
        },
    )


def build_family_ar_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build a rollout AR model from the registry descriptor."""

    return _build_family_ar_bundle(build, replay=False)


def build_family_ar_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build a replay-only AR model from the registry descriptor."""

    build.require_replay()
    return _build_family_ar_bundle(build, replay=True)


def _build_family_ar_bundle(
    build: ModelBuild,
    *,
    replay: bool,
) -> RuntimeBundle:
    from vrl.rollouts.families.registry import get_rollout_family_entry
    from vrl.utils.config import import_from_path

    if not build.family:
        raise ValueError("descriptor-driven AR build requires build.family")
    entry = get_rollout_family_entry(build.family)
    recipe = entry.ar_build
    if recipe is None:
        raise ValueError(f"rollout family {entry.family!r} has no AR build descriptor")
    config = import_from_path(recipe.config_builder)(build)
    config_cls = import_from_path(recipe.config_cls)
    model_cls = import_from_path(recipe.replay_cls if replay else recipe.model_cls)
    return build_ar_runtime_bundle(
        build,
        model=model_cls(config_cls(**config)),
        capability=entry.capability,
        replay=replay,
    )


__all__ = [
    "ar_model_config_base",
    "build_ar_runtime_bundle",
    "build_family_ar_replay_runtime_bundle",
    "build_family_ar_runtime_bundle",
    "resolve_ar_model_build",
    "resolve_family_ar_model_build",
]
