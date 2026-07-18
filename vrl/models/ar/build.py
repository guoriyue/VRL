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

from vrl.models.dtypes import dtype_to_wire_name
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RuntimeBundle,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.precision import apply_float32_precision


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
        "revision": (build.model_config or {}).get("revision") or None,
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


def build_family_ar_bundle(
    build: ModelBuild,
    *,
    replay: bool,
    entry: Any,
) -> RuntimeBundle:
    from vrl.utils.config import import_from_path

    if build.family != entry.family:
        raise ValueError(
            f"AR build family {build.family!r} does not match entry {entry.family!r}",
        )
    from vrl.families.registry import ARFamilyBuild

    recipe = entry.family_build
    if not isinstance(recipe, ARFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no AR build descriptor")
    config = import_from_path(recipe.config_builder)(build)
    config_cls = import_from_path(recipe.config_cls)
    model_cls = import_from_path(recipe.replay_cls if replay else recipe.model_cls)
    model = model_cls(config_cls(**config))
    if replay:
        build.require_replay()
    else:
        build.require_rollout()
        from vrl.models.loader import apply_rollout_quantization

        apply_rollout_quantization(model, build)

    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=None if replay else model,
        precision=build.precision,
        outer_autocast=build.outer_autocast,
        metadata=(
            minimal_replay_bundle_metadata() if replay else full_generation_bundle_metadata()
        ),
    )


__all__ = [
    "ar_model_config_base",
    "build_family_ar_bundle",
]
