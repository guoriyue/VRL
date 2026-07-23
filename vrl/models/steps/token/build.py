"""Shared descriptor-driven token-policy construction and bundle assembly.

Every token family (janus_pro, nextstep_1, ...) assembled its rollout and replay
``RuntimeBundle`` with the same model/config/import sequence. The registry now
owns those construction inputs and this module owns the sequence once.

Like the diffusion counterpart, the registry records model/config import paths
and points every token family at this module. Family runtime modules keep only
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


def token_model_config_base(build: ModelBuild) -> dict[str, Any]:
    """Project shared identity keys and explicit LoRA overrides.

    The family config dataclass owns LoRA defaults. This projection only
    carries fields explicitly provided by ``model.lora`` so partial overrides
    cannot shadow or duplicate those defaults.
    """

    config: dict[str, Any] = {
        "model_path": build.model_name_or_path,
        "revision": (build.model_config or {}).get("revision") or None,
        "dtype": dtype_to_wire_name(build.parameter_dtype),
        "device": str(build.device),
        "use_lora": build.use_lora,
    }
    if build.use_lora:
        lora = (build.model_config or {}).get("lora") or {}
        if "rank" in lora:
            config["lora_rank"] = int(lora["rank"])
        if "alpha" in lora:
            config["lora_alpha"] = int(lora["alpha"])
        if "target_modules" in lora:
            config["lora_target_modules"] = tuple(lora["target_modules"])
        if "dropout" in lora:
            config["lora_dropout"] = float(lora["dropout"])
        if "init" in lora:
            config["lora_init"] = str(lora["init"])
    return config


def build_token_family_bundle(
    build: ModelBuild,
    *,
    replay: bool,
    entry: Any,
) -> RuntimeBundle:
    from vrl.utils.config import import_from_path

    if build.family != entry.family:
        raise ValueError(
            f"token build family {build.family!r} does not match entry {entry.family!r}",
        )
    from vrl.families.registry import TokenFamilyBuild

    recipe = entry.family_build
    if not isinstance(recipe, TokenFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no token build descriptor")
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
        metadata=(
            minimal_replay_bundle_metadata() if replay else full_generation_bundle_metadata()
        ),
    )


__all__ = [
    "build_token_family_bundle",
    "token_model_config_base",
]
