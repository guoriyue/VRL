"""Shared descriptor-driven AR runtime construction and bundle assembly.

Every AR family (janus_pro, nextstep_1, ...) assembled its rollout and replay
``RuntimeBundle`` with the same model/config/import sequence. The registry now
owns those construction inputs and this module owns the sequence once.

Like the diffusion counterpart, the registry records model/config import paths
and points every AR family at this module. Family runtime modules keep only
their request executor and config projection; repeated rollout/replay builder
and spec-extractor facades are gone.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.capabilities import FamilyCapability
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.runtime_config import extract_runtime_spec


def extract_ar_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
    *,
    family: str,
    ar_task: str,
    default_model_path: str,
) -> RuntimeBuildSpec:
    """Slice AR runtime construction fields out of a whole RL cfg.

    The AR families share this exact preamble: read the model block, prefer
    ``model.dtype`` over the caller's ``weight_dtype`` over bf16, and fall back
    to the family's canonical checkpoint when ``model.path`` is unset. A family
    stub supplies only its ``ar_task`` and default path; family-specific
    post-processing (NextStep's gradient-checkpointing fold-in) stays in the
    stub.
    """

    model_config = cfg.get("model") if hasattr(cfg, "get") else None
    model_path = (model_config or {}).get("path") if model_config is not None else None
    dtype = (model_config or {}).get("dtype") if model_config is not None else None
    return extract_runtime_spec(
        cfg,
        device,
        dtype_to_config_string(dtype if dtype is not None else (weight_dtype or "bfloat16")),
        family=family,
        ar_task=ar_task,
        model_name_or_path=model_path or default_model_path,
    )


def extract_family_ar_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Extract one AR family spec from its declarative registry recipe."""

    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )
    from vrl.utils.config import import_from_path

    model = cfg.get("model") if hasattr(cfg, "get") else None
    family = normalize_rollout_family(str((model or {}).get("family") or ""))
    if not family:
        raise ValueError("AR runtime extraction requires model.family")
    entry = get_rollout_family_entry(family)
    recipe = entry.ar_build
    if recipe is None:
        raise ValueError(f"rollout family {family!r} has no AR build descriptor")
    spec = extract_ar_runtime_spec(
        cfg,
        device,
        weight_dtype,
        family=entry.family,
        ar_task=entry.task,
        default_model_path=recipe.default_model_path,
    )
    if recipe.spec_enricher is not None:
        import_from_path(recipe.spec_enricher)(spec, cfg)
    return spec


def ar_model_config_base(
    spec: RuntimeBuildSpec,
    lora_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Family-shared model_config head: identity keys + typed LoRA block.

    Merges the carried ``model.lora`` block over the family's LoRA defaults so
    the yaml only needs the values it overrides. The caller appends its
    family-specific sampling / checkpoint keys to the returned dict.
    """

    config: dict[str, Any] = {
        "model_path": spec.model_name_or_path,
        "dtype": dtype_to_config_string(spec.dtype),
        "device": str(spec.device),
        "use_lora": spec.use_lora,
    }
    if spec.use_lora:
        lora = dict(lora_defaults)
        lora.update((spec.model_config or {}).get("lora") or {})
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
    spec: RuntimeBuildSpec,
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

    Rollout-only quantization (``precision.rollout=fp8``) applies here, same
    contract as the diffusion builder; the replay core keeps its bf16 master.
    """

    if not replay:
        from vrl.models.loader import apply_rollout_quantization

        apply_rollout_quantization(model, spec)

    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=None if replay else model,
        metadata={
            "model_path": spec.model_name_or_path,
            "family": capability.family,
            "ar_task": spec.ar_task,
            "use_lora": spec.use_lora,
            **(
                minimal_replay_bundle_metadata()
                if replay
                else full_generation_bundle_metadata()
            ),
        },
    )


def build_family_ar_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a rollout AR model from the registry descriptor."""

    return _build_family_ar_bundle(spec, replay=False)


def build_family_ar_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a replay-only AR model from the registry descriptor."""

    return _build_family_ar_bundle(spec, replay=True)


def _build_family_ar_bundle(
    spec: RuntimeBuildSpec,
    *,
    replay: bool,
) -> RuntimeBundle:
    from vrl.rollouts.families.registry import get_rollout_family_entry
    from vrl.utils.config import import_from_path

    if not spec.family:
        raise ValueError("descriptor-driven AR build requires spec.family")
    entry = get_rollout_family_entry(spec.family)
    recipe = entry.ar_build
    if recipe is None:
        raise ValueError(f"rollout family {entry.family!r} has no AR build descriptor")
    config = import_from_path(recipe.config_builder)(spec)
    config_cls = import_from_path(recipe.config_cls)
    model_cls = import_from_path(recipe.replay_cls if replay else recipe.model_cls)
    return build_ar_runtime_bundle(
        spec,
        model=model_cls(config_cls(**config)),
        capability=entry.capability,
        replay=replay,
    )


__all__ = [
    "ar_model_config_base",
    "build_ar_runtime_bundle",
    "build_family_ar_replay_runtime_bundle",
    "build_family_ar_runtime_bundle",
    "extract_ar_runtime_spec",
    "extract_family_ar_runtime_spec",
]
