"""Shared denoise-policy runtime-bundle orchestration.

Single-transformer denoise families share one imperative runtime/replay build
sequence. A ``DenoiseFamilyBuild`` descriptor in the model-family registry supplies
the model classes, upstream transformer classname, and scheduler;
Ray dispatches directly to the generic builder functions in this module.

Families keep custom assembly only when construction has real per-call semantics
that a descriptor cannot express. This module stays family-agnostic and resolves
descriptor import strings lazily to avoid a registry import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrl.models.interfaces.runtime import (
    ModelBuild,
    RuntimeBundle,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.loader import (
    apply_rollout_quantization,
    load_diffusers_scheduler,
    load_diffusers_transformer,
    load_flow_match_scheduler,
    validate_rollout_quantization_support,
)
from vrl.models.precision import apply_float32_precision
from vrl.models.steps.denoise.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig


def build_denoise_runtime_bundle(
    build: ModelBuild,
    *,
    model_cls: type,
    memory_owner: str,
) -> RuntimeBundle:
    """Load one diffusion rollout model and apply the shared runtime policy."""

    build.require_rollout()
    # Reject unsupported NVFP4 hardware before checkpoint loading, LoRA wrapping,
    # or any other model mutation.
    validate_rollout_quantization_support(build)
    model = model_cls.from_build(build)

    # PEFT can wrap only plain nn.Linear, while full-finetune owns the model's
    # device move. Both paths therefore quantize before the compact policy moves
    # to CUDA, but LoRA must attach before that quantization swap.
    if build.use_lora:
        model.apply_lora(build)
        lora_config = build.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"],
                lora_config["alpha"],
            )
        apply_rollout_quantization(model, build)
        if build.precision.quantization:
            model.transformer.to(model.device)
    else:
        apply_rollout_quantization(model, build)
        model.apply_full_finetune(build)

    compile_cfg = build.torch_compile or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = build.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, a caller such as the DPO trainer sets scheduler timesteps itself.

    apply_generation_memory_policy(
        model,
        memory_config=build.memory,
        owner=memory_owner,
    )
    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        precision=build.precision,
        metadata=full_generation_bundle_metadata(),
    )


def assemble_replay_bundle(
    model: object,
    build: ModelBuild,
) -> RuntimeBundle:
    """Apply shared training knobs and assemble minimal replay metadata."""

    build.require_replay()
    if build.use_lora:
        model.apply_lora(build)
    else:
        model.apply_full_finetune(build)

    compile_cfg = build.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        precision=build.precision,
        metadata=minimal_replay_bundle_metadata(),
    )


def _check_requires_lora(entry, build: ModelBuild) -> None:
    """Fail a LoRA-only family before paying the transformer load."""

    if entry.family_build.requires_lora and not build.use_lora:
        raise RuntimeError(
            f"model family {entry.family!r} is LoRA-only; set model.use_lora=true.",
        )


def resolve_family_model_build(
    root: RootConfig,
    device,
    *,
    precision: PrecisionPolicy,
    for_rollout: bool = True,
    parameter_dtype_override=None,
) -> ModelBuild:
    """Resolve a diffusion build through the canonical family registry."""

    from vrl.config.schema import RootConfig
    from vrl.families.registry import get_model_family_entry

    if not isinstance(root, RootConfig):
        raise TypeError(
            "root must be a validated RootConfig; raw DictConfig/Mapping "
            f"inputs are not accepted (got {type(root).__name__})",
        )
    if root.model is None:
        raise ValueError("validated root is missing model configuration")
    entry = get_model_family_entry(str(root.model.family))
    return entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=for_rollout,
        parameter_dtype_override=parameter_dtype_override,
    )


def build_family_runtime_bundle(
    build: ModelBuild,
    *,
    entry=None,
) -> RuntimeBundle:
    """Build rollout through a family's declarative diffusion recipe.

    ``entry`` is supplied by the canonical registry in production. Direct
    evaluation tools may omit it; their serialized ``ModelBuild.family`` then
    selects the same canonical entry rather than recreating registry data.
    """

    from vrl.families.registry import DenoiseFamilyBuild, get_model_family_entry
    from vrl.utils.config import import_from_path

    if entry is None:
        entry = get_model_family_entry(build.family)
    recipe = entry.family_build
    if not isinstance(recipe, DenoiseFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no diffusion build descriptor")
    if build.family != entry.family:
        raise ValueError(
            f"rollout build family {build.family!r} does not match entry {entry.family!r}",
        )
    _check_requires_lora(entry, build)
    logger.info("Building %s runtime bundle (registry descriptor)", entry.family)
    return build_denoise_runtime_bundle(
        build,
        model_cls=import_from_path(recipe.model_cls),
        memory_owner=f"{entry.family} model",
    )


def build_family_replay_runtime_bundle(
    build: ModelBuild,
    *,
    entry,
) -> RuntimeBundle:
    """Build replay through a family's declarative diffusion recipe."""

    from vrl.families.registry import DenoiseFamilyBuild
    from vrl.utils.config import import_from_path

    build.require_replay()
    recipe = entry.family_build
    if not isinstance(recipe, DenoiseFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no diffusion build descriptor")
    if build.family != entry.family:
        raise ValueError(
            f"replay build family {build.family!r} does not match entry {entry.family!r}",
        )
    _check_requires_lora(entry, build)
    if recipe.replay_cls is None or recipe.transformer_classname is None:
        raise ValueError(
            f"model family {entry.family!r} has no generic replay recipe; "
            "invoke its registered replay_runtime_builder instead",
        )
    logger.info(
        "Building %s replay runtime bundle (registry descriptor) from %s",
        entry.family,
        build.model_name_or_path,
    )
    model = import_from_path(recipe.replay_cls)(
        transformer=load_diffusers_transformer(build, recipe.transformer_classname),
        scheduler=(
            load_diffusers_scheduler(build, recipe.scheduler_classname)
            if recipe.scheduler_classname is not None
            else load_flow_match_scheduler(build)
        ),
        device=build.device,
    )
    # Family replay models may finish replay-only setup after their transformer
    # and scheduler exist (FLUX derives its dynamic-shift timesteps here).
    model.prepare_replay(build)
    return assemble_replay_bundle(model, build)


__all__ = [
    "assemble_replay_bundle",
    "build_denoise_runtime_bundle",
    "build_family_replay_runtime_bundle",
    "build_family_runtime_bundle",
    "resolve_family_model_build",
]
