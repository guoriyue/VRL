"""Shared diffusion runtime-bundle orchestration.

Single-transformer diffusion families share one imperative runtime/replay build
sequence. A ``DiffusionFamilyBuild`` descriptor in the model-family registry supplies
the model classes, upstream transformer classname, and scheduler;
Ray dispatches directly to the generic builder functions in this module.

Families keep custom assembly only when construction has real per-call semantics
that a descriptor cannot express. This module stays family-agnostic and resolves
descriptor import strings lazily to avoid a registry import cycle.
"""

from __future__ import annotations

from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
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
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def assemble_replay_bundle(
    model: object,
    build: ModelBuild,
) -> RuntimeBundle:
    """Shared replay-bundle assembly: training knobs + behavior metadata.

    One construction site for the lora/full-finetune + compile tail — the
    generic replay builder AND the hand-written-construction families (anima,
    cosmos3, echo) all finish here, so a knob fix (e.g. the dual-stage
    apply_full_finetune contract) lands once. Model METHODS, not loader
    helpers: the family decides which transformer trains/compiles.
    """
    build.require_replay()
    if build.use_lora:
        model.apply_lora(build)
    else:
        model.apply_full_finetune()

    compile_cfg = build.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        metadata=minimal_replay_bundle_metadata(),
    )


# -- registry-descriptor path (families with NO builder functions) -----------
# A family whose build is pure data records a ``DiffusionFamilyBuild`` on its
# registry entry. The Ray worker receives only the serialized build and looks
# the descriptor up again by its canonical family. Registry imports stay local
# to avoid an import cycle.


def _check_requires_lora(entry, build: ModelBuild) -> None:
    """Fail a LoRA-only family loud before paying the transformer load."""

    if entry.family_build.requires_lora and not build.use_lora:
        raise RuntimeError(
            f"model family {entry.family!r} is LoRA-only; set model.use_lora=true.",
        )


def _check_base_parameter_dtype(entry, build: ModelBuild) -> None:
    """Enforce a family-owned parameter dtype on every construction path."""

    expected = entry.family_build.base_parameter_dtype
    if expected is None:
        return
    from vrl.models.dtypes import dtype_to_wire_name

    if dtype_to_wire_name(build.parameter_dtype) != dtype_to_wire_name(expected):
        raise ValueError(
            f"diffusion family {entry.family!r} requires base parameter dtype "
            f"{expected!r}; got {build.parameter_dtype!r}. Resolve ModelBuild through "
            "the registered family resolver instead of overriding model dtype.",
        )


def build_family_runtime_bundle(
    build: ModelBuild,
    *,
    entry,
) -> RuntimeBundle:
    """Generic rollout builder driven by the family's registry descriptor."""

    from vrl.families.registry import DiffusionFamilyBuild
    from vrl.utils.config import import_from_path

    recipe = entry.family_build
    if not isinstance(recipe, DiffusionFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no diffusion build descriptor")
    _check_base_parameter_dtype(entry, build)
    _check_requires_lora(entry, build)
    rollout = build.require_rollout()
    # Reject unsupported NVFP4 hardware before checkpoint loading, LoRA wrapping,
    # or any other model mutation.
    validate_rollout_quantization_support(build)
    model_cls = import_from_path(recipe.model_cls)
    model = model_cls.from_build(build)
    # The executor reads the resolved autocast behavior from the model so prompt
    # embedding storage cannot silently choose the transformer's compute dtype.
    # This remains fp32/bf16/fp16 when selected GEMMs are quantized to FP8 or
    # NVFP4.
    model.autocast_dtype = rollout.autocast_dtype

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
        if rollout.quantization_format:
            model.transformer.to(model.device)
    else:
        apply_rollout_quantization(model, build)
        model.apply_full_finetune()

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
        owner=f"{entry.family} model",
    )
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        metadata=full_generation_bundle_metadata(),
    )


def build_family_replay_runtime_bundle(
    build: ModelBuild,
    *,
    entry,
) -> RuntimeBundle:
    """Generic replay builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    build.require_replay()
    from vrl.families.registry import DiffusionFamilyBuild

    recipe = entry.family_build
    if not isinstance(recipe, DiffusionFamilyBuild):
        raise ValueError(f"model family {entry.family!r} has no diffusion build descriptor")
    if build.family != entry.family:
        raise ValueError(
            f"replay build family {build.family!r} does not match entry {entry.family!r}",
        )
    _check_base_parameter_dtype(entry, build)
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
    "build_family_replay_runtime_bundle",
    "build_family_runtime_bundle",
]
