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

from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.models.loader import (
    load_diffusers_scheduler,
    load_diffusers_transformer,
    load_flow_match_scheduler,
    validate_rollout_quantization_support,
)
from vrl.models.precision import apply_float32_precision
from vrl.nn.optimization import apply_rollout_optimizations
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_denoise_runtime_bundle(
    build: ModelBuild,
    *,
    model_cls: type,
) -> RuntimeBundle:
    """Load one diffusion rollout model and apply the shared runtime policy."""

    rollout = build.require_rollout()
    # Reject unsupported NVFP4 hardware before checkpoint loading, LoRA wrapping,
    # or any other model mutation.
    validate_rollout_quantization_support(build)
    model = model_cls.from_build(build)
    pipeline_offload = rollout.pipeline_offload_mode != "none"

    # PEFT can wrap only plain nn.Linear, while full-finetune owns the model's
    # device move. Both paths therefore optimize before the compact policy moves
    # to CUDA, but LoRA must attach before the quantization swap.
    def move_to_device() -> None:
        """Place the policy on its device between the quantize and compile passes.

        Both branches move the (now compact) policy AFTER quantization and
        BEFORE compile: ``requires_grad_`` / ``.to()`` must act on the real
        module, not on a compiled wrapper.
        """

        if build.use_lora:
            if build.precision.quantization and not pipeline_offload:
                model.transformer.to(model.device)
        else:
            model.apply_full_finetune(build)

    if build.use_lora:
        model.apply_lora(build)
        lora_config = build.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"],
                lora_config["alpha"],
            )
    # Quantize -> device move -> compile -> offload hooks -> VAE decode memory.
    # The whole sequence and its ordering constraints live in the pass layer.
    apply_rollout_optimizations(model, build, before_compile=move_to_device)

    num_steps = build.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, a caller such as the DPO trainer sets scheduler timesteps itself.

    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        precision=build.precision,
        loads_full_generation_modules=True,
        adapter_roots=model.adapter_roots,
    )


def assemble_replay_bundle(
    model: object,
    build: ModelBuild,
) -> RuntimeBundle:
    """Apply shared training knobs and assemble a minimal replay bundle."""

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
        loads_full_generation_modules=False,
        adapter_roots=model.adapter_roots,
    )


def _check_requires_lora(entry, build: ModelBuild) -> None:
    """Fail a LoRA-only family before paying the transformer load."""

    if entry.family_build.requires_lora and not build.use_lora:
        raise RuntimeError(
            f"model family {entry.family!r} is LoRA-only; set model.use_lora=true.",
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

    from vrl.models.families.registry import DenoiseFamilyBuild, get_model_family_entry
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
    )


def build_family_replay_runtime_bundle(
    build: ModelBuild,
    *,
    entry,
) -> RuntimeBundle:
    """Build replay through a family's declarative diffusion recipe."""

    from vrl.models.families.registry import DenoiseFamilyBuild
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
]
