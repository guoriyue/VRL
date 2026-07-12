"""Shared diffusion runtime-bundle orchestration.

Single-transformer diffusion families share one imperative runtime/replay build
sequence. A ``DiffusionFamilyBuild`` descriptor in the rollout registry supplies
the model classes, upstream transformer classname, scheduler, and memory owner;
Ray dispatches directly to the generic builder functions in this module.

Families keep custom assembly only when construction has real per-call semantics
that a descriptor cannot express. This module stays family-agnostic and resolves
descriptor import strings lazily to avoid a registry import cycle.
"""

from __future__ import annotations

from collections.abc import Callable

from vrl.generation.capabilities import FamilyCapability
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


def build_diffusion_runtime_bundle(
    build: ModelBuild,
    *,
    model_cls: type,
    capability: FamilyCapability,
    memory_owner: str,
    extra_metadata: Callable[[object, ModelBuild], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Generic rollout bundle: load the family model and apply the shared policy.

    ``model_cls`` is the family rollout model (implements ``from_build`` +
    ``apply_lora`` / ``apply_full_finetune`` + the ``trainable_modules`` /
    ``scheduler`` / ``raw_handle`` properties). ``memory_owner`` labels the VAE
    in the generation memory policy log.

    Family build-time extras (FLUX's DiffusionNFT ``previous`` adapter) are
    model knowledge and live in the family's ``apply_lora`` override, not in
    builder hooks.

    Capability flags are NOT re-published on the bundle: the registry entry's
    capability is the single stored copy and reaches the worker via the launch
    contract.

    ``extra_metadata(model, build)`` returns family metadata merged over the
    generic keys (e.g. cosmos predict2's ``reference_image``).
    """

    rollout = build.require_rollout()
    # Reject unsupported NVFP4 hardware before checkpoint loading, LoRA wrapping,
    # or any other model mutation.
    validate_rollout_quantization_support(build)
    model = model_cls.from_build(build)
    # The executor reads the resolved autocast behavior from the model so prompt
    # embedding storage cannot silently choose the transformer's compute dtype.
    # This remains fp32/bf16/fp16 when selected GEMMs are quantized to FP8 or
    # NVFP4.
    model.autocast_dtype = rollout.autocast_dtype

    # Quantization/knob ordering is path-dependent, both directions forced:
    # - LoRA path: PEFT only wraps plain nn.Linear, so LoRA attaches on CPU
    #   BEFORE the quantization swap. Quantization then drops unsynced base-precision
    #   masters and only the compact policy moves to GPU. This ordering is what
    #   lets a 17B quantized LoRA rollout avoid a >32GB bf16 construction peak.
    # - Full path: apply_full_finetune owns the .to(device) move, so the swap
    #   must run BEFORE it — quantize on CPU, then move the compact cache (a 17B
    #   bf16 transformer may not fit a 32GB card in source precision). Rollout
    #   workers never backprop, so requires_grad on swapped modules is inert.
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
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    # The functional keys are the full/minimal-generation marker
    # (colocated-RAM guard) and memory_policy; the rest is provenance.
    metadata: dict[str, object] = {
        **_provenance_metadata(
            build,
            capability.family,
            task_variant=capability.task,
        ),
        **full_generation_bundle_metadata(),
        **apply_generation_memory_policy(
            model,
            memory_config=getattr(build, "memory", None),
            owner=memory_owner,
        ),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata(model, build))
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        metadata=metadata,
    )


def build_diffusion_replay_runtime_bundle(
    build: ModelBuild,
    *,
    replay_cls: type,
    transformer_classname: str,
    capability: FamilyCapability,
    scheduler_classname: str | None = None,
    extra_metadata: Callable[[object, ModelBuild], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Generic replay bundle for single-transformer diffusion families.

    Loads only the trainable transformer + scheduler (no prompt/VAE modules), so
    the trainer's colocated-RAM guard sees ``minimal_replay_bundle_metadata``.
    ``transformer_classname`` is the diffusers class to instantiate (e.g.
    ``"SD3Transformer2DModel"``). Multi-transformer families do not use this.
    ``scheduler_classname`` mirrors it for the scheduler: ``None`` loads the
    flow-match Euler scheduler (every flow-matching family); a classname loads
    that diffusers scheduler instead (Cosmos Predict2.5 ships UniPC, and replay
    must recompute log-probs under the same schedule the rollout sampled with).

    Right after construction the builder calls ``model.prepare_replay(build)``
    (a ``DiffusionModelBase`` no-op) so a family can finish replay-only setup
    with the build inputs in hand — FLUX sets its dynamic-shift timesteps there.

    ``extra_metadata`` follows the rollout builder's contract (family metadata
    merged over generic keys).
    """

    build.require_replay()
    model = replay_cls(
        transformer=load_diffusers_transformer(build, transformer_classname),
        scheduler=(
            load_diffusers_scheduler(build, scheduler_classname)
            if scheduler_classname is not None
            else load_flow_match_scheduler(build)
        ),
        device=build.device,
    )

    model.prepare_replay(build)
    return assemble_replay_bundle(
        model,
        build,
        family=capability.family,
        task_variant=capability.task,
        extra_metadata=extra_metadata,
    )


def assemble_replay_bundle(
    model: object,
    build: ModelBuild,
    *,
    family: str,
    task_variant: str | None = None,
    extra_metadata: Callable[[object, ModelBuild], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Shared replay-bundle assembly: training knobs + provenance + bundle.

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

    metadata: dict[str, object] = {
        **_provenance_metadata(build, family, task_variant=task_variant),
        **minimal_replay_bundle_metadata(),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata(model, build))
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        metadata=metadata,
    )


def _provenance_metadata(
    build: ModelBuild,
    family: str,
    *,
    task_variant: str | None = None,
) -> dict[str, object]:
    """Single construction site for the provenance keys on both bundle kinds.

    The family registry remains the source of truth for the display-only task
    variant; ``ModelBuild`` must not carry a duplicate field that no runtime
    branch consumes.
    """

    from vrl.rollouts.families.registry import get_rollout_family_entry

    if task_variant is None:
        entry = get_rollout_family_entry(family)
        task_variant = (
            entry.build.task_variant if entry.build is not None else entry.capability.task
        )

    return {
        "model_path": build.model_name_or_path,
        "family": family,
        "task_variant": task_variant,
        "parameter_dtype": str(build.parameter_dtype),
        "use_lora": build.use_lora,
    }


# -- registry-descriptor path (families with NO builder functions) -----------
# A family whose build is pure data records a ``DiffusionFamilyBuild`` on its
# registry entry and points ``runtime_builder`` / ``model_build_resolver``
# at the two generic functions below. The resolver stamps the canonical
# family onto the build (it rides the Ray launch payload), and the builder
# looks the recipe back up worker-side. Registry imports stay local: the
# registry module imports family runtimes, so a top-level import here would
# cycle.


def _family_build_entry(family: str | None):
    from vrl.rollouts.families.registry import get_rollout_family_entry

    if not family:
        raise ValueError(
            "the generic family builder requires build.family; resolve the build "
            "through vrl.models.diffusion.build:resolve_family_model_build",
        )
    entry = get_rollout_family_entry(family)
    if entry.build is None:
        raise ValueError(
            f"rollout family {entry.family!r} has no DiffusionFamilyBuild "
            "descriptor; it builds through its own runtime.py functions",
        )
    return entry


def _check_requires_lora(entry, build: ModelBuild) -> None:
    """Fail a LoRA-only family loud before paying the transformer load."""

    if entry.build.requires_lora and not build.use_lora:
        raise RuntimeError(
            f"rollout family {entry.family!r} is LoRA-only; set model.use_lora=true.",
        )


def _check_base_parameter_dtype(entry, build: ModelBuild) -> None:
    """Enforce a family-owned parameter dtype on every construction path."""

    expected = entry.build.base_parameter_dtype
    if expected is None:
        return
    from vrl.models.dtypes import dtype_to_wire_name

    if dtype_to_wire_name(build.parameter_dtype) != dtype_to_wire_name(expected):
        raise ValueError(
            f"diffusion family {entry.family!r} requires base parameter dtype "
            f"{expected!r}; got {build.parameter_dtype!r}. Resolve ModelBuild through "
            "the registered family resolver instead of overriding model dtype.",
        )


def resolve_family_model_build(
    cfg,
    device,
    *,
    for_rollout: bool = True,
    parameter_dtype_override=None,
) -> ModelBuild:
    """Resolve one diffusion family build from ``cfg.model.family``."""

    from vrl.models.model_build import resolve_model_build

    entry = _family_build_entry(str(cfg.model.family))
    configured_dtype = cfg.model.get("dtype")
    family_dtype = entry.build.base_parameter_dtype
    if configured_dtype is not None:
        reason = (
            f"its base parameter dtype is fixed to {family_dtype!r} by the family build descriptor"
            if family_dtype is not None
            else "parameter precision follows the top-level precision block"
        )
        raise ValueError(
            f"model.dtype is not configurable for diffusion family {entry.family!r}; "
            f"{reason}. Remove model.dtype.",
        )
    if family_dtype is not None and parameter_dtype_override is not None:
        from vrl.models.dtypes import dtype_to_wire_name

        if dtype_to_wire_name(parameter_dtype_override) != dtype_to_wire_name(
            family_dtype,
        ):
            raise ValueError(
                f"diffusion family {entry.family!r} requires base parameter dtype "
                f"{family_dtype!r}; explicit parameter_dtype_override="
                f"{parameter_dtype_override!r} conflicts with that invariant.",
            )
    # The shared resolver derives ordinary parameter precision from the role:
    # Replay follows precision.training.dtype; rollout follows
    # precision.rollout.dtype. Quantization is projected separately.
    # Only a genuine family invariant (or an explicit direct-tool override) may
    # replace that config-derived value.
    parameter_dtype = family_dtype or parameter_dtype_override
    return resolve_model_build(
        cfg,
        device,
        family=entry.family,
        for_rollout=for_rollout,
        parameter_dtype_override=parameter_dtype,
    )


def build_family_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Generic rollout builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    entry = _family_build_entry(build.family)
    recipe = entry.build
    _check_base_parameter_dtype(entry, build)
    _check_requires_lora(entry, build)
    logger.info("Building %s runtime bundle (registry descriptor)", entry.family)
    return build_diffusion_runtime_bundle(
        build,
        model_cls=import_from_path(recipe.model_cls),
        capability=entry.capability,
        memory_owner=recipe.memory_owner,
    )


def build_family_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Generic replay builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    build.require_replay()
    entry = _family_build_entry(build.family)
    recipe = entry.build
    _check_base_parameter_dtype(entry, build)
    _check_requires_lora(entry, build)
    if recipe.replay_cls is None or recipe.transformer_classname is None:
        # Hand-written-replay families (echo/cosmos3/anima) register their
        # builder on the entry; the descriptor covers the rollout side only.
        if entry.replay_runtime_builder is not None:
            logger.info(
                "Building %s replay runtime bundle (hand-written builder) from %s",
                entry.family,
                build.model_name_or_path,
            )
            return import_from_path(entry.replay_runtime_builder)(build)
        raise ValueError(
            f"rollout family {entry.family!r} declares neither a replay recipe "
            "nor a replay_runtime_builder; wire one on its registry entry",
        )
    logger.info(
        "Building %s replay runtime bundle (registry descriptor) from %s",
        entry.family,
        build.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        build,
        replay_cls=import_from_path(recipe.replay_cls),
        transformer_classname=recipe.transformer_classname,
        scheduler_classname=recipe.scheduler_classname,
        capability=entry.capability,
    )


def build_family_replay_runtime_bundle_from_cfg(cfg, device) -> RuntimeBundle:
    """Whole-cfg convenience for the trainer path: cfg -> build -> replay bundle."""

    return build_family_replay_runtime_bundle(
        resolve_family_model_build(
            cfg,
            device,
            for_rollout=False,
        ),
    )


__all__ = [
    "assemble_replay_bundle",
    "build_diffusion_replay_runtime_bundle",
    "build_diffusion_runtime_bundle",
    "build_family_replay_runtime_bundle",
    "build_family_replay_runtime_bundle_from_cfg",
    "build_family_runtime_bundle",
    "resolve_family_model_build",
]
