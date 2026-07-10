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
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.loader import (
    apply_rollout_quantization,
    load_diffusers_scheduler,
    load_diffusers_transformer,
    load_flow_match_scheduler,
)
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_diffusion_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    model_cls: type,
    capability: FamilyCapability,
    memory_owner: str,
    extra_metadata: Callable[[object, RuntimeBuildSpec], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Generic rollout bundle: load the family model and apply the shared policy.

    ``model_cls`` is the family rollout model (implements ``from_spec`` +
    ``apply_lora`` / ``apply_full_finetune`` + the ``trainable_modules`` /
    ``scheduler`` / ``raw_handle`` properties). ``memory_owner`` labels the VAE
    in the generation memory policy log.

    Family build-time extras (FLUX's DiffusionNFT ``previous`` adapter) are
    model knowledge and live in the family's ``apply_lora`` override, not in
    builder hooks.

    Capability flags are NOT re-published on the bundle: the registry entry's
    capability is the single stored copy and reaches the worker via the launch
    contract.

    ``extra_metadata(model, spec)`` returns family metadata merged over the
    generic keys (e.g. cosmos predict2's ``reference_image``).
    """

    model = model_cls.from_spec(spec)

    # Quantization/knob ordering is path-dependent, both directions forced:
    # - LoRA path: PEFT only wraps plain nn.Linear, so LoRA must attach BEFORE
    #   the fp8 swap (a 17B LoRA+fp8 rollout therefore still needs the bf16
    #   weights resident once; PEFT-aware quantized linears would lift that).
    # - Full path: apply_full_finetune owns the .to(device) move, so the swap
    #   must run BEFORE it — quantize on CPU, move the halved weights (a 17B
    #   bf16 transformer never fits a 32GB card, its fp8 form does). Rollout
    #   workers never backprop, so requires_grad on swapped modules is inert.
    if spec.use_lora:
        model.apply_lora(spec)
        lora_config = spec.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"],
                lora_config["alpha"],
            )
        apply_rollout_quantization(model, spec)
    else:
        apply_rollout_quantization(model, spec)
        model.apply_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    # model_path/family/task_variant/dtype/use_lora are provenance-only: no
    # runtime consumer reads them (audited 2026-07-02) — they identify what a
    # bundle was built from when inspecting it. The functional keys are the
    # full/minimal-generation marker (colocated-RAM guard) and memory_policy.
    metadata: dict[str, object] = {
        "model_path": spec.model_name_or_path,
        "family": capability.family,
        "task_variant": spec.task_variant,
        "dtype": str(spec.dtype),
        "use_lora": spec.use_lora,
        **full_generation_bundle_metadata(),
        **apply_generation_memory_policy(
            model,
            memory_config=getattr(spec, "memory", None),
            owner=memory_owner,
        ),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata(model, spec))
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        metadata=metadata,
    )


def build_diffusion_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    replay_cls: type,
    transformer_classname: str,
    capability: FamilyCapability,
    scheduler_classname: str | None = None,
    extra_metadata: Callable[[object, RuntimeBuildSpec], dict[str, object]] | None = None,
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

    Right after construction the builder calls ``model.prepare_replay(spec)``
    (a ``DiffusionModelBase`` no-op) so a family can finish replay-only setup
    with the spec in hand — FLUX sets its dynamic-shift timesteps there.

    ``extra_metadata`` follows the rollout builder's contract (family metadata
    merged over generic keys).
    """

    model = replay_cls(
        transformer=load_diffusers_transformer(spec, transformer_classname),
        scheduler=(
            load_diffusers_scheduler(spec, scheduler_classname)
            if scheduler_classname is not None
            else load_flow_match_scheduler(spec)
        ),
        device=spec.device,
    )

    model.prepare_replay(spec)
    return assemble_replay_bundle(
        model,
        spec,
        family=capability.family,
        extra_metadata=extra_metadata,
    )


def assemble_replay_bundle(
    spec_model: object,
    spec: RuntimeBuildSpec,
    *,
    family: str,
    extra_metadata: Callable[[object, RuntimeBuildSpec], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Shared replay-bundle assembly: training knobs + provenance + bundle.

    One construction site for the lora/full-finetune + compile tail — the
    generic replay builder AND the hand-written-construction families (anima,
    cosmos3, echo) all finish here, so a knob fix (e.g. the dual-stage
    apply_full_finetune contract) lands once. Model METHODS, not loader
    helpers: the family decides which transformer trains/compiles.
    """
    model = spec_model
    if spec.use_lora:
        model.apply_lora(spec)
    else:
        model.apply_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    metadata: dict[str, object] = {
        "model_path": spec.model_name_or_path,
        "family": family,
        "task_variant": spec.task_variant,
        "dtype": str(spec.dtype),
        "use_lora": spec.use_lora,
        **minimal_replay_bundle_metadata(),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata(model, spec))
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        metadata=metadata,
    )


# -- registry-descriptor path (families with NO builder functions) -----------
# A family whose build is pure data records a ``DiffusionFamilyBuild`` on its
# registry entry and points ``runtime_builder`` / ``runtime_spec_extractor``
# at the two generic functions below. The extractor stamps the canonical
# family onto the spec (it rides the Ray launch payload), and the builder
# looks the recipe back up worker-side. Registry imports stay local: the
# registry module imports family runtimes, so a top-level import here would
# cycle.


def _family_build_entry(family: str | None):
    from vrl.rollouts.families.registry import get_rollout_family_entry

    if not family:
        raise ValueError(
            "the generic family builder requires spec.family; build the spec "
            "through vrl.models.diffusion.build:extract_family_runtime_spec",
        )
    entry = get_rollout_family_entry(family)
    if entry.build is None:
        raise ValueError(
            f"rollout family {entry.family!r} has no DiffusionFamilyBuild "
            "descriptor; it builds through its own runtime.py functions",
        )
    return entry


def _check_requires_lora(entry, spec: RuntimeBuildSpec) -> None:
    """Fail a LoRA-only family loud before paying the transformer load."""

    if entry.build.requires_lora and not spec.use_lora:
        raise RuntimeError(
            f"rollout family {entry.family!r} is LoRA-only; "
            "set model.use_lora=true.",
        )


def extract_family_runtime_spec(cfg, device, weight_dtype) -> RuntimeBuildSpec:
    """Generic extractor: resolve the family from ``cfg.model.family``."""

    from vrl.models.runtime_config import extract_runtime_spec

    entry = _family_build_entry(str(cfg.model.family))
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant=entry.build.task_variant,
        family=entry.family,
    )


def build_family_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic rollout builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    entry = _family_build_entry(spec.family)
    recipe = entry.build
    _check_requires_lora(entry, spec)
    logger.info("Building %s runtime bundle (registry descriptor)", entry.family)
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=import_from_path(recipe.model_cls),
        capability=entry.capability,
        memory_owner=recipe.memory_owner,
    )


def build_family_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic replay builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    entry = _family_build_entry(spec.family)
    recipe = entry.build
    if recipe.replay_cls is None or recipe.transformer_classname is None:
        raise ValueError(
            f"rollout family {entry.family!r} keeps its own replay builder; "
            "the registry descriptor covers the rollout side only",
        )
    _check_requires_lora(entry, spec)
    logger.info(
        "Building %s replay runtime bundle (registry descriptor) from %s",
        entry.family,
        spec.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=import_from_path(recipe.replay_cls),
        transformer_classname=recipe.transformer_classname,
        scheduler_classname=recipe.scheduler_classname,
        capability=entry.capability,
    )


def build_family_runtime_bundle_from_cfg(cfg, device, weight_dtype) -> RuntimeBundle:
    """Whole-cfg convenience for the trainer path: cfg -> spec -> bundle."""

    return build_family_runtime_bundle(
        extract_family_runtime_spec(cfg, device, weight_dtype),
    )


def build_family_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype) -> RuntimeBundle:
    """Whole-cfg convenience for the trainer path: cfg -> spec -> replay bundle."""

    return build_family_replay_runtime_bundle(
        extract_family_runtime_spec(cfg, device, weight_dtype),
    )


__all__ = [
    "assemble_replay_bundle",
    "build_diffusion_replay_runtime_bundle",
    "build_diffusion_runtime_bundle",
    "build_family_replay_runtime_bundle",
    "build_family_replay_runtime_bundle_from_cfg",
    "build_family_runtime_bundle",
    "build_family_runtime_bundle_from_cfg",
    "extract_family_runtime_spec",
]
