"""Shared diffusion runtime-bundle orchestration.

Every single-transformer diffusion family (sd3_5, flux, qwen_image, ...) built
its runtime and replay bundles with the same imperative sequence, copied per
family. That sequence lives here once; a family's ``runtime.py`` keeps only a
thin ``build_<family>_*`` stub that passes its ``Model`` / ``ReplayModel`` class,
capability, and diffusers transformer classname in.

The stub — not this module — stays the dispatch target: the rollout family
registry stores ``runtime_builder`` as a ``module:function`` string that Ray
imports and calls with ``spec`` only, so each family must still expose a
module-level ``build_<family>_runtime_bundle(spec)``. This module is
family-agnostic on purpose: it must not import any family model class (the stub
supplies them), which also keeps it free of import cycles.

Multi-transformer / single-file families (wan, cosmos, anima) construct their
replay model differently and keep their own builder; they can still reuse
``build_diffusion_runtime_bundle`` for the rollout side.
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
    compile_transformer,
    enable_transformer_full_finetune,
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
    supports_reference_conditioning: bool = False,
    after_lora: Callable[[object, RuntimeBuildSpec], None] | None = None,
    runtime_caps: dict[str, object] | None = None,
    extra_metadata: Callable[[object, RuntimeBuildSpec], dict[str, object]] | None = None,
) -> RuntimeBundle:
    """Generic rollout bundle: load the family model and apply the shared policy.

    ``model_cls`` is the family rollout model (implements ``from_spec`` +
    ``apply_lora`` / ``apply_full_finetune`` + the ``trainable_modules`` /
    ``scheduler`` / ``raw_handle`` properties). ``memory_owner`` labels the VAE
    in the generation memory policy log.

    ``after_lora`` is a family extension hook run once after ``apply_lora`` (LoRA
    path only). FLUX uses it to attach the frozen DiffusionNFT ``previous``
    adapter; families with no such step leave it ``None``. Keeping it a hook
    means the generic body never learns family-specific concepts.

    ``runtime_caps``, when given, replaces the default caps dict verbatim. The
    cosmos families historically publish caps WITHOUT ``family_capability``
    (the worker then falls back to the registry-declared capability — see
    ``FamilyCapability.with_runtime_caps``), so a migrated family passes its
    exact historical dict to keep behavior bit-identical. Unifying the caps
    contract is a separate audit, not this builder's job.

    ``extra_metadata(model, spec)`` returns family metadata merged over the
    generic keys (e.g. cosmos predict2's ``reference_image``).
    """

    model = model_cls.from_spec(spec)

    if spec.use_lora:
        model.apply_lora(spec)
        lora_config = spec.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"],
                lora_config["alpha"],
            )
        if after_lora is not None:
            after_lora(model, spec)
    else:
        model.apply_full_finetune()

    apply_rollout_quantization(model, spec)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

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
        runtime_caps=(
            dict(runtime_caps)
            if runtime_caps is not None
            else {
                "family_capability": capability.to_dict(),
                "supports_reference_conditioning": supports_reference_conditioning,
            }
        ),
        metadata=metadata,
    )


def build_diffusion_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    replay_cls: type,
    transformer_classname: str,
    capability: FamilyCapability,
    scheduler_classname: str | None = None,
    supports_reference_conditioning: bool = False,
    after_construct: Callable[[object, RuntimeBuildSpec], None] | None = None,
    after_lora: Callable[[object, RuntimeBuildSpec], None] | None = None,
    runtime_caps: dict[str, object] | None = None,
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

    Two family extension hooks keep family-specific replay logic out of the
    generic body: ``after_construct`` runs right after the replay model is built
    (FLUX sets its dynamic-shift timesteps here); ``after_lora`` runs after
    ``apply_lora`` on the LoRA path (FLUX attaches the frozen NFT ``previous``
    adapter). Families needing neither leave both ``None``.

    ``runtime_caps`` / ``extra_metadata`` follow the rollout builder's contract
    (verbatim caps override; family metadata merged over generic keys).
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

    if after_construct is not None:
        after_construct(model, spec)

    if spec.use_lora:
        model.apply_lora(spec)
        if after_lora is not None:
            after_lora(model, spec)
    else:
        enable_transformer_full_finetune(model)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        compile_transformer(model, compile_cfg["mode"])

    metadata: dict[str, object] = {
        "model_path": spec.model_name_or_path,
        "family": capability.family,
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
        runtime_caps=(
            dict(runtime_caps)
            if runtime_caps is not None
            else {
                "family_capability": capability.to_dict(),
                "supports_reference_conditioning": supports_reference_conditioning,
            }
        ),
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
    logger.info("Building %s runtime bundle (registry descriptor)", entry.family)
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=import_from_path(recipe.model_cls),
        capability=entry.capability,
        memory_owner=recipe.memory_owner,
        runtime_caps=None if recipe.runtime_caps is None else dict(recipe.runtime_caps),
    )


def build_family_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic replay builder driven by the family's registry descriptor."""

    from vrl.utils.config import import_from_path

    entry = _family_build_entry(spec.family)
    recipe = entry.build
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
        runtime_caps=None if recipe.runtime_caps is None else dict(recipe.runtime_caps),
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
    "build_diffusion_replay_runtime_bundle",
    "build_diffusion_runtime_bundle",
    "build_family_replay_runtime_bundle",
    "build_family_replay_runtime_bundle_from_cfg",
    "build_family_runtime_bundle",
    "build_family_runtime_bundle_from_cfg",
    "extract_family_runtime_spec",
]
