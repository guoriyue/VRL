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

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        runtime_caps={
            "family_capability": capability.to_dict(),
            "supports_reference_conditioning": supports_reference_conditioning,
        },
        metadata={
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
        },
    )


def build_diffusion_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    replay_cls: type,
    transformer_classname: str,
    capability: FamilyCapability,
    supports_reference_conditioning: bool = False,
    after_construct: Callable[[object, RuntimeBuildSpec], None] | None = None,
    after_lora: Callable[[object, RuntimeBuildSpec], None] | None = None,
) -> RuntimeBundle:
    """Generic replay bundle for single-transformer diffusion families.

    Loads only the trainable transformer + scheduler (no prompt/VAE modules), so
    the trainer's colocated-RAM guard sees ``minimal_replay_bundle_metadata``.
    ``transformer_classname`` is the diffusers class to instantiate (e.g.
    ``"SD3Transformer2DModel"``). Multi-transformer families do not use this.

    Two family extension hooks keep family-specific replay logic out of the
    generic body: ``after_construct`` runs right after the replay model is built
    (FLUX sets its dynamic-shift timesteps here); ``after_lora`` runs after
    ``apply_lora`` on the LoRA path (FLUX attaches the frozen NFT ``previous``
    adapter). Families needing neither leave both ``None``.
    """

    model = replay_cls(
        transformer=load_diffusers_transformer(spec, transformer_classname),
        scheduler=load_flow_match_scheduler(spec),
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

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        runtime_caps={
            "family_capability": capability.to_dict(),
            "supports_reference_conditioning": supports_reference_conditioning,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": capability.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


__all__ = [
    "build_diffusion_replay_runtime_bundle",
    "build_diffusion_runtime_bundle",
]
