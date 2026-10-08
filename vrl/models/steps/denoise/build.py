"""Shared denoise-policy runtime-bundle orchestration.

Single-transformer denoise families share one imperative runtime/replay build
sequence. A ``DenoiseFamilyBuild`` descriptor in the model-family registry supplies
the model classes, upstream transformer classname, and scheduler;
The registry dispatches rollout and replay construction to these shared builders.

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
)
from vrl.models.parking import module_on_host
from vrl.models.precision import apply_float32_precision
from vrl.nn.optimization import QuantizationPass, apply_rollout_optimizations
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_denoise_runtime_bundle(
    build: ModelBuild,
    *,
    model_cls: type,
) -> RuntimeBundle:
    """Load one diffusion rollout model and apply the shared runtime policy."""

    logger.info("Building %s runtime bundle", build.family)
    rollout = build.require_rollout()
    # Reject unsupported NVFP4 hardware before checkpoint loading, LoRA wrapping,
    # or any other model mutation.
    QuantizationPass.validate_support(build)
    model = model_cls.from_build(build)
    memory = build.generation_memory
    cpu_resident = frozenset(() if memory is None else memory.cpu_resident)
    if cpu_resident and getattr(model, "_cpu_resident", frozenset()) != cpu_resident:
        raise RuntimeError(
            f"{type(model).__name__}.from_build did not honor model.memory.cpu_resident="
            f"{sorted(cpu_resident)}; host placement is implemented by the shared "
            "DiffusersPipelineModelBase loader only",
        )
    pipeline_offload = rollout.pipeline_offload_mode != "none"

    # The single rollout placement site. Attach and full-finetune leave the
    # trainable roots where the loader put them; the compact policy moves AFTER
    # quantization (PEFT wraps only plain nn.Linear, so LoRA attaches first) and
    # BEFORE compile (``requires_grad_`` / ``.to()`` must act on the real
    # module, not on a compiled wrapper).
    def move_to_device() -> None:
        if not build.use_lora:
            model.apply_full_finetune(build)
        if pipeline_offload:
            # Accelerate's hooks, installed after this seam, own residency.
            return
        for root in model.trainable_modules.values():
            # A loader that already dispatched a root (block-partitioned H3)
            # leaves nothing on the host; only host-resident roots move.
            if module_on_host(root):
                root.to(model.device)

    if build.use_lora:
        model.apply_lora(build)
        lora_config = build.require_lora_config()
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d)",
            lora_config.rank,
            lora_config.alpha,
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
        adapter_roots=model.adapter_roots,
    )


def assemble_replay_bundle(
    model: object,
    build: ModelBuild,
) -> RuntimeBundle:
    """Apply shared training knobs to a replay model and assemble its bundle."""

    build.require_replay()
    if build.use_lora:
        model.apply_lora(build)
    else:
        model.apply_full_finetune(build)

    compile_cfg = build.torch_compile
    if compile_cfg is not None:
        model.torch_compile_transformer(compile_cfg["mode"], regional=compile_cfg["regional"])

    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        precision=build.precision,
        adapter_roots=model.adapter_roots,
    )


def build_family_replay_runtime_bundle(
    build: ModelBuild,
    *,
    entry,
    materialize_weights: bool = True,
) -> RuntimeBundle:
    """Build replay through a family's declarative diffusion recipe.

    Called only by ``ModelFamilyEntry.build_replay``, which has already matched
    ``build.family`` and dispatched custom or unavailable replay elsewhere, so
    the descriptor here always carries ``replay_cls`` and a transformer class.
    ``materialize_weights=False`` loads the transformer as a meta-parameter
    skeleton for the training strategy to fill from its primary rank.
    """

    from vrl.utils.config import import_from_path

    build.require_replay()
    family_build = entry.family_build
    replay_cls = import_from_path(family_build.replay_cls)
    logger.info(
        "Building %s replay runtime bundle (registry descriptor) from %s",
        entry.family,
        build.model_name_or_path,
    )
    model = replay_cls(
        transformer=load_diffusers_transformer(
            build,
            family_build.transformer_classname,
            materialize_weights=materialize_weights,
        ),
        scheduler=(
            load_diffusers_scheduler(build, family_build.scheduler_classname)
            if family_build.scheduler_classname is not None
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
]
