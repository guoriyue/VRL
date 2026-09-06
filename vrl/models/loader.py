"""Shared model loaders: load diffusers transformer/scheduler pieces and
prepare the transformer (LoRA / full fine-tune / compile) for family runtimes."""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild


def load_diffusers_transformer(
    build: ModelBuild,
    class_name: str,
    *,
    subfolder: str = "transformer",
) -> Any:
    """Load only a diffusers transformer component from a model repository."""

    import diffusers

    load_kwargs = build.pretrained_kwargs
    transformer_cls = getattr(diffusers, class_name)
    return transformer_cls.from_pretrained(
        build.model_name_or_path,
        subfolder=subfolder,
        torch_dtype=build.parameter_dtype,
        **load_kwargs,
    )


def load_diffusers_scheduler(
    build: ModelBuild,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    import diffusers

    load_kwargs = build.pretrained_kwargs
    scheduler_cls = getattr(diffusers, class_name)
    scheduler = scheduler_cls.from_pretrained(
        build.model_name_or_path,
        subfolder=subfolder,
        **load_kwargs,
    )
    num_steps = build.num_steps
    # Dynamic-shifting schedulers (e.g. FLUX's FlowMatchEulerDiscreteScheduler)
    # derive their timestep/sigma schedule from a resolution-dependent ``mu`` that
    # is unknown here. Eager-setting without ``mu`` raises; defer to the family,
    # which sets the dynamic timesteps once the resolution is known (rollout:
    # prepare_sampling; replay: build_*_replay_runtime_bundle). Only eager-set the
    # static schedules (SD3.5 / Wan), whose sigmas depend solely on num_steps.
    if num_steps is not None and not getattr(scheduler.config, "use_dynamic_shifting", False):
        scheduler.set_timesteps(int(num_steps), device=build.device)
    return scheduler


def load_flow_match_scheduler(
    build: ModelBuild,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load the lightweight FlowMatch scheduler needed for replay log-prob math."""

    scheduler = load_diffusers_scheduler(
        build,
        "FlowMatchEulerDiscreteScheduler",
        subfolder=subfolder,
    )
    flow_shift = getattr(scheduler.config, "flow_shift", None)
    if flow_shift is None:
        return scheduler
    # SANA stores the rectified-flow shift under the DPM-facing ``flow_shift``
    # name. Rebuild the lightweight replay scheduler with the same value used by
    # the rollout conversion instead of FlowMatch's unrelated default shift=1.
    scheduler_cls = type(scheduler)
    rebuilt = scheduler_cls.from_config(dict(scheduler.config), shift=float(flow_shift))
    num_steps = build.num_steps
    if num_steps is not None:
        rebuilt.set_timesteps(int(num_steps), device=build.device)
    return rebuilt


def validate_rollout_quantization_support(build: ModelBuild) -> None:
    """Fail before model mutation when the requested rollout format cannot run."""

    quantization = build.precision.quantization
    if quantization is None or quantization.format != "nvfp4":
        return
    from vrl.nn.quantization import nvfp4_available

    if not nvfp4_available(build.device):
        raise RuntimeError(
            "precision.rollout.quantization.format='nvfp4' requires an "
            "NVFP4-capable CUDA target (Blackwell-class, compute capability "
            f">= 10.0); got {build.device!r}",
        )


def apply_rollout_quantization(model: Any, build: Any) -> int:
    """Swap the rollout transformer's big GEMMs to the configured low-precision scheme.

    Reads ``build.precision.quantization`` and dispatches to its module
    swap. FP8 targets eligible attention projections and MLPs; NVFP4 keeps the
    validated MLP-only target profile. Quantization is rollout-only and layered
    on the ordinary rollout dtype, which remains responsible for unswapped
    operations and parameter masters.

    Hardware/format checks happen before the model's quantization method is
    called, and a zero-match result fails loudly. Call after LoRA attachment and
    before compile so PEFT can see plain linears and inductor can see the final
    quantized modules.
    """

    import logging

    rollout = getattr(build, "rollout", None)
    quantization = getattr(getattr(build, "precision", None), "quantization", None)
    if quantization is None:  # bf16/fp16/fp32 rollout — no selective GEMM swap
        return 0
    format_name = quantization.format
    validate_rollout_quantization_support(build)
    recipe = quantization.recipe
    # blockwise delegates to vLLM's triton kernel, whose wrapper dynamo cannot
    # trace (lru_cache'd deep_gemm check + ctypes pynvml call): measured 45 graph
    # breaks on SD3.5 and a compiled forward ~10x SLOWER than eager
    # (SPRINT_rollout_optimization_layer item 2). Refuse the combination instead
    # of silently shipping the regression.
    if (
        format_name == "fp8"
        and recipe == "blockwise"
        and getattr(
            build,
            "torch_compile",
            None,
        )
    ):
        raise ValueError(
            "precision.rollout.quantization.recipe='blockwise' is incompatible with "
            "model.torch_compile (the vLLM block kernel graph-breaks inductor; the "
            "compiled forward is ~10x slower than eager). Use recipe='rowwise' "
            "(compile-clean) or disable model.torch_compile.",
        )
    from vrl.nn.quantization import QUANTIZATION_SCHEMES

    scheme = QUANTIZATION_SCHEMES.get(format_name)
    if scheme is None:
        supported = ", ".join(sorted(QUANTIZATION_SCHEMES))
        raise NotImplementedError(
            f"precision.rollout.quantization.format={format_name!r} has no rollout "
            f"swap yet (supported: {supported}); add a QuantizedLinear subclass and "
            "register it in vrl.nn.quantization.QUANTIZATION_SCHEMES.",
        )
    # The scheme owns its target scope and kernel; the model declares only which
    # roots to walk and what to exclude. Neither knows about the other, so a new
    # scheme needs no model change and a new family needs no scheme change.
    # ``QuantizationPolicy`` already normalized the recipe and filled the format's
    # default, so its presence — not a second per-scheme table — decides whether
    # this scheme takes one.
    swap_kwargs: dict[str, Any] = {"exclude": model.quantization_exclude}
    if recipe is not None:
        swap_kwargs["recipe"] = recipe
        policy_detail = f"recipe={recipe}, profile={scheme.default_target_profile}"
    else:
        policy_detail = f"profile={scheme.default_target_profile}"
    # Paths are prefixed by root so the log names WHICH expert changed on a
    # multi-root family.
    swapped = [
        f"{root_name}.{path}"
        for root_name, root in model.policy_cores.items()
        for path in scheme.swap_linears(root, **swap_kwargs)
    ]
    if not swapped:
        raise RuntimeError(
            f"precision.rollout.quantization.format={format_name!r} but the swap "
            "matched 0 linears — the policy has no quantizable "
            f"{scheme.default_target_profile} linears (check the exclude "
            "list / min_features). It would be a no-op.",
        )
    if not getattr(rollout, "base_weight_sync", True):
        # Base weights will never be synced into this rollout (LoRA syncs
        # adapters; sync-free contexts sync nothing), so the source masters are
        # dead weight — drop them BEFORE the device move so a large quantized
        # rollout does not retain a full source-dtype copy beside its cache.
        from vrl.nn.quantization import drop_quantized_masters

        freed = drop_quantized_masters(model)
        logging.getLogger(__name__).info(
            "%s rollout without base-weight sync: dropped source masters (%.1f GiB freed)",
            format_name,
            freed / 2**30,
        )
    logging.getLogger(__name__).info(
        "%s rollout (%s): quantized %d policy linears",
        format_name,
        policy_detail,
        len(swapped),
    )
    return len(swapped)


__all__ = [
    "apply_rollout_quantization",
    "load_diffusers_scheduler",
    "load_diffusers_transformer",
    "load_flow_match_scheduler",
    "validate_rollout_quantization_support",
]
