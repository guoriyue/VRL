"""Shared model loaders: load diffusers transformer/scheduler pieces and
prepare the transformer (LoRA / full fine-tune / compile) for family runtimes."""

from __future__ import annotations

from typing import Any


def model_revision_kwargs(build: Any) -> dict[str, str]:
    """Return the immutable model snapshot argument for every upstream loader."""

    return model_config_revision_kwargs(build, "revision")


def model_config_revision_kwargs(build: Any, field: str) -> dict[str, str]:
    """Return an optional revision kwarg owned by one model-config repository."""

    revision = (getattr(build, "model_config", None) or {}).get(field)
    return {"revision": str(revision)} if revision else {}


def load_diffusers_transformer(
    build: Any,
    class_name: str,
    *,
    subfolder: str = "transformer",
) -> Any:
    """Load only a diffusers transformer component from a model repository."""

    import diffusers

    load_kwargs = model_revision_kwargs(build)
    transformer_cls = getattr(diffusers, class_name)
    return transformer_cls.from_pretrained(
        build.model_name_or_path,
        subfolder=subfolder,
        torch_dtype=build.parameter_dtype,
        **load_kwargs,
    )


def load_diffusers_scheduler(
    build: Any,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    import diffusers

    load_kwargs = model_revision_kwargs(build)
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
        scheduler.set_timesteps(int(num_steps), device=getattr(build, "device", None))
    return scheduler


def load_flow_match_scheduler(
    build: Any,
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
        rebuilt.set_timesteps(int(num_steps), device=getattr(build, "device", None))
    return rebuilt


def validate_rollout_quantization_support(build: Any) -> None:
    """Fail before model mutation when the requested rollout format cannot run."""

    rollout = getattr(build, "rollout", None)
    format_name = getattr(rollout, "quantization_format", None)
    if format_name != "nvfp4":
        return
    from vrl.nn.quantization import nvfp4_available

    if not nvfp4_available(getattr(build, "device", None)):
        raise RuntimeError(
            "precision.rollout.quantization.format='nvfp4' requires an "
            "NVFP4-capable CUDA target (Blackwell-class, compute capability "
            f">= 10.0); got {getattr(build, 'device', None)!r}",
        )


def apply_rollout_quantization(model: Any, build: Any) -> int:
    """Swap the rollout transformer's big GEMMs to the configured low-precision scheme.

    Reads ``build.rollout.quantization_format`` and dispatches to its module
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
    format_name = getattr(rollout, "quantization_format", None)
    if not format_name:  # bf16/fp16/fp32 rollout — no selective GEMM swap
        return 0
    validate_rollout_quantization_support(build)
    recipe = getattr(rollout, "quantization_recipe", None)
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
    if format_name == "fp8":
        swapped = model.quantize_rollout_fp8(recipe=recipe or "rowwise")
        policy_detail = f"recipe={recipe or 'rowwise'}, profile=attention_mlp"
    elif format_name == "nvfp4":
        swapped = model.quantize_rollout_nvfp4()
        policy_detail = "profile=mlp_only"
    else:
        raise NotImplementedError(
            f"precision.rollout.quantization.format={format_name!r} has no rollout "
            "swap yet (supported: fp8, nvfp4); add a "
            "quantize_rollout_* method + dispatch branch for the new scheme.",
        )
    if not swapped:
        target = "MLP" if format_name == "nvfp4" else "attention/MLP"
        raise RuntimeError(
            f"precision.rollout.quantization.format={format_name!r} but the swap "
            "matched 0 linears — the "
            f"policy has no quantizable {target} linears (check the exclude "
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


def assert_rollout_quantization_applied(model: Any, build: Any) -> None:
    """Backstop guard: a rollout quantization was requested but no quantized module.

    Family- and scheme-agnostic — called once at rollout-worker policy load, after
    the family builder ran. If the builder forgot to apply the swap (e.g. a newly
    added family), the model would silently run the base dtype despite
    ``precision.rollout.quantization`` asking for fp8/nvfp4/etc.; this turns that
    into a loud startup failure instead of a fake knob. Scheme identity lives on
    ``QuantizedLinear.quantization_scheme``: an FP8 module cannot satisfy an
    NVFP4 request merely because both share the same base class.
    """

    rollout = getattr(build, "rollout", None)
    format_name = getattr(rollout, "quantization_format", None)
    if not format_name:  # None / bf16 / fp16 / fp32 rollout — nothing to verify
        return
    from vrl.nn.quantization import QuantizedLinear

    # Diffusion wrappers expose ``transformer``; AR wrappers expose
    # ``language_model``. Fall back to the model root for future families while
    # preferring the policy core so an unrelated quantized auxiliary cannot make
    # the guard pass accidentally.
    policy_core = getattr(model, "transformer", None)
    if policy_core is None:
        policy_core = getattr(model, "language_model", None)
    if policy_core is None:
        policy_core = model
    policy_core = getattr(policy_core, "_orig_mod", policy_core)  # unwrap torch.compile
    modules = getattr(policy_core, "modules", None)
    count = (
        sum(
            1
            for module in modules()
            if isinstance(module, QuantizedLinear) and module.quantization_scheme == format_name
        )
        if callable(modules)
        else 0
    )
    if count == 0:
        raise RuntimeError(
            "precision.rollout.quantization.format="
            f"{format_name!r} requested but the rollout policy core has "
            f"0 {format_name} quantized linear modules "
            f"(family={getattr(build, 'family', None)!r}). "
            "The family runtime builder did not apply the rollout quantization swap "
            f"(e.g. apply_rollout_quantization), so {format_name} would silently run at "
            "the rollout base dtype. Wire "
            "the swap into that builder (after LoRA/full-finetune, before compile).",
        )


__all__ = [
    "apply_rollout_quantization",
    "assert_rollout_quantization_applied",
    "load_diffusers_scheduler",
    "load_diffusers_transformer",
    "load_flow_match_scheduler",
    "model_config_revision_kwargs",
    "model_revision_kwargs",
    "validate_rollout_quantization_support",
]
