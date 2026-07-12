"""Shared model loaders: load diffusers transformer/scheduler pieces and
prepare the transformer (LoRA / full fine-tune / compile) for family runtimes."""

from __future__ import annotations

from typing import Any

from vrl.models.dtypes import resolve_torch_dtype


def load_diffusers_transformer(
    spec: Any,
    class_name: str,
    *,
    subfolder: str = "transformer",
) -> Any:
    """Load only a diffusers transformer component from a model repository."""

    import diffusers

    load_kwargs: dict[str, Any] = {}
    revision = (getattr(spec, "model_config", None) or {}).get("revision")
    if revision:
        load_kwargs["revision"] = revision
    transformer_cls = getattr(diffusers, class_name)
    return transformer_cls.from_pretrained(
        spec.model_name_or_path,
        subfolder=subfolder,
        torch_dtype=resolve_torch_dtype(spec.dtype),
        **load_kwargs,
    )


def load_diffusers_scheduler(
    spec: Any,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    import diffusers

    load_kwargs: dict[str, Any] = {}
    revision = (getattr(spec, "model_config", None) or {}).get("revision")
    if revision:
        load_kwargs["revision"] = revision
    scheduler_cls = getattr(diffusers, class_name)
    scheduler = scheduler_cls.from_pretrained(
        spec.model_name_or_path,
        subfolder=subfolder,
        **load_kwargs,
    )
    num_steps = spec.num_steps
    # Dynamic-shifting schedulers (e.g. FLUX's FlowMatchEulerDiscreteScheduler)
    # derive their timestep/sigma schedule from a resolution-dependent ``mu`` that
    # is unknown here. Eager-setting without ``mu`` raises; defer to the family,
    # which sets the dynamic timesteps once the resolution is known (rollout:
    # prepare_sampling; replay: build_*_replay_runtime_bundle). Only eager-set the
    # static schedules (SD3.5 / Wan), whose sigmas depend solely on num_steps.
    if num_steps is not None and not getattr(scheduler.config, "use_dynamic_shifting", False):
        scheduler.set_timesteps(int(num_steps), device=getattr(spec, "device", None))
    return scheduler


def load_flow_match_scheduler(
    spec: Any,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load the lightweight FlowMatch scheduler needed for replay log-prob math."""

    return load_diffusers_scheduler(
        spec,
        "FlowMatchEulerDiscreteScheduler",
        subfolder=subfolder,
    )


def apply_rollout_quantization(model: Any, spec: Any) -> int:
    """Swap the rollout policy's big GEMMs to the configured low-precision scheme.

    Reads the scheme off ``precision.rollout`` (``spec.rollout_quantization``) and
    dispatches to its swap — fp8 today, fp4/int8 as siblings. Quantization is
    rollout-only and is a *module swap*, not a storage dtype: a bf16/fp16/fp32
    rollout is a plain dtype set at model load, so it has nothing to apply here and
    returns 0. Raises when a scheme is requested but the swap matched no linear (a
    silent no-op would otherwise run the rollout in bf16). Call after LoRA/full-
    finetune and before compile so inductor sees the quantized modules.
    """

    import logging

    scheme = getattr(spec, "rollout_quantization", None)
    if not scheme:  # bf16/fp16/fp32 rollout — a load-time dtype, not a swap
        return 0
    recipe = getattr(spec, "rollout_quantization_recipe", None)
    if scheme == "fp8":
        effective_recipe = recipe or "rowwise"
    elif scheme == "fp4":
        if recipe not in (None, "nvfp4"):
            raise ValueError(
                f"precision.rollout_recipe={recipe!r} is not an fp4 recipe: fp4 has "
                "a single 'nvfp4' recipe (rowwise/tensorwise/blockwise are fp8).",
            )
        target_device = getattr(spec, "device", None)
        if target_device is not None:
            from vrl.nn.quantization.fp4 import nvfp4_available

            if not nvfp4_available(target_device):
                raise RuntimeError(
                    "precision.rollout='fp4' requires an NVFP4-capable CUDA target "
                    f"(Blackwell-class, compute capability >= 10.0); got {target_device!r}.",
                )
        effective_recipe = recipe or "nvfp4"
    else:
        raise NotImplementedError(
            f"precision.rollout={scheme!r} has no rollout swap yet (fp8/fp4 only); "
            "add a quantize_rollout_* method + dispatch branch for the new scheme.",
        )
    # blockwise delegates to vLLM's triton kernel, whose wrapper dynamo cannot
    # trace (lru_cache'd deep_gemm check + ctypes pynvml call): measured 45 graph
    # breaks on SD3.5 and a compiled forward ~10x SLOWER than eager
    # (SPRINT_rollout_optimization_layer item 2). Refuse the combination instead
    # of silently shipping the regression.
    compile_config = getattr(spec, "torch_compile", None)
    if effective_recipe == "blockwise" and compile_config:
        raise ValueError(
            "precision.rollout_recipe='blockwise' is incompatible with "
            "model.torch_compile (the vLLM block kernel graph-breaks inductor; the "
            "compiled forward is ~10x slower than eager). Use recipe='rowwise' "
            "(compile-clean) or disable model.torch_compile.",
        )
    if scheme == "fp8":
        swapped = model.quantize_rollout_fp8(recipe=effective_recipe)
    else:
        swapped = model.quantize_rollout_fp4()
    if not swapped:
        target = "MLP" if scheme == "fp4" else "attention/MLP"
        raise RuntimeError(
            f"precision.rollout={scheme!r} but the swap matched 0 linears — the "
            f"policy has no quantizable {target} linears (check the exclude "
            "list / min_features). It would be a no-op.",
        )
    syncs_base = getattr(spec, "rollout_weight_sync", True) and not getattr(
        spec,
        "use_lora",
        None,
    )
    if not syncs_base:
        # Base weights will never be synced into this rollout (LoRA syncs
        # adapters; sync-free contexts sync nothing), so the high-precision
        # masters are dead weight — drop them BEFORE the device move (a 17B
        # quantized rollout shrinks instead of retaining cache + master).
        from vrl.nn.quantization import drop_quantized_masters

        freed = drop_quantized_masters(model)
        logging.getLogger(__name__).info(
            "%s rollout without base-weight sync: dropped master weights (%.1f GiB freed)",
            scheme,
            freed / 2**30,
        )
    logging.getLogger(__name__).info(
        "%s rollout (recipe=%s, profile=%s): quantized %d policy linears",
        scheme,
        effective_recipe,
        "mlp_only" if scheme == "fp4" else "attention_mlp",
        len(swapped),
    )
    return len(swapped)


def assert_rollout_quantization_applied(model: Any, spec: Any) -> None:
    """Backstop guard: a rollout quantization was requested but no quantized module.

    Family- and scheme-agnostic — called once at rollout-worker policy load, after
    the family builder ran. If the builder forgot to apply the swap (e.g. a newly
    added family), or installed a different scheme, the model would silently run
    the wrong policy despite ``precision.rollout`` asking for fp8/fp4/etc. This
    turns that into a loud startup failure instead of a fake knob. Scheme identity
    lives on ``QuantizedLinear`` subclasses, so future schemes need no loader-side
    type table. Compiled modules expose their originals through normal traversal.
    """

    scheme = getattr(spec, "rollout_quantization", None)
    if not scheme:  # None / bf16 / fp16 / fp32 rollout — nothing to verify
        return
    from vrl.nn.quantization import QuantizedLinear

    # Most family models are nn.Module instances. A few runtime facades are plain
    # objects that own one or more modules, so scan every direct module attribute
    # instead of assuming diffusion's ``model.transformer`` shape. Compiled
    # modules register ``_orig_mod`` as a child, so normal module traversal also
    # reaches the underlying quantized linears.
    if hasattr(model, "modules"):
        module_roots = [model]
    else:
        module_roots = []
        for value in getattr(model, "__dict__", {}).values():
            candidate = getattr(value, "_orig_mod", value)
            if hasattr(candidate, "modules"):
                module_roots.append(candidate)
    count = sum(
        1
        for root in module_roots
        for module in root.modules()
        if isinstance(module, QuantizedLinear) and module.quantization_scheme == scheme
    )
    if count == 0:
        family = (
            getattr(spec, "family", None)
            or getattr(spec, "ar_task", None)
            or getattr(spec, "task_variant", None)
        )
        raise RuntimeError(
            f"precision.rollout={scheme!r} requested but the rollout model has "
            f"0 {scheme} quantized linear modules (family={family!r}). "
            "The family runtime builder did not apply the requested rollout quantization swap "
            f"(e.g. apply_rollout_quantization), so {scheme} would silently run in bf16. Wire "
            "the swap into that builder (after LoRA/full-finetune, before compile).",
        )


__all__ = [
    "apply_rollout_quantization",
    "assert_rollout_quantization_applied",
    "load_diffusers_scheduler",
    "load_diffusers_transformer",
    "load_flow_match_scheduler",
]
