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
    """Swap the rollout transformer's big GEMMs to the configured low-precision scheme.

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
    # blockwise delegates to vLLM's triton kernel, whose wrapper dynamo cannot
    # trace (lru_cache'd deep_gemm check + ctypes pynvml call): measured 45 graph
    # breaks on SD3.5 and a compiled forward ~10x SLOWER than eager
    # (SPRINT_rollout_optimization_layer item 2). Refuse the combination instead
    # of silently shipping the regression.
    if recipe == "blockwise" and getattr(spec, "torch_compile", None):
        raise ValueError(
            "precision.rollout_recipe='blockwise' is incompatible with "
            "model.torch_compile (the vLLM block kernel graph-breaks inductor; the "
            "compiled forward is ~10x slower than eager). Use recipe='rowwise' "
            "(compile-clean) or disable model.torch_compile.",
        )
    if scheme == "fp8":
        swapped = model.quantize_rollout_fp8(recipe=recipe or "rowwise")
    else:
        raise NotImplementedError(
            f"precision.rollout={scheme!r} has no rollout swap yet (only fp8); add a "
            "quantize_rollout_* method + dispatch branch for the new scheme.",
        )
    if not swapped:
        raise RuntimeError(
            f"precision.rollout={scheme!r} but the swap matched 0 linears — the "
            "transformer has no quantizable attention/MLP linears (check the exclude "
            "list / min_features). It would be a no-op.",
        )
    logging.getLogger(__name__).info(
        "%s rollout (recipe=%s): quantized %d transformer linears",
        scheme, recipe or "rowwise", len(swapped),
    )
    return len(swapped)


def assert_rollout_quantization_applied(model: Any, spec: Any) -> None:
    """Backstop guard: a rollout quantization was requested but no quantized module.

    Family- and scheme-agnostic — called once at rollout-worker policy load, after
    the family builder ran. If the builder forgot to apply the swap (e.g. a newly
    added family), the model would silently run bf16 despite ``precision.rollout``
    asking for fp8/fp4/etc.; this turns that into a loud startup failure instead of
    a fake knob. Counts *any* ``QuantizedLinear`` (fp8 today, fp4/int8 as they
    land) so new schemes are covered without editing this guard. Unwraps a
    ``torch.compile`` wrapper to count the real modules.
    """

    scheme = getattr(spec, "rollout_quantization", None)
    if not scheme:  # None / bf16 / fp16 / fp32 rollout — nothing to verify
        return
    from vrl.nn.quantization import QuantizedLinear

    transformer = getattr(model, "transformer", None)
    transformer = getattr(transformer, "_orig_mod", transformer)  # unwrap torch.compile
    count = (
        sum(1 for m in transformer.modules() if isinstance(m, QuantizedLinear))
        if transformer is not None
        else 0
    )
    if count == 0:
        raise RuntimeError(
            f"precision.rollout={scheme!r} requested but the rollout transformer has "
            f"0 quantized linear modules (family={getattr(spec, 'task_variant', None)!r}). "
            "The family runtime builder did not apply the rollout quantization swap "
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
