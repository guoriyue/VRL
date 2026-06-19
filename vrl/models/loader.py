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
    if num_steps is not None:
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


def apply_lora_to_transformer(model: Any, spec: Any) -> None:
    """Attach or load a PEFT LoRA adapter on ``model.transformer``."""

    from peft import LoraConfig, PeftModel, get_peft_model

    transformer = model.transformer
    transformer.requires_grad_(False)
    to = getattr(transformer, "to", None)
    if callable(to):
        to(model.device, dtype=resolve_torch_dtype(spec.dtype))

    lora_path = spec.lora_path
    if lora_path:
        wrapped = PeftModel.from_pretrained(
            transformer,
            lora_path,
            is_trainable=True,
        )
        wrapped.set_adapter("default")
        model._set_transformer(wrapped)
        return

    lora_config = spec.lora
    if lora_config is None:
        raise ValueError("LoRA runtime spec requires lora_config when lora_path is empty")
    cfg = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        init_lora_weights=lora_config.get("init_lora_weights", "gaussian"),
        target_modules=lora_config["target_modules"],
    )
    model._set_transformer(get_peft_model(transformer, cfg))


def enable_transformer_full_finetune(model: Any) -> None:
    """Mark the replay transformer fully trainable."""

    model.transformer.requires_grad_(True)
    to = getattr(model.transformer, "to", None)
    if callable(to):
        to(model.device)


def compile_transformer(model: Any, mode: str) -> None:
    """Apply ``torch.compile`` to the replay transformer."""

    import torch

    model._set_transformer(
        torch.compile(model.transformer, mode=mode, fullgraph=False),
    )


def apply_rollout_fp8(model: Any, spec: Any) -> None:
    """Swap the rollout transformer's big GEMMs to fp8 when the spec requests it.

    fp8 is rollout-only (``precision.rollout=fp8``): the model loads a bf16 master
    and quantizes inside the GEMM (``Fp8Linear``). Call after LoRA/full-finetune
    and before compile so inductor sees the fp8 modules. No-op otherwise.
    """

    import logging

    if getattr(spec, "rollout_quantization", None) != "fp8":
        return
    swapped = model.quantize_transformer_fp8()
    logging.getLogger(__name__).info(
        "fp8 rollout: quantized %d transformer linears to e4m3", len(swapped),
    )


__all__ = [
    "apply_lora_to_transformer",
    "apply_rollout_fp8",
    "compile_transformer",
    "enable_transformer_full_finetune",
    "load_diffusers_scheduler",
    "load_diffusers_transformer",
    "load_flow_match_scheduler",
]
