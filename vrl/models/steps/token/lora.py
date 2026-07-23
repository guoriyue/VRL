"""Shared PEFT installation for token-family language trunks."""

from __future__ import annotations

from typing import Any

from vrl.models.peft_adapter import load_trainable_lora_adapter


def install_token_lora_adapter(
    base: Any,
    config: Any,
    *,
    task_type: str | None = None,
) -> Any:
    """Install a fresh or warm-started trainable PEFT adapter on ``base``."""

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover
        raise ImportError("PEFT is required for use_lora=True. pip install peft>=0.12") from error

    if config.lora_path:
        try:
            wrapped = load_trainable_lora_adapter(
                base,
                config.lora_path,
                expected_rank=config.lora_rank,
                expected_alpha=config.lora_alpha,
                expected_dropout=config.lora_dropout,
                expected_target_modules=config.lora_target_modules,
                expected_task_type=task_type,
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to load trainable token LoRA adapter from {config.lora_path!r}",
            ) from error
        return wrapped

    lora_kwargs: dict[str, Any] = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "init_lora_weights": config.lora_init,
        "target_modules": list(config.lora_target_modules),
        "bias": "none",
    }
    if task_type is not None:
        lora_kwargs["task_type"] = task_type
    return get_peft_model(base, LoraConfig(**lora_kwargs))


__all__ = ["install_token_lora_adapter"]
