"""Shared PEFT installation for token-family language trunks."""

from __future__ import annotations

from typing import Any


def install_token_lora_adapter(
    base: Any,
    config: Any,
    *,
    task_type: str | None = None,
) -> Any:
    """Install a fresh or warm-started trainable PEFT adapter on ``base``."""

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as error:  # pragma: no cover
        raise ImportError("PEFT is required for use_lora=True. pip install peft>=0.12") from error

    if config.lora_path:
        try:
            wrapped = PeftModel.from_pretrained(
                base,
                config.lora_path,
                is_trainable=True,
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
