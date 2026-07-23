"""Lightweight public config for the Janus-Pro model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata

# Default latent-grid length for Janus-Pro image generation.
JANUS_IMAGE_TOKEN_NUM = 576  # 24 x 24 latent grid per image


class JanusProModelSection(ModelSection):
    """Janus-Pro optional model-wrapper keys."""

    trust_remote_code: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    vq_latent_channels: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value"),
    )


@dataclass(slots=True)
class JanusProConfig:
    """Hyper-parameters for the Janus-Pro wrapper.

    The defaults target Janus-Pro-1B (single-H100 RL feasible).
    """

    model_path: str = "deepseek-ai/Janus-Pro-1B"
    revision: str | None = None
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"

    # LoRA
    use_lora: bool = True
    lora_path: str | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # Janus' language-model uses LLaMA-style projection names.
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    lora_init: str | bool = "gaussian"  # PEFT ``init_lora_weights``

    # Generation defaults — used by the AR runtime runner.
    guidance_scale: float = 5.0
    temperature: float = 1.0
    image_token_num: int = JANUS_IMAGE_TOKEN_NUM
    r1_refine_mode: str = "selfcheck"  # "selfcheck" | "always" | "never"

    # Misc
    trust_remote_code: bool = True
    device: str = "cuda"

    # VQ decoder shape — None ⇒ auto-detect from ``vq_model.config``.
    # Janus-Pro-1B uses 8 latent channels; Janus-Pro-7B may differ.
    # Override only if auto-detect fails.
    vq_latent_channels: int | None = None


__all__ = [
    "JANUS_IMAGE_TOKEN_NUM",
    "JanusProConfig",
    "JanusProModelSection",
]
