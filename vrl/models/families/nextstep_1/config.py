"""Lightweight public config for the NextStep-1 model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.config.model_schema import ModelSection

# NextStep-1 image grid defaults for the f8ch16 VAE checkpoint.
NEXTSTEP_DEFAULT_TOKEN_NUM = 1024  # 32 x 32 patches per 256^2 image
NEXTSTEP_DEFAULT_TOKEN_DIM = 64  # latent_patch_size^2 * f8ch16 channels


class NextStep1ModelSection(ModelSection):
    """NextStep-1 tokenizer and frozen-module keys."""

    freeze_vae: Any = None
    vae_path: Any = None
    vae_revision: Any = None


@dataclass(slots=True)
class NextStep1Config:
    """Hyper-parameters for the NextStep-1 wrapper.

    Defaults target ``stepfun-ai/NextStep-1.1`` — the RL-post-trained
    14B variant — paired with the f8ch16 VAE tokenizer.
    """

    model_path: str = "stepfun-ai/NextStep-1.1"
    revision: str | None = None
    vae_path: str = "stepfun-ai/NextStep-1-f8ch16-Tokenizer"
    vae_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"

    # LoRA — applied to the LLM trunk (the 14B AR transformer)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # NextStep-1's LLM is Qwen-derived; same names as Qwen-2 attention
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    lora_init: str | bool = "gaussian"

    # Flow-head sampling — used by the AR runtime runner.
    num_steps: int = 20  # K Euler steps inside the flow ODE
    noise_level: float = 1.0  # final-step Gaussian std multiplier
    guidance_scale: float = 4.5  # CFG strength on the velocity field

    # AR loop
    image_token_num: int = NEXTSTEP_DEFAULT_TOKEN_NUM
    token_dim: int = NEXTSTEP_DEFAULT_TOKEN_DIM

    # Frozen sub-modules
    freeze_vae: bool = True

    # Memory
    gradient_checkpointing: bool = True


__all__ = [
    "NEXTSTEP_DEFAULT_TOKEN_DIM",
    "NEXTSTEP_DEFAULT_TOKEN_NUM",
    "NextStep1Config",
    "NextStep1ModelSection",
]
