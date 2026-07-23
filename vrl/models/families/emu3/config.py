"""Lightweight runtime config for the Emu3 model family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Emu3Config:
    """Hyper-parameters for the Emu3 wrapper (defaults target Emu3-Gen ~9B)."""

    model_path: str = "BAAI/Emu3-Gen-hf"
    revision: str | None = None
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"

    # LoRA — Emu3's text model uses LLaMA-style projection names.
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    lora_init: str = "gaussian"  # PEFT ``init_lora_weights``

    # Generation defaults — used by the AR runtime runner.
    guidance_scale: float = 3.0
    temperature: float = 1.0
    # Target pixel area for the generated image; the processor derives the
    # latent grid (height, width) from it: 262144 = 512x512 -> a 64x64 grid.
    image_area: int = 262_144
    ratio: str = "1:1"

    # Misc
    device: str = "cuda"


__all__ = ["Emu3Config"]
