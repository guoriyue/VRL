"""Lightweight runtime config for the GLM-Image model family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GlmImageConfig:
    """Hyper-parameters for the GLM-Image wrapper (defaults target the 9B AR)."""

    model_path: str = "zai-org/GLM-Image"
    revision: str | None = None
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"

    # LoRA — GLM-Image's text trunk uses LLaMA-style projection names.
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
    lora_init: str | bool = "gaussian"  # PEFT ``init_lora_weights``

    # AR sampling defaults — the checkpoint's generation_config.json values
    # (do_sample=True, temperature=0.9, top_p=0.75). There is NO AR-side CFG.
    temperature: float = 0.9
    top_p: float = 0.75
    # Target image size in pixels; must be a multiple of 32. The processor
    # derives the AR token grids (large + preview) from it.
    image_height: int = 1024
    image_width: int = 1024

    # Frozen DiT decode segment knobs (rollout postprocess only, never
    # trained). The pipeline default is 50 steps / 1.5 guidance; 20 steps is
    # the probe-speed default, override via sampling config for quality runs.
    decode_num_inference_steps: int = 20
    decode_guidance_scale: float = 1.5
    # Move the 9B AR model to CPU while the ~15GB DiT+VAE+ByT5 stack runs on
    # GPU (18G + 15G does not fit a 32GB card). Set False on >=48GB cards to
    # skip the transfers.
    decode_offload_ar: bool = True

    # Misc
    device: str = "cuda"


__all__ = ["GlmImageConfig"]
