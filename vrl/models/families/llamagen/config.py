"""Lightweight public config for the LlamaGen model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.config.model_schema import ModelSection

# Defaults for the released t2i_XL_stage1_256 checkpoint.
LLAMAGEN_IMAGE_TOKEN_NUM = 256  # 16 x 16 latent grid per 256 px image
LLAMAGEN_DOWNSAMPLE_SIZE = 16  # VQ-16 spatial downsample factor
LLAMAGEN_CAPTION_TOKEN_NUM = 120  # fixed T5 caption prefix length
LLAMAGEN_CAPTION_DIM = 2048  # flan-t5-xl d_model
LLAMAGEN_HF_REPO = "peizesun/llamagen_t2i"
LLAMAGEN_GPT_CKPT = "t2i_XL_stage1_256.pt"
LLAMAGEN_VQ_CKPT = "vq_ds16_t2i.pt"
LLAMAGEN_T5_PATH = "google/flan-t5-xl"


class LlamaGenModelSection(ModelSection):
    """LlamaGen checkpoint-file and frozen-T5 keys."""

    gpt_ckpt: Any = None
    gpt_model: Any = None
    t5_path: Any = None
    t5_revision: Any = None
    vq_ckpt: Any = None


@dataclass(slots=True)
class LlamaGenConfig:
    """Hyper-parameters for the LlamaGen wrapper.

    Defaults target ``t2i_XL_stage1_256.pt`` (775M GPT-XL, 256 px stage1).
    """

    model_path: str = LLAMAGEN_HF_REPO
    revision: str | None = None
    gpt_ckpt: str = LLAMAGEN_GPT_CKPT
    vq_ckpt: str = LLAMAGEN_VQ_CKPT
    gpt_model: str = "GPT-XL"
    t5_path: str = LLAMAGEN_T5_PATH
    t5_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"

    # LoRA — attaches to the vendored GPT's fused attention projections.
    use_lora: bool = True
    lora_path: str | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("wqkv", "wo")
    lora_init: str | bool = "gaussian"

    # Generation defaults — used by the AR runtime runner. Upstream demo uses
    # cfg_scale=7.5, top_k=1000; top_k defaults to 0 (off) here so the rollout
    # behavior distribution stays closest to the conditional policy GRPO scores.
    guidance_scale: float = 7.5
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    image_token_num: int = LLAMAGEN_IMAGE_TOKEN_NUM
    cls_token_num: int = LLAMAGEN_CAPTION_TOKEN_NUM
    caption_dim: int = LLAMAGEN_CAPTION_DIM
    downsample_size: int = LLAMAGEN_DOWNSAMPLE_SIZE
    codebook_embed_dim: int = 8


__all__ = [
    "LLAMAGEN_CAPTION_DIM",
    "LLAMAGEN_CAPTION_TOKEN_NUM",
    "LLAMAGEN_DOWNSAMPLE_SIZE",
    "LLAMAGEN_GPT_CKPT",
    "LLAMAGEN_HF_REPO",
    "LLAMAGEN_IMAGE_TOKEN_NUM",
    "LLAMAGEN_T5_PATH",
    "LLAMAGEN_VQ_CKPT",
    "LlamaGenConfig",
    "LlamaGenModelSection",
]
