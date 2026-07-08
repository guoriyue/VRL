"""Tiny-real GLM-Image fixtures built straight from transformers configs.

GLM-Image ships inside ``transformers`` (>= 5.13), so these helpers construct
GENUINE ``GlmImageForConditionalGeneration`` / replay-core modules from tiny
configs — no checkpoint download, CPU-only, float32.

The tiny vocabulary layout mirrors the real checkpoint's structure at 1/1000
scale (real: codebook 16384, vision vocab 16512, text vocab 168064):

  * ids 0..15   — the VQ codebook (``vq_config.num_embeddings`` = 16); these
    double as text-vocab embedding rows, exactly like the real checkpoint
  * id 16       — image_start (= codebook size, like the real 16384)
  * id 17       — image_end / EOS
  * ids 18..23  — reserved vision-vocab tail (vision_vocab_size = 24)
  * ids 24..63  — "text" ids (text vocab_size = 64; image_token_id = 63)

so the generation vocab is the 16-wide codebook prefix of the 24-wide lm_head.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.models.ar.glm_image.model import (
    GlmImageConfig,
    GlmImageModel,
    GlmImageReplayCore,
    GlmImageReplayModel,
)

TINY_CODEBOOK = 16
TINY_VISION_VOCAB = 24
TINY_TEXT_VOCAB = 64
TINY_HIDDEN = 16
TINY_IMAGE_START_ID = 16
TINY_IMAGE_END_ID = 17


def tiny_hf_glm_image_config():
    """Build a tiny (~40k param) transformers GlmImageConfig."""

    from transformers.models.glm_image.configuration_glm_image import (
        GlmImageConfig as HFGlmImageConfig,
    )
    from transformers.models.glm_image.configuration_glm_image import (
        GlmImageTextConfig,
        GlmImageVisionConfig,
        GlmImageVQVAEConfig,
    )

    text_config = GlmImageTextConfig(
        vocab_size=TINY_TEXT_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        pad_token_id=0,
        vision_vocab_size=TINY_VISION_VOCAB,
        eos_token_id=TINY_IMAGE_END_ID,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            # head_dim = 16/2 = 8 -> 4 inv-freq pairs; sections sum to 4.
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 1.0,
        },
    )
    vision_config = GlmImageVisionConfig(
        depth=1,
        hidden_size=32,
        num_heads=2,
        intermediate_size=32,
        image_size=32,
        patch_size=16,
    )
    vq_config = GlmImageVQVAEConfig(
        embed_dim=8,
        num_embeddings=TINY_CODEBOOK,
        latent_channels=32,
    )
    return HFGlmImageConfig(
        text_config=text_config,
        vision_config=vision_config,
        vq_config=vq_config,
        image_token_id=TINY_TEXT_VOCAB - 1,
        image_start_token_id=TINY_IMAGE_START_ID,
        image_end_token_id=TINY_IMAGE_END_ID,
    )


def _stub_processor() -> SimpleNamespace:
    """Minimal processor surface: only ``tokenizer`` id attrs are read on the
    non-encode paths these tests exercise (real prompt encoding needs the
    checkpoint tokenizer/chat-template files, which tiny tests never
    download)."""

    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            pad_token_id=TINY_IMAGE_END_ID,
            eos_token_id=TINY_IMAGE_END_ID,
            padding_side="left",
        ),
    )


def build_tiny_glm_image_model(*, use_lora: bool = False) -> GlmImageModel:
    from transformers.models.glm_image.modeling_glm_image import (
        GlmImageForConditionalGeneration,
    )

    torch.manual_seed(0)
    hf_model = GlmImageForConditionalGeneration(tiny_hf_glm_image_config())
    config = GlmImageConfig(
        model_path="tiny-glm-image",
        dtype="float32",
        device="cpu",
        use_lora=use_lora,
    )
    return GlmImageModel(config, glm=hf_model, processor=_stub_processor())


def build_tiny_glm_image_replay_model(*, use_lora: bool = False) -> GlmImageReplayModel:
    torch.manual_seed(0)
    core = GlmImageReplayCore(tiny_hf_glm_image_config())
    config = GlmImageConfig(
        model_path="tiny-glm-image",
        dtype="float32",
        device="cpu",
        use_lora=use_lora,
    )
    return GlmImageReplayModel(config, glm=core)


__all__ = [
    "TINY_CODEBOOK",
    "TINY_HIDDEN",
    "TINY_IMAGE_END_ID",
    "TINY_IMAGE_START_ID",
    "TINY_TEXT_VOCAB",
    "TINY_VISION_VOCAB",
    "build_tiny_glm_image_model",
    "build_tiny_glm_image_replay_model",
    "tiny_hf_glm_image_config",
]
