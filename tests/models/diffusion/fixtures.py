"""Tiny real (cache-free) diffusion model fixtures for CPU tests.

Each ``build_tiny_*`` returns a genuine diffusers transformer constructed straight
from config — no ``from_pretrained`` / no download / no cached weights — so the
source fully defines it, forward outputs are reproducible, and tests run real
inference on CPU. ``add_lora_adapters`` attaches real diffusers-native LoRA.
"""

from __future__ import annotations

from typing import Any

import torch

# Tiny real Wan DiT geometry (CPU, ~6.7K params): latent video [B, C, T, H, W]
# with patch size (1, 2, 2); text embeds are [B, TEXT_LEN, TEXT_DIM].
TINY_WAN_LATENT_SHAPE = (1, 4, 1, 4, 4)
TINY_WAN_TEXT_LEN = 3
TINY_WAN_TEXT_DIM = 16
_TINY_WAN_LORA_TARGETS = ["to_q", "to_v"]


def build_tiny_wan_transformer(*, seed: int = 0) -> Any:
    """A real ~6.7K-param ``WanTransformer3DModel`` on CPU, random-init from a seed.

    Built straight from config — no ``from_pretrained`` / no download / no cached
    weights — so the source fully defines it and forward outputs are reproducible.
    Use this instead of a hand-written fake when a test needs the genuine
    transformer's adapter API or real gradient flow (e.g. DiffusionNFT branches).
    """

    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=TINY_WAN_LATENT_SHAPE[1],
        out_channels=TINY_WAN_LATENT_SHAPE[1],
        text_dim=TINY_WAN_TEXT_DIM,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=64,
    )


TINY_COSMOS_LATENT_SHAPE = (2, 4, 1, 4, 4)
TINY_COSMOS_TEXT_DIM = 16


def build_tiny_cosmos_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``CosmosTransformer3DModel`` on CPU, cache-free (config-init).

    Cosmos predict2/2.5 concatenate a 1-channel condition mask into the latent
    channel axis, so ``in_channels`` is the latent channels (4) + 1; ``out_channels``
    stays at the latent channels. ``attention_head_dim`` is 16 (8 divides the 3D
    rope unevenly and trips a div-by-zero at construction).
    """

    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(seed)
    return CosmosTransformer3DModel(
        in_channels=TINY_COSMOS_LATENT_SHAPE[1] + 1,
        out_channels=TINY_COSMOS_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        mlp_ratio=2.0,
        text_embed_dim=TINY_COSMOS_TEXT_DIM,
        adaln_lora_dim=8,
        max_size=(4, 16, 16),
        patch_size=(1, 2, 2),
        concat_padding_mask=True,
    )


TINY_SD3_LATENT_SHAPE = (2, 4, 8, 8)
TINY_SD3_JOINT_DIM = 16
TINY_SD3_POOLED_DIM = 16


def build_tiny_sd3_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``SD3Transformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import SD3Transformer2DModel

    torch.manual_seed(seed)
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=TINY_SD3_LATENT_SHAPE[1],
        out_channels=TINY_SD3_LATENT_SHAPE[1],
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=TINY_SD3_JOINT_DIM,
        caption_projection_dim=16,
        pooled_projection_dim=TINY_SD3_POOLED_DIM,
        pos_embed_max_size=8,
    )


def build_tiny_wan_i2v_transformer(*, seed: int = 0) -> Any:
    """Tiny real Wan I2V ``WanTransformer3DModel`` on CPU, cache-free.

    I2V cats the conditioning latent into the channel axis, so ``in_channels`` is
    doubled (4 latent + 4 condition); ``image_dim`` enables the CLIP image-embed
    cross-attention branch. Same config-init/no-download contract as
    :func:`build_tiny_wan_transformer`.
    """

    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=2 * TINY_WAN_LATENT_SHAPE[1],
        out_channels=TINY_WAN_LATENT_SHAPE[1],
        text_dim=TINY_WAN_TEXT_DIM,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=64,
        image_dim=TINY_WAN_TEXT_DIM,
    )


def add_lora_adapters(
    transformer: Any,
    *,
    names: tuple[str, ...] = ("default", "previous"),
    rank: int = 4,
    seed: int = 0,
) -> Any:
    """Attach independently gaussian-init LoRA adapters via diffusers' native API.

    Uses ``PeftAdapterMixin.add_adapter`` (not ``get_peft_model``) so the model
    exposes the ``disable_adapters`` / ``enable_adapters`` / ``set_adapter`` surface
    DiffusionNFT drives. One shared RNG stream seeds all adapters, so they end up
    with distinct weights; the first name is left active.
    """

    from peft import LoraConfig

    torch.manual_seed(seed)
    transformer.requires_grad_(False)
    for name in names:
        transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                init_lora_weights="gaussian",
                target_modules=_TINY_WAN_LORA_TARGETS,
            ),
            adapter_name=name,
        )
    if names:
        transformer.set_adapter(names[0])
    return transformer
