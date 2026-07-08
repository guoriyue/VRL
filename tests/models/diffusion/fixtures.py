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


def build_tiny_anima_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``CosmosTransformer3DModel`` with Anima's channel geometry.

    Anima reuses the Cosmos Predict2 Text2Image backbone but, unlike predict2/2.5,
    feeds the latents to the transformer DIRECTLY (no runner-side condition-mask
    channel expansion — see ``AnimaModel.forward_step``), so ``in_channels`` equals
    ``out_channels`` (the bare latent channels) rather than latent + 1. It still
    sets ``concat_padding_mask=True`` (the real Anima config does), so the model
    appends a resized padding-mask channel internally; the wrapper therefore must
    pass a ``[1, 1, H, W]`` ``padding_mask``. This asymmetry — in==out at the
    wrapper boundary, with the mask channel added inside the model — is exactly
    what a hand-written ``torch.ones_like(hidden_states)`` fake cannot exercise.
    """

    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(seed)
    return CosmosTransformer3DModel(
        in_channels=TINY_COSMOS_LATENT_SHAPE[1],
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


# Tiny real FLUX geometry (CPU): PACKED latents [B, seq, C*4] with C=4 -> 16
# in_channels (patch_size=1 in packed token space). axes_dims_rope must sum to
# attention_head_dim (8). guidance_embeds=True mirrors FLUX.1-dev.
TINY_FLUX_IN_CHANNELS = 16
TINY_FLUX_JOINT_DIM = 16
TINY_FLUX_POOLED_DIM = 16


def build_tiny_flux_transformer(*, seed: int = 0, guidance_embeds: bool = True) -> Any:
    """Tiny real ``FluxTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import FluxTransformer2DModel

    torch.manual_seed(seed)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=TINY_FLUX_IN_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=TINY_FLUX_JOINT_DIM,
        pooled_projection_dim=TINY_FLUX_POOLED_DIM,
        guidance_embeds=guidance_embeds,
        axes_dims_rope=(2, 2, 4),
    )


# Tiny real Qwen-Image geometry (CPU): PACKED latents [B, seq, C*4] with C=4 ->
# 16 in_channels. out_channels(4) * patch_size**2(4) == in_channels(16) so the
# noise_pred matches the packed latent for the SDE step. axes_dims_rope sums to
# attention_head_dim (16).
TINY_SANA_LATENT_SHAPE = (2, 4, 8, 8)
TINY_SANA_CAPTION_DIM = 16


def build_tiny_sana_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``SanaTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import SanaTransformer2DModel

    torch.manual_seed(seed)
    return SanaTransformer2DModel(
        in_channels=TINY_SANA_LATENT_SHAPE[1],
        out_channels=TINY_SANA_LATENT_SHAPE[1],
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,
        cross_attention_dim=16,
        caption_channels=TINY_SANA_CAPTION_DIM,
        sample_size=TINY_SANA_LATENT_SHAPE[2],
        patch_size=1,
    )


TINY_LUMINA2_LATENT_SHAPE = (2, 4, 8, 8)
TINY_LUMINA2_CAP_DIM = 16


def build_tiny_lumina2_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``Lumina2Transformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import Lumina2Transformer2DModel

    torch.manual_seed(seed)
    return Lumina2Transformer2DModel(
        sample_size=TINY_LUMINA2_LATENT_SHAPE[2],
        patch_size=2,
        in_channels=TINY_LUMINA2_LATENT_SHAPE[1],
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=2,
        multiple_of=16,
        axes_dim_rope=(4, 2, 2),
        cap_feat_dim=TINY_LUMINA2_CAP_DIM,
    )


TINY_HUNYUAN_VIDEO_LATENT_SHAPE = (2, 4, 3, 8, 8)
TINY_HUNYUAN_VIDEO_TEXT_DIM = 16
TINY_HUNYUAN_VIDEO_POOLED_DIM = 8


def build_tiny_hunyuan_video_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``HunyuanVideoTransformer3DModel`` on CPU, cache-free."""

    from diffusers import HunyuanVideoTransformer3DModel

    torch.manual_seed(seed)
    return HunyuanVideoTransformer3DModel(
        in_channels=TINY_HUNYUAN_VIDEO_LATENT_SHAPE[1],
        out_channels=TINY_HUNYUAN_VIDEO_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        patch_size=2,
        patch_size_t=1,
        guidance_embeds=True,
        text_embed_dim=TINY_HUNYUAN_VIDEO_TEXT_DIM,
        pooled_projection_dim=TINY_HUNYUAN_VIDEO_POOLED_DIM,
        rope_axes_dim=(2, 4, 2),
    )


TINY_QWEN_IN_CHANNELS = 16
TINY_QWEN_JOINT_DIM = 16


def build_tiny_qwen_image_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``QwenImageTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(seed)
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=TINY_QWEN_IN_CHANNELS,
        out_channels=TINY_QWEN_IN_CHANNELS // 4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=TINY_QWEN_JOINT_DIM,
        guidance_embeds=False,
        axes_dims_rope=(8, 4, 4),
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


def record_forward_calls(module: torch.nn.Module) -> list[dict[str, Any]]:
    """Capture the kwargs of every forward call on ``module``.

    Registers a forward pre-hook and returns the list it appends to, so a test can
    assert exactly how a wrapper invoked a real transformer (call count + kwargs)
    against the genuine signature — instead of a hand-written fake that re-declares
    that signature and silently rots when the real model changes it.
    """

    calls: list[dict[str, Any]] = []
    module.register_forward_pre_hook(
        lambda _m, _args, kwargs: calls.append(dict(kwargs)),
        with_kwargs=True,
    )
    return calls
