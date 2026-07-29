"""Pin the diffusers attention-processor default that SD3.5 rollout/replay rides on.

VRL used to overwrite every SD3 attention block with its own byte-equivalent copy
of ``JointAttnProcessor2_0``. That copy is gone, so the attention numerics now
belong to the pinned diffusers version. This is the dependency contract that
replaces it: if a diffusers bump changes the default processor for
``SD3Transformer2DModel``, CI fails here and forces an explicit numerics
re-evaluation instead of silently swapping the attention backend underneath the
rollout/replay SDE log-prob match.
"""

from __future__ import annotations

from diffusers.models.attention_processor import JointAttnProcessor2_0
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel


def _tiny_transformer() -> SD3Transformer2DModel:
    """Build a CPU-sized SD3 transformer from config only (no weight download)."""

    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        out_channels=4,
        pos_embed_max_size=16,
        # SD3.5-Medium shape: dual-attention blocks plus qk rms_norm, so the
        # contract covers the ``attn2`` processors too.
        dual_attention_layers=(0,),
        qk_norm="rms_norm",
    )


def test_sd3_transformer_exposes_attention_processors() -> None:
    """Checks the SD3 transformer exposes a non-empty attention-processor map."""
    processors = _tiny_transformer().attn_processors

    assert processors


def test_sd3_transformer_defaults_every_block_to_joint_attn_processor() -> None:
    """Checks every stock SD3 attention block uses diffusers JointAttnProcessor2_0."""
    processors = _tiny_transformer().attn_processors

    unexpected = {
        name: type(processor).__name__
        for name, processor in processors.items()
        if type(processor) is not JointAttnProcessor2_0
    }

    assert not unexpected, (
        "diffusers changed the default SD3 attention processor; re-evaluate "
        f"rollout/replay numerics before accepting the bump: {unexpected}"
    )
