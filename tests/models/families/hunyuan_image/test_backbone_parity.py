"""Real-inference tests for the HunyuanImage true-CFG dual-text-stream wrapper.

No fake transformer: a real (tiny, cache-free) ``HunyuanImageTransformer2DModel``
runs through ``forward_step``; the separate-branch CFG (cond/uncond at DIFFERENT
sequence lengths, both text streams threaded per branch) and the classic CFG
combine are verified against the model's OWN real outputs.

The tiny builder lives here (not in the shared fixtures module) because
hunyuan_image is the only consumer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tests.models.steps.denoise.fixtures import record_forward_calls, stamp_model_precision
from vrl.models.families.hunyuan_image.model import (
    HunyuanImageModel,
    HunyuanImageReplayModel,
    HunyuanImageSamplingState,
)

# Tiny real HunyuanImage geometry (CPU, ~4.3M params): 4D latents [B, C, H, W]
# with patch_size (1, 1); text stream 1 (Qwen) is [B, len, TEXT_DIM], glyph
# stream 2 (byT5) is [B, len2, TEXT_2_DIM].
TINY_HUNYUAN_IMAGE_LATENT_SHAPE = (2, 4, 8, 8)
TINY_HUNYUAN_IMAGE_TEXT_DIM = 16
TINY_HUNYUAN_IMAGE_TEXT_2_DIM = 12
_GLYPH_LEN = 3


def build_tiny_hunyuan_image_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``HunyuanImageTransformer2DModel`` on CPU, cache-free (config-init).

    ``text_embed_2_dim`` is set so the byT5 glyph branch (``context_embedder_2``)
    exists, mirroring the 2.1 checkpoint; ``guidance_embeds=False`` mirrors the
    community base checkpoint. ``rope_axes_dim`` must sum to
    ``attention_head_dim`` (2D image RoPE).
    """

    from diffusers import HunyuanImageTransformer2DModel

    torch.manual_seed(seed)
    return HunyuanImageTransformer2DModel(
        in_channels=TINY_HUNYUAN_IMAGE_LATENT_SHAPE[1],
        out_channels=TINY_HUNYUAN_IMAGE_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=(1, 1),
        guidance_embeds=False,
        text_embed_dim=TINY_HUNYUAN_IMAGE_TEXT_DIM,
        text_embed_2_dim=TINY_HUNYUAN_IMAGE_TEXT_2_DIM,
        rope_axes_dim=(4, 4),
    )


def _model(transformer: torch.nn.Module) -> HunyuanImageModel:
    model = HunyuanImageModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _glyph_pair(bsz: int, *, zeros: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Glyph stream pair; ``zeros=True`` mirrors the no-quoted-text zero-fill."""
    if zeros:
        return (
            torch.zeros(bsz, _GLYPH_LEN, TINY_HUNYUAN_IMAGE_TEXT_2_DIM),
            torch.zeros(bsz, _GLYPH_LEN, dtype=torch.int64),
        )
    return (
        torch.randn(bsz, _GLYPH_LEN, TINY_HUNYUAN_IMAGE_TEXT_2_DIM),
        torch.ones(bsz, _GLYPH_LEN, dtype=torch.int64),
    )


def _state(*, do_cfg: bool) -> HunyuanImageSamplingState:
    bsz = TINY_HUNYUAN_IMAGE_LATENT_SHAPE[0]
    embeds_2, mask_2 = _glyph_pair(bsz, zeros=False)
    if do_cfg:
        # Negative branch: DIFFERENT (shorter) main text sequence + zero-filled
        # glyph stream — exactly what batched CFG packing cannot handle.
        neg_embeds_2, neg_mask_2 = _glyph_pair(bsz, zeros=True)
        negatives: dict[str, Any] = {
            "negative_prompt_embeds": torch.randn(bsz, 2, TINY_HUNYUAN_IMAGE_TEXT_DIM),
            "negative_prompt_embeds_mask": torch.ones(bsz, 2, dtype=torch.int64),
            "negative_prompt_embeds_2": neg_embeds_2,
            "negative_prompt_embeds_mask_2": neg_mask_2,
        }
    else:
        negatives = {
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_mask": None,
            "negative_prompt_embeds_2": None,
            "negative_prompt_embeds_mask_2": None,
        }
    return HunyuanImageSamplingState(
        latents=torch.randn(TINY_HUNYUAN_IMAGE_LATENT_SHAPE),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, 5, TINY_HUNYUAN_IMAGE_TEXT_DIM),
        prompt_embeds_mask=torch.ones(bsz, 5, dtype=torch.int64),
        prompt_embeds_2=embeds_2,
        prompt_embeds_mask_2=mask_2,
        guidance_scale=4.0,
        do_cfg=do_cfg,
        guidance_embeds=False,
        **negatives,
    )


def test_hunyuan_image_forward_step_single_branch_when_no_cfg() -> None:
    """do_cfg=False: one forward with both text streams; guidance rides as None."""
    transformer = build_tiny_hunyuan_image_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    # Both text streams threaded into the single call.
    assert calls[0]["encoder_hidden_states"].shape[1] == 5
    assert calls[0]["encoder_attention_mask"].shape == (2, 5)
    assert calls[0]["encoder_hidden_states_2"].shape == (
        2,
        _GLYPH_LEN,
        TINY_HUNYUAN_IMAGE_TEXT_2_DIM,
    )
    assert calls[0]["encoder_attention_mask_2"].shape == (2, _GLYPH_LEN)
    # Base (non-distilled) checkpoint: no guidance embedding.
    assert calls[0]["guidance"] is None
    # Raw scheduler t, no /1000.
    torch.testing.assert_close(calls[0]["timestep"], torch.full((2,), 500.0))
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.all(out["noise_pred_uncond"] == 0)
    assert out["noise_pred"].shape == TINY_HUNYUAN_IMAGE_LATENT_SHAPE


def test_hunyuan_image_forward_step_runs_separate_cfg_with_uneven_seq_lengths() -> None:
    """do_cfg=True runs TWO forwards; each branch carries ITS OWN stream pair."""
    transformer = build_tiny_hunyuan_image_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    # Two separate forwards (cond then uncond), not one batched call.
    assert len(calls) == 2
    assert calls[0]["encoder_hidden_states"].shape[1] == 5
    assert calls[1]["encoder_hidden_states"].shape[1] == 2
    # The glyph stream follows its branch: real embeds on cond, zeros on uncond.
    assert not torch.all(calls[0]["encoder_hidden_states_2"] == 0)
    assert torch.all(calls[1]["encoder_hidden_states_2"] == 0)
    assert torch.all(calls[1]["encoder_attention_mask_2"] == 0)
    cond, uncond = out["noise_pred_cond"], out["noise_pred_uncond"]
    assert not torch.allclose(cond, uncond)
    # Classic CFG combine (NOT the pipeline's APG guider — see model docstring).
    torch.testing.assert_close(out["noise_pred"], uncond + 4.0 * (cond - uncond))


def test_hunyuan_image_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_hunyuan_image_transformer()
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_HUNYUAN_IMAGE_LATENT_SHAPE[0],), 500.0)

    restored = HunyuanImageReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    assert restored.do_cfg is True
    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
