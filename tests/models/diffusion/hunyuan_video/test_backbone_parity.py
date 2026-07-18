"""Real-inference tests for the HunyuanVideo guidance-distilled backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``HunyuanVideoTransformer3DModel``
runs through ``forward_step``; the single-branch distilled-guidance behavior
(guidance x1000, no CFG doubling) is verified against the model's OWN real
outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion.fixtures import (
    TINY_HUNYUAN_VIDEO_LATENT_SHAPE,
    TINY_HUNYUAN_VIDEO_POOLED_DIM,
    TINY_HUNYUAN_VIDEO_TEXT_DIM,
    build_tiny_hunyuan_video_transformer,
    record_forward_calls,
    stamp_test_contract,
)
from vrl.models.diffusion.hunyuan_video.model import (
    HunyuanVideoModel,
    HunyuanVideoReplayModel,
    HunyuanVideoSamplingState,
)

_TEXT_LEN = 5


def _model(transformer: torch.nn.Module) -> HunyuanVideoModel:
    model = HunyuanVideoModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    return stamp_test_contract(model)


def _state() -> HunyuanVideoSamplingState:
    bsz = TINY_HUNYUAN_VIDEO_LATENT_SHAPE[0]
    return HunyuanVideoSamplingState(
        latents=torch.randn(TINY_HUNYUAN_VIDEO_LATENT_SHAPE),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, _TEXT_LEN, TINY_HUNYUAN_VIDEO_TEXT_DIM),
        prompt_attention_mask=torch.ones(bsz, _TEXT_LEN, dtype=torch.long),
        pooled_prompt_embeds=torch.randn(bsz, TINY_HUNYUAN_VIDEO_POOLED_DIM),
        guidance_scale=6.0,
    )


def test_hunyuan_video_forward_step_single_branch_with_scaled_guidance() -> None:
    """One forward (batch not doubled); guidance rides as scale*1000."""
    transformer = build_tiny_hunyuan_video_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state()

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == TINY_HUNYUAN_VIDEO_LATENT_SHAPE[0]
    torch.testing.assert_close(
        calls[0]["guidance"],
        torch.full((TINY_HUNYUAN_VIDEO_LATENT_SHAPE[0],), 6000.0),
    )
    # Raw scheduler t, no /1000.
    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((TINY_HUNYUAN_VIDEO_LATENT_SHAPE[0],), 500.0),
    )
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.all(out["noise_pred_uncond"] == 0)
    assert out["noise_pred"].shape == TINY_HUNYUAN_VIDEO_LATENT_SHAPE


def test_hunyuan_video_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_hunyuan_video_transformer()
    model = _model(transformer)
    state = _state()

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_HUNYUAN_VIDEO_LATENT_SHAPE[0],), 500.0)

    restored = HunyuanVideoReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
