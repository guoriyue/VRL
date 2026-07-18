"""Real-inference tests for the CogVideoX backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``CogVideoXTransformer3DModel``
runs through ``forward_step``; the batched CFG on the BFCHW layout, the raw
integer timestep, and the deterministic external-RoPE recompute (5b config)
are verified against the model's OWN real outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_COGVIDEOX_LATENT_SHAPE,
    TINY_COGVIDEOX_TEXT_DIM,
    TINY_COGVIDEOX_TEXT_LEN,
    build_tiny_cogvideox_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.cogvideox.model import (
    CogVideoXModel,
    CogVideoXReplayModel,
    CogVideoXSamplingState,
    cogvideox_rotary_embeds,
)


def _model(transformer: torch.nn.Module) -> CogVideoXModel:
    model = CogVideoXModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool) -> CogVideoXSamplingState:
    bsz = TINY_COGVIDEOX_LATENT_SHAPE[0]
    return CogVideoXSamplingState(
        latents=torch.randn(TINY_COGVIDEOX_LATENT_SHAPE),
        timesteps=torch.tensor([499]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, TINY_COGVIDEOX_TEXT_LEN, TINY_COGVIDEOX_TEXT_DIM),
        negative_prompt_embeds=(
            torch.randn(bsz, TINY_COGVIDEOX_TEXT_LEN, TINY_COGVIDEOX_TEXT_DIM) if do_cfg else None
        ),
        guidance_scale=6.0 if do_cfg else 1.0,
        do_cfg=do_cfg,
        height=64,
        width=64,
        vae_scale_factor_spatial=8,
    )


def test_cogvideox_forward_step_runs_real_batched_cfg_on_bfchw() -> None:
    """One doubled-batch call on the frame-first layout; standard CFG combine."""
    transformer = build_tiny_cogvideox_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    doubled = 2 * TINY_COGVIDEOX_LATENT_SHAPE[0]
    assert calls[0]["hidden_states"].shape == (doubled, *TINY_COGVIDEOX_LATENT_SHAPE[1:])
    # Raw integer timestep (sinusoidal proj consumes it directly).
    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((doubled,), 499, dtype=state.timesteps.dtype),
    )
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 6.0 * (cond - uncond))
    assert out["noise_pred"].shape == TINY_COGVIDEOX_LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_cogvideox_rope_recompute_is_deterministic_and_threaded() -> None:
    """5b config: external RoPE derives from config+shape and reaches forward."""
    transformer = build_tiny_cogvideox_transformer(rope=True)
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    model.forward_step(state, 0)

    rope = calls[0]["image_rotary_emb"]
    assert rope is not None
    again = cogvideox_rotary_embeds(
        transformer.config,
        height=state.height,
        width=state.width,
        num_latent_frames=state.latents.shape[1],
        vae_scale_factor_spatial=state.vae_scale_factor_spatial,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(rope[0], again[0])
    torch.testing.assert_close(rope[1], again[1])


def test_cogvideox_2b_passes_no_rope() -> None:
    """2b config (no rotary embeddings): forward receives image_rotary_emb=None."""
    transformer = build_tiny_cogvideox_transformer(rope=False)
    calls = record_forward_calls(transformer)
    model = _model(transformer)

    model.forward_step(_state(do_cfg=False), 0)

    assert calls[0]["image_rotary_emb"] is None


def test_cogvideox_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_cogvideox_transformer(rope=True)
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_COGVIDEOX_LATENT_SHAPE[0],), 499)

    restored = CogVideoXReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
