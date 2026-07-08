"""Real-inference tests for the Lumina2 CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``Lumina2Transformer2DModel``
runs through ``forward_step``; the separate-branch CFG, the per-branch sign
flip (Lumina predicts the reversed-time velocity), the norm-preserving
combine, and the reversed timestep are verified against the model's OWN real
outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion.fixtures import (
    TINY_LUMINA2_CAP_DIM,
    TINY_LUMINA2_LATENT_SHAPE,
    build_tiny_lumina2_transformer,
    record_forward_calls,
)
from vrl.models.diffusion.lumina2.model import (
    Lumina2Model,
    Lumina2ReplayModel,
    Lumina2SamplingState,
)

_TEXT_LEN = 5
_NUM_TRAIN_TIMESTEPS = 1000


def _model(transformer: torch.nn.Module) -> Lumina2Model:
    return Lumina2Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )


def _state(*, do_cfg: bool) -> Lumina2SamplingState:
    bsz = TINY_LUMINA2_LATENT_SHAPE[0]
    return Lumina2SamplingState(
        latents=torch.randn(TINY_LUMINA2_LATENT_SHAPE),
        timesteps=torch.tensor([400.0]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, _TEXT_LEN, TINY_LUMINA2_CAP_DIM),
        prompt_attention_mask=torch.ones(bsz, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=(
            torch.randn(bsz, _TEXT_LEN, TINY_LUMINA2_CAP_DIM) if do_cfg else None
        ),
        negative_prompt_attention_mask=(
            torch.ones(bsz, _TEXT_LEN, dtype=torch.long) if do_cfg else None
        ),
        guidance_scale=4.0 if do_cfg else 1.0,
        do_cfg=do_cfg,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )


def test_lumina2_forward_step_runs_separate_cfg_with_norm_rescale() -> None:
    """Two branch calls; combine matches diffusers' renormed CFG on negated preds."""
    transformer = build_tiny_lumina2_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    # Separate CFG: two transformer calls, batch NOT doubled.
    assert len(calls) == 2
    assert calls[0]["hidden_states"].shape[0] == TINY_LUMINA2_LATENT_SHAPE[0]
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    combined = uncond + 4.0 * (cond - uncond)
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    comb_norm = torch.norm(combined, dim=-1, keepdim=True)
    torch.testing.assert_close(out["noise_pred"], combined * (cond_norm / comb_norm))
    assert not torch.allclose(cond, uncond)


def test_lumina2_forward_step_negates_raw_prediction() -> None:
    """The branch output is the NEGATED raw transformer output (velocity sign)."""
    transformer = build_tiny_lumina2_transformer()
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    with torch.no_grad():
        raw = transformer(
            hidden_states=state.latents,
            timestep=torch.full(
                (TINY_LUMINA2_LATENT_SHAPE[0],),
                1.0 - 400.0 / _NUM_TRAIN_TIMESTEPS,
            ),
            encoder_hidden_states=state.prompt_embeds,
            encoder_attention_mask=state.prompt_attention_mask,
            return_dict=False,
        )[0]
    torch.testing.assert_close(out["noise_pred"], -raw)


def test_lumina2_forward_step_reverses_timestep() -> None:
    """The transformer sees 1 - t/num_train_timesteps, not the raw scheduler t."""
    transformer = build_tiny_lumina2_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    model.forward_step(state, 0)

    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((TINY_LUMINA2_LATENT_SHAPE[0],), 1.0 - 400.0 / _NUM_TRAIN_TIMESTEPS),
    )


def test_lumina2_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_lumina2_transformer()
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_LUMINA2_LATENT_SHAPE[0],), 400.0)

    restored = Lumina2ReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
