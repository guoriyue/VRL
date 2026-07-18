"""Real-inference tests for the Mochi reversed-time backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``MochiTransformer3DModel``
runs through ``forward_step``; the batched CFG, the per-branch negation
(Mochi predicts a velocity in its reversed time axis), the reversed clock
input, and the standardized (descending-sigma) scheduler are verified against
the model's OWN real outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion.fixtures import (
    TINY_MOCHI_LATENT_SHAPE,
    TINY_MOCHI_TEXT_DIM,
    build_tiny_mochi_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.diffusion.mochi.model import (
    MochiModel,
    MochiReplayModel,
    MochiSamplingState,
    standard_mochi_scheduler,
)

_TEXT_LEN = 5
_NUM_TRAIN_TIMESTEPS = 1000


def _model(transformer: torch.nn.Module) -> MochiModel:
    model = MochiModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool) -> MochiSamplingState:
    bsz = TINY_MOCHI_LATENT_SHAPE[0]
    return MochiSamplingState(
        latents=torch.randn(TINY_MOCHI_LATENT_SHAPE),
        timesteps=torch.tensor([400.0]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, _TEXT_LEN, TINY_MOCHI_TEXT_DIM),
        prompt_attention_mask=torch.ones(bsz, _TEXT_LEN, dtype=torch.bool),
        negative_prompt_embeds=(
            torch.randn(bsz, _TEXT_LEN, TINY_MOCHI_TEXT_DIM) if do_cfg else None
        ),
        negative_prompt_attention_mask=(
            torch.ones(bsz, _TEXT_LEN, dtype=torch.bool) if do_cfg else None
        ),
        guidance_scale=4.5 if do_cfg else 1.0,
        do_cfg=do_cfg,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )


def test_mochi_forward_step_runs_real_batched_cfg_on_reversed_clock() -> None:
    """One doubled-batch call; transformer sees 1000 - t; combine is standard CFG."""
    transformer = build_tiny_mochi_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    doubled = 2 * TINY_MOCHI_LATENT_SHAPE[0]
    assert calls[0]["hidden_states"].shape[0] == doubled
    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((doubled,), float(_NUM_TRAIN_TIMESTEPS) - 400.0),
    )
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 4.5 * (cond - uncond))
    assert not torch.allclose(cond, uncond)


def test_mochi_forward_step_negates_raw_prediction() -> None:
    """The branch output is the NEGATED raw transformer output (velocity sign)."""
    transformer = build_tiny_mochi_transformer()
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    with torch.no_grad():
        raw = transformer(
            hidden_states=state.latents,
            encoder_hidden_states=state.prompt_embeds,
            timestep=torch.full(
                (TINY_MOCHI_LATENT_SHAPE[0],),
                float(_NUM_TRAIN_TIMESTEPS) - 400.0,
            ),
            encoder_attention_mask=state.prompt_attention_mask,
            return_dict=False,
        )[0]
    torch.testing.assert_close(out["noise_pred"], -raw)


def test_standard_mochi_scheduler_descends() -> None:
    """The standardized scheduler runs the ordinary descending-sigma domain."""
    scheduler = standard_mochi_scheduler(
        {"num_train_timesteps": 1000, "shift": 1.0, "invert_sigmas": True},
        8,
        torch.device("cpu"),
    )
    assert scheduler.config.invert_sigmas is False
    ts = scheduler.timesteps
    assert ts[0] > ts[-1], "timesteps must descend (standard flow domain)"
    sig = scheduler.sigmas
    assert sig[0] > sig[-1]
    # Genmo's linear-quadratic ladder starts at pure noise.
    assert float(sig[0]) == 1.0


def test_mochi_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_mochi_transformer()
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_MOCHI_LATENT_SHAPE[0],), 400.0)

    restored = MochiReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
