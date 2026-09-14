"""Real-inference tests for the PixArt-Sigma batched-CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``PixArtTransformer2DModel``
runs through ``forward_step``; the batched classifier-free guidance, the raw
integer timestep, and the learned-sigma channel batch are verified against the
model's OWN real outputs, never a re-derived formula.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tests.models.steps.denoise.fixtures import record_forward_calls, stamp_model_precision
from vrl.models.families.pixart_sigma.model import (
    PixArtSigmaModel,
    PixArtSigmaReplayModel,
    PixArtSigmaSamplingState,
    pixart_ddim_scheduler,
)

# Tiny real PixArt geometry (CPU): latents [B, C, H, W] with the LEARNED-SIGMA
# transformer emitting 2*C channels (epsilon prediction + variance).
TINY_PIXART_LATENT_SHAPE = (2, 4, 8, 8)
TINY_PIXART_CAPTION_DIM = 16
_TEXT_LEN = 5


def build_tiny_pixart_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``PixArtTransformer2DModel`` on CPU, cache-free (config-init).

    ``out_channels = 2 * in_channels`` mirrors the real checkpoints' learned
    sigma; ``sample_size=8`` keeps ``use_additional_conditions`` off (only the
    128-sample_size 2K checkpoints turn it on), matching 512/1024-MS.
    """

    from diffusers import PixArtTransformer2DModel

    torch.manual_seed(seed)
    return PixArtTransformer2DModel(
        sample_size=8,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        in_channels=TINY_PIXART_LATENT_SHAPE[1],
        out_channels=2 * TINY_PIXART_LATENT_SHAPE[1],
        cross_attention_dim=16,
        caption_channels=TINY_PIXART_CAPTION_DIM,
    )


def _model(transformer: torch.nn.Module) -> PixArtSigmaModel:
    model = PixArtSigmaModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool) -> PixArtSigmaSamplingState:
    bsz = TINY_PIXART_LATENT_SHAPE[0]
    return PixArtSigmaSamplingState(
        latents=torch.randn(TINY_PIXART_LATENT_SHAPE),
        timesteps=torch.tensor([499]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, _TEXT_LEN, TINY_PIXART_CAPTION_DIM),
        prompt_attention_mask=torch.ones(bsz, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=(
            torch.randn(bsz, _TEXT_LEN, TINY_PIXART_CAPTION_DIM) if do_cfg else None
        ),
        negative_prompt_attention_mask=(
            torch.ones(bsz, _TEXT_LEN, dtype=torch.long) if do_cfg else None
        ),
        guidance_scale=4.5 if do_cfg else 1.0,
        do_cfg=do_cfg,
    )


def test_pixart_forward_step_runs_real_batched_cfg() -> None:
    """One doubled-batch transformer call; combine matches uncond+s*(cond-uncond)."""
    transformer = build_tiny_pixart_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    doubled = 2 * TINY_PIXART_LATENT_SHAPE[0]
    assert calls[0]["hidden_states"].shape[0] == doubled
    # The attention masks batched with the branches (uncond rows first), and
    # the added_cond_kwargs dict rode both branches unchanged.
    assert calls[0]["encoder_attention_mask"].shape == (doubled, _TEXT_LEN)
    assert calls[0]["added_cond_kwargs"] == {"resolution": None, "aspect_ratio": None}
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 4.5 * (cond - uncond))
    assert out["noise_pred"].shape == TINY_PIXART_LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_pixart_forward_step_single_branch_skips_cfg() -> None:
    """No CFG: single conditional forward, noise == cond, zero uncond placeholder."""
    transformer = build_tiny_pixart_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == TINY_PIXART_LATENT_SHAPE[0]
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.all(out["noise_pred_uncond"] == 0)


def test_pixart_learned_sigma_chunks_prediction_channels() -> None:
    """noise_pred is the FIRST half of the transformer's 2*C-channel output."""
    transformer = build_tiny_pixart_transformer()
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    bsz, channels = TINY_PIXART_LATENT_SHAPE[0], TINY_PIXART_LATENT_SHAPE[1]
    raw = transformer(
        hidden_states=state.latents,
        timestep=torch.full((bsz,), 499, dtype=torch.long),
        encoder_hidden_states=state.prompt_embeds,
        encoder_attention_mask=state.prompt_attention_mask,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        return_dict=False,
    )[0]
    assert raw.shape[1] == 2 * channels
    assert out["noise_pred"].shape[1] == channels
    torch.testing.assert_close(out["noise_pred"], raw.chunk(2, dim=1)[0])


def test_pixart_forward_step_passes_raw_integer_timestep() -> None:
    """The scheduler timestep reaches the transformer unscaled, as int64."""
    transformer = build_tiny_pixart_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    model.forward_step(state, 0)

    timestep = calls[0]["timestep"]
    assert timestep.dtype == torch.int64
    torch.testing.assert_close(
        timestep,
        torch.full((TINY_PIXART_LATENT_SHAPE[0],), 499, dtype=torch.int64),
    )


def test_pixart_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_pixart_transformer()
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full(
        (TINY_PIXART_LATENT_SHAPE[0],),
        499,
        dtype=torch.int64,
    )

    restored = PixArtSigmaReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])


def test_pixart_ddim_scheduler_is_epsilon_ddim_without_clipping() -> None:
    """The RL scheduler is a DDIM on the shipped beta ladder, Gaussian-safe."""
    from diffusers import DDIMScheduler

    # The keys a shipped PixArt-Sigma DPMSolverMultistepScheduler config
    # carries (solver-only keys must be ignored, betas must be honored).
    shipped_dpm_config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "trained_betas": None,
        "prediction_type": "epsilon",
        "solver_order": 2,
        "algorithm_type": "dpmsolver++",
    }

    scheduler = pixart_ddim_scheduler(shipped_dpm_config, 10, "cpu")

    assert isinstance(scheduler, DDIMScheduler)
    assert scheduler.config.prediction_type == "epsilon"
    assert scheduler.config.clip_sample is False
    assert scheduler.config.thresholding is False
    assert scheduler.config.set_alpha_to_one is True
    assert scheduler.config.timestep_spacing == "leading"
    assert scheduler.config.beta_schedule == "linear"
    assert scheduler.config.num_train_timesteps == 1000
    assert len(scheduler.timesteps) == 10
