"""Real-inference tests for the SANA batched-CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``SanaTransformer2DModel`` runs
through ``forward_step``; the batched classifier-free guidance and the
``encoder_attention_mask`` threading are verified against the model's OWN real
outputs, never a re-derived formula.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion.fixtures import (
    TINY_SANA_CAPTION_DIM,
    TINY_SANA_LATENT_SHAPE,
    build_tiny_sana_transformer,
    record_forward_calls,
)
from vrl.models.diffusion.sana.model import (
    SanaModel,
    SanaReplayModel,
    SanaSamplingState,
)

_TEXT_LEN = 5


def _model(transformer: torch.nn.Module) -> SanaModel:
    return SanaModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )


def _state(*, do_cfg: bool) -> SanaSamplingState:
    bsz = TINY_SANA_LATENT_SHAPE[0]
    return SanaSamplingState(
        latents=torch.randn(TINY_SANA_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(bsz, _TEXT_LEN, TINY_SANA_CAPTION_DIM),
        prompt_attention_mask=torch.ones(bsz, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=(
            torch.randn(bsz, _TEXT_LEN, TINY_SANA_CAPTION_DIM) if do_cfg else None
        ),
        negative_prompt_attention_mask=(
            torch.ones(bsz, _TEXT_LEN, dtype=torch.long) if do_cfg else None
        ),
        guidance_scale=4.5 if do_cfg else 1.0,
        do_cfg=do_cfg,
    )


def test_sana_requires_validated_forward_without_outer_autocast() -> None:
    """SANA's family descriptor owns one shared forward requirement."""

    from vrl.families.registry import get_model_family_entry

    requirement = get_model_family_entry("sana").family_build.forward_precision
    assert requirement.autocast == "off"
    assert requirement.float32_precision == "ieee"
    assert not hasattr(SanaModel, "disable_rollout_autocast")
    assert not hasattr(SanaModel, "disable_train_autocast")
    assert not hasattr(SanaModel, "allow_tf32")


def test_sana_fp16_output_is_promoted_before_cfg() -> None:
    """SANA matches native FP32 CFG instead of combining rounded FP16 branches."""

    transformer = build_tiny_sana_transformer().half()
    calls = record_forward_calls(transformer)
    raw_outputs: list[torch.Tensor] = []
    transformer.register_forward_hook(
        lambda _module, _args, _kwargs, output: raw_outputs.append(output[0].detach()),
        with_kwargs=True,
    )
    state = _state(do_cfg=True)

    out = _model(transformer).forward_step(state, 0)

    raw_uncond, raw_cond = raw_outputs[0].float().chunk(2)
    native_cfg = raw_uncond + state.guidance_scale * (raw_cond - raw_uncond)
    fp16_cfg = (
        raw_outputs[0].chunk(2)[0]
        + state.guidance_scale * (raw_outputs[0].chunk(2)[1] - raw_outputs[0].chunk(2)[0])
    ).float()
    torch.testing.assert_close(out["noise_pred"], native_cfg, rtol=0.0, atol=0.0)
    assert not torch.equal(out["noise_pred"], fp16_cfg)
    assert calls[0]["timestep"].dtype is torch.float32


def test_sana_forward_step_runs_real_batched_cfg() -> None:
    """One doubled-batch transformer call; combine matches uncond+s*(cond-uncond)."""
    transformer = build_tiny_sana_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    doubled = 2 * TINY_SANA_LATENT_SHAPE[0]
    assert calls[0]["hidden_states"].shape[0] == doubled
    # The attention masks batched with the branches (uncond rows first).
    assert calls[0]["encoder_attention_mask"].shape == (doubled, _TEXT_LEN)
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 4.5 * (cond - uncond))
    assert out["noise_pred"].dtype is torch.float32
    assert out["noise_pred"].shape == TINY_SANA_LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_sana_forward_step_single_branch_skips_cfg() -> None:
    """No CFG: single conditional forward, noise == cond, zero uncond placeholder."""
    transformer = build_tiny_sana_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == TINY_SANA_LATENT_SHAPE[0]
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.all(out["noise_pred_uncond"] == 0)


def test_sana_forward_step_applies_timestep_scale() -> None:
    """The raw scheduler timestep is scaled by transformer.config.timestep_scale."""
    transformer = build_tiny_sana_transformer()
    transformer.register_to_config(timestep_scale=0.5)
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    model.forward_step(state, 0)

    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((TINY_SANA_LATENT_SHAPE[0],), 5.0 * 0.5),
    )
    assert calls[0]["timestep"].dtype is state.timesteps.dtype


def test_sana_replay_roundtrip_restores_equivalent_state() -> None:
    """export -> restore rebuilds a state whose forward matches the original."""
    transformer = build_tiny_sana_transformer()
    model = _model(transformer)
    state = _state(do_cfg=True)

    replay = model.export_replay_tensors(state)
    context = model.export_batch_context(state)
    replay["timesteps"] = torch.full((TINY_SANA_LATENT_SHAPE[0],), 5.0)

    restored = SanaReplayModel(
        transformer=transformer,
        scheduler=None,
    ).restore_eval_state(replay, context, state.latents, 0)

    out_orig = model.forward_step(state, 0)
    out_restored = model.forward_step(restored, 0)
    torch.testing.assert_close(out_orig["noise_pred"], out_restored["noise_pred"])
