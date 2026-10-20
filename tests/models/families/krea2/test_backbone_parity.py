"""Real-inference tests for the Krea 2 backbone wrapper.

A real (tiny, cache-free) ``Krea2Transformer2DModel`` runs through
``forward_step``; the flow-time scaling, the batch-shared rotary grid, the
separate-branch CFG, the pipeline-less replay restore and the packed-latent
prepare / decode through a real ``Krea2Pipeline`` shell over a real tiny
Qwen-Image VAE are verified against the model's OWN outputs.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_KREA2_IN_CHANNELS,
    TINY_KREA2_TEXT_DIM,
    TINY_KREA2_TEXT_LAYERS,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import DenoiseRequest
from vrl.models.families.krea2.model import (
    Krea2Model,
    Krea2PipelineModel,
    Krea2SamplingState,
)

# 64x64 px / 16 px per token -> 4x4 latent grid -> 16 packed tokens.
_H = _W = 64
_SEQ = 16
_TEXT_LEN = 5
_GUIDANCE = 4.5


def _model(transformer: torch.nn.Module, pipeline: object | None = None) -> Krea2PipelineModel:
    model = Krea2PipelineModel(
        pipeline=pipeline
        if pipeline is not None
        else SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool) -> Krea2SamplingState:
    torch.manual_seed(1)
    mask = torch.ones(2, _TEXT_LEN, dtype=torch.bool)
    mask[1, 3:] = False  # the second row is padded
    return Krea2SamplingState(
        latents=torch.randn(2, _SEQ, TINY_KREA2_IN_CHANNELS),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_KREA2_TEXT_LAYERS, TINY_KREA2_TEXT_DIM),
        prompt_embeds_mask=mask,
        negative_prompt_embeds=(
            torch.randn(2, _TEXT_LEN, TINY_KREA2_TEXT_LAYERS, TINY_KREA2_TEXT_DIM)
            if do_cfg
            else None
        ),
        negative_prompt_embeds_mask=torch.ones(2, _TEXT_LEN, dtype=torch.bool) if do_cfg else None,
        position_ids=Krea2Model._position_ids(_TEXT_LEN, _H, _W, torch.device("cpu")),
        guidance_scale=_GUIDANCE,
        do_cfg=do_cfg,
        height=_H,
        width=_W,
    )


def test_krea2_forward_step_feeds_flow_time_and_the_shared_rotary_grid() -> None:
    """do_cfg=False: one forward at t/1000 with the [S + img, 3] grid (text rows at the
    origin, image rows on the 4x4 latent grid) and the bool text mask."""
    transformer = build_tiny_transformer("krea2")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["timestep"], torch.full((2,), 0.5))
    position_ids = calls[0]["position_ids"]
    assert position_ids.shape == (_TEXT_LEN + _SEQ, 3)
    assert torch.equal(position_ids[:_TEXT_LEN], torch.zeros(_TEXT_LEN, 3))
    assert position_ids[_TEXT_LEN:, 1].max() == 3 and position_ids[_TEXT_LEN:, 2].max() == 3
    assert calls[0]["encoder_attention_mask"].dtype is torch.bool
    assert calls[0]["encoder_hidden_states"].shape == (
        2,
        _TEXT_LEN,
        TINY_KREA2_TEXT_LAYERS,
        TINY_KREA2_TEXT_DIM,
    )
    assert out["noise_pred"].shape == (2, _SEQ, TINY_KREA2_IN_CHANNELS)


def test_krea2_forward_step_runs_separate_cfg_branches() -> None:
    """do_cfg=True runs TWO forwards sharing the rotary grid; the combine is the standard
    ``uncond + s * (cond - uncond)`` on the real branch outputs."""
    transformer = build_tiny_transformer("krea2")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    assert torch.equal(calls[0]["position_ids"], calls[1]["position_ids"])
    cond = model.forward_step(replace(state, do_cfg=False), 0)["noise_pred"]
    uncond = model.forward_step(
        replace(
            state,
            do_cfg=False,
            prompt_embeds=state.negative_prompt_embeds,
            prompt_embeds_mask=state.negative_prompt_embeds_mask,
        ),
        0,
    )["noise_pred"]
    assert not torch.allclose(cond, uncond)
    torch.testing.assert_close(out["noise_pred"], uncond + _GUIDANCE * (cond - uncond))


def test_krea2_replay_model_rebuilds_the_rotary_grid_without_a_pipeline() -> None:
    """The pipeline-less replay model restores the CFG state from trajectory tensors and
    context, rebuilds position_ids from the prompt length and spatial shape, and
    reproduces the rollout forward on the same weights."""
    transformer = build_tiny_transformer("krea2")
    rollout = _model(transformer)
    replay = Krea2Model(transformer=transformer, scheduler=None, device=torch.device("cpu"))
    stamp_model_precision(replay)
    state = _state(do_cfg=True)

    restored = replay.restore_eval_state(
        {
            **rollout.export_replay_tensors(state),
            "timesteps": torch.tensor([[900.0, 500.0], [900.0, 500.0]]),
        },
        rollout.export_batch_context(state),
        state.latents,
        1,
    )

    assert restored.do_cfg is True
    torch.testing.assert_close(restored.position_ids, state.position_ids)
    torch.testing.assert_close(
        replay.forward_step(restored, 0)["noise_pred"],
        rollout.forward_step(state, 0)["noise_pred"],
    )
    exported = rollout.export_replay_tensors(_state(do_cfg=False))
    assert "negative_prompt_embeds" not in exported
    assert exported["latents_clean"].requires_grad is False


def _tiny_pipeline_shell(transformer: torch.nn.Module, *, latents_mean: float = 0.0):
    """A real ``Krea2Pipeline`` over the tiny DiT, a real tiny Qwen-Image video VAE
    (f8 spatial, z_dim 4) and the released scheduler config; no text encoder."""
    import json
    from pathlib import Path

    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline

    config = json.loads(
        Path("tests/models/steps/denoise/fixtures/scheduler_configs.json").read_text()
    )["krea2"]["config"]
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=TINY_KREA2_IN_CHANNELS // 4,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        latents_mean=[latents_mean] * (TINY_KREA2_IN_CHANNELS // 4),
        latents_std=[1.0] * (TINY_KREA2_IN_CHANNELS // 4),
    ).eval()
    return Krea2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )


def test_krea2_prepare_sampling_packs_latents_and_sets_the_mu_shifted_grid() -> None:
    """Through a real Krea2Pipeline shell: 64 px -> 16 packed tokens of 16 channels,
    a [S + 16, 3] grid, the dynamic-shift schedule set from the token count, and
    geometry that is not a multiple of 16 rejected instead of rounded."""
    transformer = build_tiny_transformer("krea2")
    model = _model(transformer, _tiny_pipeline_shell(transformer))
    encoded = {
        "prompt_embeds": torch.randn(2, _TEXT_LEN, TINY_KREA2_TEXT_LAYERS, TINY_KREA2_TEXT_DIM),
        "prompt_embeds_mask": torch.ones(2, _TEXT_LEN, dtype=torch.bool),
    }
    request = DenoiseRequest(
        width=_W, height=_H, frame_count=1, num_steps=4, guidance_scale=1.0, seed=7
    )

    state = model.prepare_sampling(request, encoded)

    assert state.latents.shape == (2, _SEQ, TINY_KREA2_IN_CHANNELS)
    assert state.latents.dtype is torch.float32
    assert state.position_ids.shape == (_TEXT_LEN + _SEQ, 3)
    assert state.do_cfg is False
    assert len(state.timesteps) == 4 and state.timesteps[0] == 1000.0
    torch.testing.assert_close(state.timesteps, model.scheduler.timesteps)
    with pytest.raises(ValueError, match="multiples of 16"):
        model.prepare_sampling(replace(request, height=72), encoded)


def test_krea2_decode_unpacks_and_denormalizes_through_a_real_qwen_image_vae() -> None:
    """Packed [B, 16, 16] latents unpack to the VAE's [B, 4, 1, 8, 8], decode to RGB at 16
    px per token with the temporal frame dropped, and the VAE's latents_mean shifts
    the picture."""
    transformer = build_tiny_transformer("krea2")
    model = _model(transformer, _tiny_pipeline_shell(transformer))
    model._decode_height, model._decode_width = _H, _W
    latents = torch.randn(2, _SEQ, TINY_KREA2_IN_CHANNELS)

    image = model.decode_latents(latents)

    assert image.shape == (2, 3, _H, _W)
    assert image.min() >= 0.0 and image.max() <= 1.0
    shifted = _model(transformer, _tiny_pipeline_shell(transformer, latents_mean=1.0))
    shifted._decode_height, shifted._decode_width = _H, _W
    assert not torch.allclose(shifted.decode_latents(latents), image)


def test_krea2_prepare_replay_sets_the_rollout_timestep_grid() -> None:
    """The pipeline-less replay scheduler carries the rollout's mu-shifted grid: rollout
    derives ``mu`` from the packed token count in ``prepare_sampling``; ``prepare_replay``
    rebuilds it from the run's fixed resolution (512 px -> 1024 tokens)."""
    import json
    from pathlib import Path

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.config.precision import RolePrecision
    from vrl.models.interfaces.runtime import ModelBuild

    config = json.loads(
        Path("tests/models/steps/denoise/fixtures/scheduler_configs.json").read_text()
    )["krea2"]["config"]
    assert config["use_dynamic_shifting"] is True and config["max_image_seq_len"] == 6400
    height, width, num_steps = 512, 512, 10

    rollout = _model(build_tiny_transformer("krea2"))
    rollout_scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    rollout.pipeline.scheduler = rollout_scheduler
    expected = rollout._set_dynamic_timesteps(num_steps, 1024, torch.device("cpu")).clone()

    replay = Krea2Model(
        transformer=build_tiny_transformer("krea2"),
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config),
        device=torch.device("cpu"),
    )
    replay.prepare_replay(
        ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype=torch.float32,
            family="krea2",
            precision=RolePrecision("fp32", "ieee", outer_autocast=False),
            sampling_config={"height": height, "width": width, "num_steps": num_steps},
        )
    )

    torch.testing.assert_close(replay.scheduler.timesteps, expected)
    torch.testing.assert_close(replay.scheduler.sigmas, rollout_scheduler.sigmas)
