"""Qwen-Image-Layered binding: condition tokens, packed layer latents, and their codec.

A real tiny ``QwenImageTransformer2DModel`` (with the layered checkpoint's
``use_additional_t_cond``) runs through ``forward_step``; the VAE is a fake
that is exactly invertible on 8x8-constant images, so the pack/unpack codec is
checked end to end rather than against itself.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageLayeredPipeline
from diffusers.image_processor import VaeImageProcessor

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN_IN_CHANNELS,
    TINY_QWEN_JOINT_DIM,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.qwen_image.layered import (
    QwenImageLayeredModel,
    QwenImageLayeredReplayModel,
    QwenImageLayeredSamplingState,
)

_LAYERS = 2
_GRID = (2, 2)  # packed tokens per frame: 32x32 pixels / 16
_TOKENS = _GRID[0] * _GRID[1]
_Z = TINY_QWEN_IN_CHANNELS // 4


class _FakeVAE:
    """Subsample-by-8 encoder / nearest-upsample decoder: exact on 8x8-constant clips."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            z_dim=_Z,
            input_channels=4,
            latents_mean=[0.5, -0.25, 0.0, 1.0][:_Z],
            latents_std=[2.0, 0.5, 1.0, 4.0][:_Z],
            temperal_downsample=[False, True, True],
        )
        self.dtype = torch.float32

    def encode(self, pixels: torch.Tensor):
        latent = pixels[:, :_Z, :, ::8, ::8]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latents: torch.Tensor, *, return_dict: bool):
        assert not return_dict
        image = latents.repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2)
        return (image,)


def _pipeline(transformer: torch.nn.Module) -> SimpleNamespace:
    vae = _FakeVAE()
    pipe = SimpleNamespace(
        transformer=transformer,
        vae=vae,
        vae_scale_factor=8,
        latent_channels=_Z,
        image_processor=VaeImageProcessor(vae_scale_factor=16),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        device=torch.device("cpu"),
        _pack_latents=QwenImageLayeredPipeline._pack_latents,
    )
    pipe.prepare_latents = lambda *args, **kwargs: QwenImageLayeredPipeline.prepare_latents(
        pipe, *args, **kwargs
    )
    return pipe


def _model(transformer: torch.nn.Module) -> QwenImageLayeredModel:
    model = QwenImageLayeredModel(pipeline=_pipeline(transformer), device=torch.device("cpu"))
    stamp_model_precision(model)
    return model


def _state(scheduler, *, cfg: bool) -> QwenImageLayeredSamplingState:
    torch.manual_seed(1)
    return QwenImageLayeredSamplingState(
        latents=torch.randn(2, (_LAYERS + 1) * _TOKENS, TINY_QWEN_IN_CHANNELS),
        timesteps=scheduler.timesteps,
        scheduler=scheduler,
        prompt_embeds=torch.randn(2, 3, TINY_QWEN_JOINT_DIM),
        prompt_embeds_mask=torch.ones(2, 3, dtype=torch.int64),
        negative_prompt_embeds=torch.randn(2, 5, TINY_QWEN_JOINT_DIM) if cfg else None,
        negative_prompt_embeds_mask=torch.ones(2, 5, dtype=torch.int64) if cfg else None,
        guidance_scale=2.0,
        do_cfg=cfg,
        guidance_embeds=False,
        height=32,
        width=32,
        vae_scale_factor=8,
        condition_latents=torch.randn(2, _TOKENS, TINY_QWEN_IN_CHANNELS),
        layers=_LAYERS,
    )


def test_forward_appends_condition_tokens_and_returns_only_the_generated_prefix() -> None:
    transformer = build_tiny_transformer("qwen_image", use_additional_t_cond=True)
    calls = record_forward_calls(transformer)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    model = _model(transformer)
    state = _state(scheduler, cfg=False)

    pred = model.forward_step(state, 0)["noise_pred"]

    assert pred.shape == state.latents.shape
    (call,) = calls
    torch.testing.assert_close(call["hidden_states"][:, : state.latents.shape[1]], state.latents)
    torch.testing.assert_close(
        call["hidden_states"][:, state.latents.shape[1] :], state.condition_latents
    )
    assert call["img_shapes"] == [[(1, 2, 2)] * (_LAYERS + 1) + [(1, 2, 2)]] * 2
    torch.testing.assert_close(call["additional_t_cond"], torch.zeros(2, dtype=torch.long))
    direct = transformer(**call)[0]
    torch.testing.assert_close(pred, direct[:, : state.latents.shape[1]])


def test_cfg_replay_matches_rollout_logprob_and_the_condition_matters() -> None:
    transformer = build_tiny_transformer("qwen_image", use_additional_t_cond=True)
    calls = record_forward_calls(transformer)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    model = _model(transformer)
    state = _state(scheduler, cfg=True)

    pred = model.forward_step(state, 0)["noise_pred"]
    cond = transformer(**calls[0])[0][:, : state.latents.shape[1]]
    uncond = transformer(**calls[1])[0][:, : state.latents.shape[1]]
    combined = uncond + 2.0 * (cond - uncond)
    expected = combined * (cond.norm(dim=-1, keepdim=True) / combined.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(pred, expected)

    step = sde_step_with_logprob(scheduler, pred, state.timesteps[:1], state.latents, step_index=0)
    replay = QwenImageLayeredReplayModel(
        transformer=transformer, scheduler=scheduler, device=torch.device("cpu")
    )
    stamp_model_precision(replay)
    tensors = {
        **model.export_replay_tensors(state),
        "timesteps": state.timesteps.unsqueeze(0).expand(2, -1),
    }
    assert "condition_latents" in tensors and "latents_clean" in tensors
    context = model.export_batch_context(state)
    assert context["layers"] == _LAYERS
    restored = replay.restore_eval_state(tensors, context, state.latents, 0)
    replay_pred = replay.forward_step(restored, 0)["noise_pred"]
    replay_step = sde_step_with_logprob(
        scheduler,
        replay_pred,
        state.timesteps[:1],
        state.latents,
        prev_sample=step.prev_sample.detach(),
        step_index=0,
    )
    torch.testing.assert_close(replay_step.log_prob, step.log_prob)
    changed = replay.forward_step(
        replace(restored, condition_latents=restored.condition_latents + 2), 0
    )["noise_pred"]
    assert not torch.allclose(changed, replay_pred)
    (-replay_step.log_prob.mean()).backward()
    gradients = [p.grad for p in transformer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_dynamic_timesteps_follow_the_layered_pipeline_shift() -> None:
    model = _model(build_tiny_transformer("qwen_image", use_additional_t_cond=True))
    condition_tokens = 40 * 40  # a 640x640 source

    timesteps = model._set_dynamic_timesteps(5, condition_tokens, torch.device("cpu"))

    reference = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    reference.set_timesteps(
        sigmas=np.linspace(1.0, 0.0, 6)[:-1], mu=(condition_tokens / 256) ** 0.5
    )
    torch.testing.assert_close(timesteps, reference.timesteps)
    assert model.packed_token_count(640, 640) == condition_tokens


def _blocky_clip(frames: int, channels: int = 4) -> torch.Tensor:
    torch.manual_seed(3)
    coarse = torch.rand(1, channels, frames, 4, 4)
    return coarse.repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2)


def test_encode_video_is_the_exact_inverse_of_decode_on_the_packed_layout() -> None:
    model = _model(build_tiny_transformer("qwen_image", use_additional_t_cond=True))
    model._decode_height = model._decode_width = 32
    model._decode_layers = _LAYERS
    clip = _blocky_clip(_LAYERS + 1)

    packed = model.encode_video_to_latents(clip)

    assert packed.shape == (1, (_LAYERS + 1) * _TOKENS, TINY_QWEN_IN_CHANNELS)
    layers = model.decode_latents(packed)
    assert layers.shape == (1, _LAYERS, 4, 32, 32)
    torch.testing.assert_close(layers, clip[:, :, 1:].permute(0, 2, 1, 3, 4))


def test_condition_image_gets_opaque_alpha_in_rgb_mode() -> None:
    model = _model(build_tiny_transformer("qwen_image", use_additional_t_cond=True))
    source = _blocky_clip(1)[:, :, 0]  # [1, 4, 32, 32]
    opaque = source.clone()
    opaque[:, 3] = 1.0

    rgb_tokens = model.encode_condition_image(source[:, :3], height=32, width=32)
    rgba_tokens = model.encode_condition_image(source, height=32, width=32, input_mode="rgba")
    expected = model.encode_video_to_latents(opaque.unsqueeze(2))

    assert rgb_tokens.shape == (1, _TOKENS, TINY_QWEN_IN_CHANNELS)
    torch.testing.assert_close(rgb_tokens, expected)
    assert not torch.allclose(rgba_tokens, rgb_tokens)
    with pytest.raises(ValueError, match="request geometry"):
        model.encode_condition_image(source[:, :, :16], height=32, width=32)


def test_sft_weight_map_tiles_the_mask_over_every_frame_in_token_order() -> None:
    model = _model(build_tiny_transformer("qwen_image", use_additional_t_cond=True))
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :16, :16] = 1.0  # bright top-left token only

    weights = model.sft_weight_map(
        mask, latent_shape=((_LAYERS + 1) * _TOKENS, TINY_QWEN_IN_CHANNELS), height=32, width=32
    )

    assert weights.shape == ((_LAYERS + 1) * _TOKENS, 1)
    expected_frame = torch.tensor([4.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(weights[:, 0], expected_frame.repeat(_LAYERS + 1))
    assert float(weights.mean()) == 1.0
    with pytest.raises(ValueError, match="whole number"):
        model.sft_weight_map(mask, latent_shape=(5, TINY_QWEN_IN_CHANNELS), height=32, width=32)


def test_prepare_sampling_packs_layers_plus_one_frames_and_records_decode_geometry() -> None:
    from vrl.generation.types import DenoiseRequest

    transformer = build_tiny_transformer("qwen_image", use_additional_t_cond=True)
    model = _model(transformer)
    encoded = {
        "prompt_embeds": torch.randn(2, 3, TINY_QWEN_JOINT_DIM),
        "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.int64),
        "condition_latents": torch.randn(2, _TOKENS, TINY_QWEN_IN_CHANNELS),
    }
    request = DenoiseRequest(
        width=32, height=32, frame_count=1, num_steps=4, guidance_scale=1.0, seed=7
    )

    state = model.prepare_sampling(request, encoded, layers=_LAYERS)

    assert state.latents.shape == (2, (_LAYERS + 1) * _TOKENS, TINY_QWEN_IN_CHANNELS)
    assert state.layers == _LAYERS and state.timesteps.shape == (4,)
    assert (model._decode_height, model._decode_width, model._decode_layers) == (32, 32, _LAYERS)
    assert not state.do_cfg
