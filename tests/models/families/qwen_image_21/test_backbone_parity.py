"""Real-inference tests for the Qwen-Image-2.1 single-stream backbone wrapper.

A real (tiny, cache-free) ``QwenImage21Transformer2DModel`` runs through
``forward_step``; the joint ``img_mask`` construction, the no-CFG default
(one forward), the separate-branch true CFG with a plain linear combine, and the
pipeline-less replay restore are verified against the model's OWN outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN21_CONTEXT_DIM,
    TINY_QWEN21_IN_CHANNELS,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.qwen_image_21.model import (
    QwenImage21Model,
    QwenImage21ReplayModel,
    QwenImage21SamplingState,
)

# 64x64 px / vae_scale_factor 16 -> 4x4 latent grid -> 16 packed tokens.
_H = _W = 64
_VSF = 16
_SEQ = (_H // _VSF) * (_W // _VSF)


def _model(transformer: torch.nn.Module) -> QwenImage21Model:
    model = QwenImage21Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, txt_len: int = 3, negative_len: int | None = None) -> QwenImage21SamplingState:
    negative = negative_len is not None
    return QwenImage21SamplingState(
        latents=torch.randn(2, _SEQ, TINY_QWEN21_IN_CHANNELS),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, txt_len, TINY_QWEN21_CONTEXT_DIM),
        prompt_embeds_mask=None,
        image_pad_mask=torch.zeros(2, txt_len, dtype=torch.bool),
        negative_prompt_embeds=(
            torch.randn(2, negative_len, TINY_QWEN21_CONTEXT_DIM) if negative else None
        ),
        negative_prompt_embeds_mask=None,
        negative_image_pad_mask=(
            torch.zeros(2, negative_len, dtype=torch.bool) if negative else None
        ),
        guidance_scale=4.0 if negative else 1.0,
        do_cfg=negative,
        height=_H,
        width=_W,
        vae_scale_factor=_VSF,
    )


def test_qwen21_forward_step_single_branch_when_no_cfg() -> None:
    """Default true_cfg_scale=1.0: ONE forward; img_mask spans text + target slots."""
    transformer = build_tiny_transformer("qwen_image_21")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(txt_len=3)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    call = calls[0]
    # Timestep normalized by 1000 (500 -> 0.5); latents fed unpatched.
    torch.testing.assert_close(call["timestep"], torch.full((2,), 0.5))
    assert call["hidden_states"].shape == (2, _SEQ, TINY_QWEN21_IN_CHANNELS)
    assert call["img_shapes"] == [[(1, _H // _VSF, _W // _VSF)]] * 2
    # Joint mask: 3 text slots (all False for pure text) + one slot per 2x2 latent group.
    img_mask = call["img_mask"]
    assert img_mask.dtype is torch.bool
    assert img_mask.shape == (2, 3 + _SEQ // 4)
    assert not img_mask[:, :3].any()
    assert img_mask[:, 3:].all()
    # No prefix KV caching on either path (LoRA changes the prefix KV).
    assert call.get("kv_cache") is None and call.get("kv_cache_mode") is None
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert out["noise_pred"].shape == (2, _SEQ, TINY_QWEN21_IN_CHANNELS)


def test_qwen21_forward_step_runs_separate_cfg_with_linear_combine() -> None:
    """do_cfg=True runs TWO forwards at different text lengths; combine is plain linear."""
    transformer = build_tiny_transformer("qwen_image_21")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(txt_len=5, negative_len=2)

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    assert calls[0]["encoder_hidden_states"].shape[1] == 5
    assert calls[1]["encoder_hidden_states"].shape[1] == 2
    assert calls[1]["img_mask"].shape == (2, 2 + _SEQ // 4)
    cond, uncond = out["noise_pred_cond"], out["noise_pred_uncond"]
    assert not torch.allclose(cond, uncond)
    # QwenImage21Pipeline: noise = uncond + s * (cond - uncond), no norm rescale.
    torch.testing.assert_close(out["noise_pred"], uncond + 4.0 * (cond - uncond))


def test_qwen21_replay_model_restores_state_without_a_pipeline() -> None:
    """The pipeline-less replay model rebuilds img_shapes/img_mask and matches rollout."""
    transformer = build_tiny_transformer("qwen_image_21")
    rollout = _model(transformer)
    state = _state(txt_len=3)
    expected = rollout.forward_step(state, 0)["noise_pred"]

    replay = QwenImage21ReplayModel(
        transformer=transformer,
        scheduler=None,
        device=torch.device("cpu"),
    )
    stamp_model_precision(replay)
    replay_tensors = {
        **rollout.export_replay_tensors(state),
        # [B, num_steps] per-sample timesteps.
        "timesteps": torch.tensor([[500.0, 600.0], [500.0, 600.0]]),
    }
    # The vision-slot mask travels as int64 and is cast back to bool for the model.
    assert replay_tensors["image_pad_mask"].dtype is torch.int64
    batch_context = rollout.export_batch_context(state)

    restored = replay.restore_eval_state(replay_tensors, batch_context, state.latents, 0)
    out = replay.forward_step(restored, 0)

    torch.testing.assert_close(out["noise_pred"], expected)


def test_qwen21_prepare_replay_sets_the_rollout_timestep_grid() -> None:
    """The pipeline-less replay scheduler must carry the rollout's mu-shifted grid.

    Rollout derives ``mu`` from the packed latent length inside ``prepare_sampling``;
    the generic replay loader knows nothing about it, so ``prepare_replay`` must
    rebuild the same grid from the run's fixed resolution. The SDE log-prob math
    indexes ``scheduler.sigmas`` by timestep, so a missing/unshifted grid makes
    every replay step read the wrong sigma.
    """
    import json
    from pathlib import Path

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.config.precision import RolePrecision
    from vrl.models.interfaces.runtime import ModelBuild

    fixture = Path("tests/models/steps/denoise/fixtures/scheduler_configs.json")
    config = json.loads(fixture.read_text())["qwen_image_21"]["config"]
    assert config["use_dynamic_shifting"] is True
    height, width, num_steps = 512, 512, 10

    rollout = _model(build_tiny_transformer("qwen_image_21"))
    rollout_scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    rollout.pipeline.scheduler = rollout_scheduler
    # 512 px -> 32x32 latent cells, packed unpatched -> 1024 tokens.
    expected = rollout._set_dynamic_timesteps(num_steps, 1024, torch.device("cpu")).clone()

    replay = QwenImage21ReplayModel(
        transformer=build_tiny_transformer("qwen_image_21"),
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config),
        device=torch.device("cpu"),
    )
    replay.prepare_replay(
        ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype=torch.float32,
            family="qwen_image_21",
            precision=RolePrecision("fp32", "ieee", outer_autocast=False),
            sampling_config={"height": height, "width": width, "num_steps": num_steps},
        )
    )

    torch.testing.assert_close(replay.scheduler.timesteps, expected)
    torch.testing.assert_close(replay.scheduler.sigmas, rollout_scheduler.sigmas)
    # Un-shifted grid would differ: the shift is what the replay must reproduce.
    plain = FlowMatchEulerDiscreteScheduler.from_config(config)
    plain.set_timesteps(num_steps, sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=0.0)
    assert not torch.allclose(plain.timesteps, expected)


def _decode_pipeline(transformer: torch.nn.Module) -> SimpleNamespace:
    """A pipeline whose VAE decodes 4 latent channels straight into RGBA pixels."""

    def unpack(latents, height, width, vae_scale_factor):
        h, w = height // vae_scale_factor, width // vae_scale_factor
        return latents.transpose(1, 2).reshape(latents.shape[0], -1, 1, h, w)

    vae = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4),
        decode=lambda batch, return_dict=False: (batch,),
    )
    return SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        vae=vae,
        vae_scale_factor=_VSF,
        _unpack_latents=unpack,
        image_processor=SimpleNamespace(postprocess=lambda image, output_type: image),
    )


def test_qwen21_decode_composites_alpha_over_white_unless_rgba() -> None:
    """RGB mode flattens the decoded alpha onto white; RGBA mode keeps four channels."""
    model = QwenImage21Model(
        pipeline=_decode_pipeline(build_tiny_transformer("qwen_image_21")),
        device=torch.device("cpu"),
    )
    model._decode_height, model._decode_width = _H, _W
    latents = torch.full((1, _SEQ, 4), 0.25)
    latents[..., 3] = 0.5  # alpha

    model._output_mode = "rgb"
    rgb = model.decode_latents(latents)
    model._output_mode = "rgba"
    rgba = model.decode_latents(latents)

    assert rgb.shape == (1, 3, _H // _VSF, _W // _VSF)
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.25 * 0.5 + 0.5))
    assert rgba.shape == (1, 4, _H // _VSF, _W // _VSF)
    torch.testing.assert_close(rgba[:, 3], torch.full_like(rgba[:, 3], 0.5))
