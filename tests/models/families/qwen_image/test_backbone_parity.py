"""Real-inference tests for the Qwen-Image true-CFG backbone wrapper.

A real (tiny, cache-free) ``QwenImageTransformer2DModel`` runs through
``forward_step``; the separate-branch CFG (cond/uncond at DIFFERENT sequence
lengths) and the norm-preserving combine are verified against the model's OWN
real outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN_IN_CHANNELS,
    TINY_QWEN_JOINT_DIM,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.qwen_image.model import (
    QwenImageModel,
    QwenImageReplayModel,
    QwenImageSamplingState,
)

_SEQ = 16  # packed latent token count (lh * lw, lh = lw = 4)


def _model(transformer: torch.nn.Module) -> QwenImageModel:
    model = QwenImageModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def test_qwen_forward_step_single_branch_when_no_cfg() -> None:
    """do_cfg=False: one transformer forward, noise == cond, uncond is zeros."""
    transformer = build_tiny_transformer("qwen_image")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = QwenImageSamplingState(
        latents=torch.randn(2, _SEQ, TINY_QWEN_IN_CHANNELS),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, 3, TINY_QWEN_JOINT_DIM),
        prompt_embeds_mask=torch.ones(2, 3, dtype=torch.int64),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        guidance_scale=4.0,
        do_cfg=False,
        guidance_embeds=False,
        height=128,
        width=128,
        vae_scale_factor=16,
    )

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    torch.testing.assert_close(
        out["noise_pred_uncond"], torch.zeros_like(out["noise_pred_uncond"])
    )
    # Timestep normalized by 1000 (500 -> 0.5).
    torch.testing.assert_close(calls[0]["timestep"], torch.full((2,), 0.5))
    assert out["noise_pred"].shape == (2, _SEQ, TINY_QWEN_IN_CHANNELS)


def test_qwen_forward_step_runs_separate_cfg_with_uneven_seq_lengths() -> None:
    """do_cfg=True runs TWO forwards; cond/uncond may differ in sequence length."""
    transformer = build_tiny_transformer("qwen_image")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = QwenImageSamplingState(
        latents=torch.randn(2, _SEQ, TINY_QWEN_IN_CHANNELS),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, 5, TINY_QWEN_JOINT_DIM),
        prompt_embeds_mask=torch.ones(2, 5, dtype=torch.int64),
        # Unconditional branch has a DIFFERENT (shorter) text sequence — this is
        # exactly what batched CFG packing cannot handle, hence separate_cfg.
        negative_prompt_embeds=torch.randn(2, 2, TINY_QWEN_JOINT_DIM),
        negative_prompt_embeds_mask=torch.ones(2, 2, dtype=torch.int64),
        guidance_scale=4.0,
        do_cfg=True,
        guidance_embeds=False,
        height=128,
        width=128,
        vae_scale_factor=16,
    )

    out = model.forward_step(state, 0)

    # Two separate forwards (cond then uncond), not one batched call.
    assert len(calls) == 2
    assert calls[0]["encoder_hidden_states"].shape[1] == 5
    assert calls[1]["encoder_hidden_states"].shape[1] == 2
    cond, uncond = out["noise_pred_cond"], out["noise_pred_uncond"]
    assert not torch.allclose(cond, uncond)
    # Norm-preserving combine: the final pred carries the conditional branch's norm.
    comb = uncond + 4.0 * (cond - uncond)
    expected = comb * (
        torch.norm(cond, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True)
    )
    torch.testing.assert_close(out["noise_pred"], expected)


def test_qwen_replay_model_restores_state_without_a_pipeline() -> None:
    """The pipeline-less replay model rebuilds img_shapes and runs forward_step."""
    transformer = build_tiny_transformer("qwen_image")
    model = QwenImageReplayModel(
        transformer=transformer,
        scheduler=None,
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    replay_tensors = {
        "prompt_embeds": torch.randn(2, 3, TINY_QWEN_JOINT_DIM),
        "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.int64),
        # [B, num_steps] per-sample timesteps.
        "timesteps": torch.tensor([[500.0, 600.0], [500.0, 600.0]]),
    }
    batch_context = {
        "guidance_scale": 4.0,
        "cfg": False,
        "guidance_embeds": False,
        "height": 128,
        "width": 128,
        # vae_scale_factor 16 -> latent grid 128/16/2 = 4 per axis -> 16 tokens.
        "vae_scale_factor": 16,
    }
    latents = torch.randn(2, _SEQ, TINY_QWEN_IN_CHANNELS)

    state = model.restore_eval_state(replay_tensors, batch_context, latents, 0)
    out = model.forward_step(state, 0)
    assert out["noise_pred"].shape == (2, _SEQ, TINY_QWEN_IN_CHANNELS)


def test_qwen_prepare_replay_sets_the_rollout_timestep_grid() -> None:
    """The pipeline-less replay scheduler must carry the rollout's mu-shifted grid.

    Rollout derives ``mu`` from the packed latent length in ``prepare_sampling``;
    the generic replay loader knows nothing about it. The SDE log-prob math
    indexes ``scheduler.sigmas`` by timestep, so ``prepare_replay`` must rebuild
    the same grid from the run's fixed resolution.
    """
    import json
    from pathlib import Path

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.config.precision import RolePrecision
    from vrl.models.interfaces.runtime import ModelBuild

    fixture = Path("tests/models/steps/denoise/fixtures/scheduler_configs.json")
    config = json.loads(fixture.read_text())["qwen_image"]["config"]
    assert config["use_dynamic_shifting"] is True
    height, width, num_steps = 512, 512, 10

    rollout = _model(build_tiny_transformer("qwen_image"))
    rollout_scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    rollout.pipeline.scheduler = rollout_scheduler
    # 512 px -> 64x64 latent cells, 2x2 patch pack -> 1024 tokens.
    expected = rollout._set_dynamic_timesteps(num_steps, 1024, torch.device("cpu")).clone()

    replay = QwenImageReplayModel(
        transformer=build_tiny_transformer("qwen_image"),
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config),
        device=torch.device("cpu"),
    )
    replay.prepare_replay(
        ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype=torch.float32,
            family="qwen_image",
            precision=RolePrecision("fp32", "ieee", outer_autocast=False),
            sampling_config={"height": height, "width": width, "num_steps": num_steps},
        )
    )

    torch.testing.assert_close(replay.scheduler.timesteps, expected)
    torch.testing.assert_close(replay.scheduler.sigmas, rollout_scheduler.sigmas)
