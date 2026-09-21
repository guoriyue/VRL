"""Real-inference tests for the Qwen-Image-2.1 single-stream backbone wrapper.

A real (tiny, cache-free) ``QwenImage21Transformer2DModel`` runs through
``forward_step``; the joint ``img_mask`` construction, the no-CFG default
(one forward), the separate-branch true CFG with a plain linear combine, and the
pipeline-less replay restore are verified against the model's OWN outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN21_CONTEXT_DIM,
    TINY_QWEN21_IN_CHANNELS,
    build_tiny_qwen_image_21_transformer,
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
    transformer = build_tiny_qwen_image_21_transformer()
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
    transformer = build_tiny_qwen_image_21_transformer()
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
    transformer = build_tiny_qwen_image_21_transformer()
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
