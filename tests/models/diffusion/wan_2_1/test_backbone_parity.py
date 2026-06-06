"""Real-inference tests for the Wan T2V/I2V CFG backbone wrapper.

No fake transformer: these run a real (tiny, cache-free) ``WanTransformer3DModel``
through ``forward_step`` and verify the wrapper's classifier-free-guidance
behavior against the model's OWN real outputs — never a hand-re-derived formula.

What is pinned, all from real inference:
- batched CFG issues exactly ONE transformer forward (un/cond concatenated);
- ``noise_pred == uncond + scale * (cond - uncond)`` holds on the real branches,
  so a sign/formula error in the wrapper is caught;
- ``cond != uncond`` — the real model actually responds to the conditioning, which
  a closed-form fake cannot demonstrate;
- I2V threads the condition latent into the channel axis and the image embeds into
  the image cross-attention branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tests.models.diffusion.fixtures import (
    build_tiny_wan_i2v_transformer,
    build_tiny_wan_transformer,
    record_forward_calls,
)
from vrl.models.diffusion.wan_2_1.model import (
    WanI2VDiffusersModel,
    WanI2VSamplingState,
    WanT2VDiffusersModel,
    WanT2VSamplingState,
)

_TEXT_LEN = 3
_TEXT_DIM = 16
_LATENT_SHAPE = (2, 4, 1, 4, 4)


def _model(cls: type, transformer: torch.nn.Module) -> Any:
    return cls(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )


def test_wan_t2v_forward_step_runs_real_batched_cfg() -> None:
    """Checks Wan T2V forward step runs real batched CFG."""
    transformer = build_tiny_wan_transformer()
    calls = record_forward_calls(transformer)
    model = _model(WanT2VDiffusersModel, transformer)
    state = WanT2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        guidance_scale=3.0,
        do_cfg=True,
        seed=0,
    )

    out = model.forward_step(state, 0)

    # One batched forward: un/cond concatenated along batch -> 2*B rows.
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == 2 * _LATENT_SHAPE[0]
    # CFG combination verified on the real model's own branch outputs.
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 3.0 * (cond - uncond))
    assert out["noise_pred"].shape == _LATENT_SHAPE
    # The real transformer genuinely responds to the prompt (not a no-op fake).
    assert not torch.allclose(cond, uncond)


def test_wan_i2v_forward_step_threads_condition_and_image_embeds() -> None:
    """Checks Wan I2V forward step threads condition and image embeds."""
    transformer = build_tiny_wan_i2v_transformer()
    calls = record_forward_calls(transformer)
    model = _model(WanI2VDiffusersModel, transformer)
    state = WanI2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        image_embeds=torch.randn(2, 2, _TEXT_DIM),
        condition=torch.randn(_LATENT_SHAPE),
        guidance_scale=2.0,
        do_cfg=True,
        seed=0,
    )

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    call = calls[0]
    # Condition latent is concatenated into the channel axis (4 latent + 4 cond),
    # and batched CFG doubles the batch.
    assert call["hidden_states"].shape == (2 * _LATENT_SHAPE[0], 8, 1, 4, 4)
    assert call["encoder_hidden_states_image"] is not None
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 2.0 * (cond - uncond))
    assert out["noise_pred"].shape == _LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_wan_i2v_replay_state_roundtrip_keeps_conditioning_tensors() -> None:
    # Pure state plumbing (no forward): export -> restore must preserve the
    # conditioning tensors and slice the per-step timestep.
    """Checks Wan I2V replay state roundtrip keeps conditioning tensors."""
    model = _model(WanI2VDiffusersModel, build_tiny_wan_i2v_transformer())
    state = WanI2VSamplingState(
        latents=torch.zeros(_LATENT_SHAPE),
        timesteps=torch.tensor([9.0, 7.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.zeros(2, _TEXT_LEN, _TEXT_DIM),
        image_embeds=torch.full((2, 2, _TEXT_DIM), 0.5),
        condition=torch.full(_LATENT_SHAPE, 10.0),
        guidance_scale=2.0,
        do_cfg=True,
        seed=0,
    )
    replay = model.export_replay_tensors(state)
    replay["timesteps"] = torch.tensor([[9.0, 7.0], [9.0, 7.0]])

    restored = model.restore_eval_state(
        replay,
        model.export_batch_context(state),
        state.latents,
        1,
    )

    torch.testing.assert_close(restored.condition, state.condition)
    torch.testing.assert_close(restored.image_embeds, state.image_embeds)
    torch.testing.assert_close(restored.timesteps, torch.tensor([[7.0, 7.0]]))
