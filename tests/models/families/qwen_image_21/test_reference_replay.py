"""Reference prefixes survive SDE rollout/replay and remain conditioning, not actions."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN21_CONTEXT_DIM,
    TINY_QWEN21_IN_CHANNELS,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.qwen_image_21.model import (
    QwenImage21Model,
    QwenImage21ReplayModel,
    QwenImage21SamplingState,
)


def test_two_reference_cfg_replay_matches_rollout_logprob_and_backpropagates() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    transformer = build_tiny_transformer("qwen_image_21")
    calls = record_forward_calls(transformer)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    model = QwenImage21Model(
        pipeline=SimpleNamespace(transformer=transformer), device=torch.device("cpu")
    )
    stamp_model_precision(model)
    # Two references with distinct geometries: 4 and 8 tokens, occupying 3 VLM slots.
    refs = torch.randn(2, 12, TINY_QWEN21_IN_CHANNELS)
    state = QwenImage21SamplingState(
        latents=torch.randn(2, 16, TINY_QWEN21_IN_CHANNELS),
        timesteps=scheduler.timesteps,
        scheduler=scheduler,
        prompt_embeds=torch.randn(2, 7, TINY_QWEN21_CONTEXT_DIM),
        prompt_embeds_mask=None,
        image_pad_mask=torch.tensor([[False, True, False, True, True, False, False]] * 2),
        negative_prompt_embeds=torch.randn(2, 5, TINY_QWEN21_CONTEXT_DIM),
        negative_prompt_embeds_mask=None,
        negative_image_pad_mask=torch.tensor([[True, False, True, True, False]] * 2),
        guidance_scale=2.0,
        do_cfg=True,
        height=64,
        width=64,
        vae_scale_factor=16,
        reference_latents=refs,
        reference_shapes=((1, 2, 2), (1, 2, 4)),
    )
    pred = model.forward_step(state, 0)["noise_pred"]
    assert pred.shape == state.latents.shape
    assert calls[0]["img_shapes"] == [[(1, 2, 2), (1, 2, 4), (1, 4, 4)]] * 2
    torch.testing.assert_close(calls[0]["hidden_states"][:, :12], refs)
    # Compare to the actual transformer with the reference pipeline's target-tail slice.
    direct = transformer(**calls[0])[0][:, -16:]
    negative = transformer(**calls[1])[0][:, -16:]
    torch.testing.assert_close(pred, negative + 2 * (direct - negative))
    step = sde_step_with_logprob(scheduler, pred, state.timesteps[:1], state.latents, step_index=0)
    replay = QwenImage21ReplayModel(
        transformer=transformer, scheduler=scheduler, device=torch.device("cpu")
    )
    stamp_model_precision(replay)
    tensors = {
        **model.export_replay_tensors(state),
        "timesteps": state.timesteps.unsqueeze(0).expand(2, -1),
    }
    restored = replay.restore_eval_state(
        tensors, model.export_batch_context(state), state.latents, 0
    )
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
    assert step.prev_sample.shape == state.latents.shape
    torch.testing.assert_close(restored.reference_latents, refs)
    # The references have a real effect; a lost prefix cannot pass parity accidentally.
    changed = replay.forward_step(replace(restored, reference_latents=refs + 2), 0)["noise_pred"]
    assert not torch.allclose(changed, replay_pred)
    (-replay_step.log_prob.mean()).backward()
    gradients = [p.grad for p in transformer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
