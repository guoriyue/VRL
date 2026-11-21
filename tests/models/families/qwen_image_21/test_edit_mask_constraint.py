"""Outside an edit mask the transformer reads the re-noised reference, at rollout and replay alike."""

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
from vrl.models.families.qwen_image_21.model import (
    QwenImage21Model,
    QwenImage21PipelineModel,
    QwenImage21SamplingState,
)


def test_held_tokens_reach_the_transformer_as_the_noised_reference_and_replay_agrees() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    transformer = build_tiny_transformer("qwen_image_21")
    calls = record_forward_calls(transformer)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    model = QwenImage21PipelineModel(
        pipeline=SimpleNamespace(transformer=transformer), device=torch.device("cpu")
    )
    stamp_model_precision(model)
    reference = torch.randn(2, 16, TINY_QWEN21_IN_CHANNELS)
    noise = torch.randn(2, 16, TINY_QWEN21_IN_CHANNELS)
    edit = torch.zeros(2, 16, dtype=torch.bool)
    edit[:, 5:9] = True
    state = QwenImage21SamplingState(
        latents=torch.randn(2, 16, TINY_QWEN21_IN_CHANNELS),
        timesteps=scheduler.timesteps,
        scheduler=scheduler,
        prompt_embeds=torch.randn(2, 6, TINY_QWEN21_CONTEXT_DIM),
        prompt_embeds_mask=None,
        image_pad_mask=torch.tensor([[False, True, True, True, True, False]] * 2),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_image_pad_mask=None,
        guidance_scale=1.0,
        do_cfg=False,
        height=64,
        width=64,
        vae_scale_factor=16,
        reference_latents=reference,
        reference_shapes=((1, 4, 4),),
        edit_mask=edit,
        edit_noise=noise,
    )
    pred = model.forward_step(state, 1)["noise_pred"]
    sigma = scheduler.sigmas[1]
    target = calls[0]["hidden_states"][:, 16:]
    torch.testing.assert_close(target[:, 5:9], state.latents[:, 5:9])
    held = (1 - sigma) * reference + sigma * noise
    torch.testing.assert_close(target[:, :5], held[:, :5])
    torch.testing.assert_close(target[:, 9:], held[:, 9:])
    # What the path holds outside the mask never reaches the model.
    drifted = state.latents.clone()
    drifted[:, :5] += 3
    torch.testing.assert_close(
        model.forward_step(replace(state, latents=drifted), 1)["noise_pred"], pred
    )

    replay = QwenImage21Model(
        transformer=transformer, scheduler=scheduler, device=torch.device("cpu")
    )
    stamp_model_precision(replay)
    tensors = {
        **model.export_replay_tensors(state),
        "timesteps": state.timesteps.unsqueeze(0).expand(2, -1),
    }
    restored = replay.restore_eval_state(
        tensors, model.export_batch_context(state), state.latents, 1
    )
    torch.testing.assert_close(replay.forward_step(restored, 0)["noise_pred"], pred)
