from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion_fixtures import FakeDiffusionTransformer, fake_transformer_output
from vrl.models.diffusion.wan_2_1.model import (
    WanI2VDiffusersModel,
    WanI2VSamplingState,
    WanT2VDiffusersModel,
    WanT2VSamplingState,
)


def test_wan_forward_step_matches_previous_inline_cfg_math() -> None:
    transformer = FakeDiffusionTransformer()
    model = WanT2VDiffusersModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = WanT2VSamplingState(
        latents=torch.arange(16.0).view(2, 1, 2, 2, 2),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, 3, 4),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        guidance_scale=3.0,
        do_cfg=True,
        seed=0,
    )

    actual = model.forward_step(state, 0)
    expected = _wan_reference(state)

    assert len(transformer.calls) == 1
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def test_wan_i2v_forward_step_threads_condition_and_image_embeds() -> None:
    transformer = _FakeWanI2VTransformer()
    model = WanI2VDiffusersModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = WanI2VSamplingState(
        latents=torch.arange(8.0).view(2, 1, 1, 2, 2),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, 3, 4),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        image_embeds=torch.full((2, 2, 3), 0.5),
        condition=torch.full((2, 1, 1, 2, 2), 10.0),
        guidance_scale=2.0,
        do_cfg=True,
        seed=0,
    )

    actual = model.forward_step(state, 0)
    expected = _wan_i2v_reference(state)

    assert len(transformer.calls) == 1
    call = transformer.calls[0]
    assert call["hidden_states"].shape == (4, 2, 1, 2, 2)
    assert "encoder_hidden_states_image" in call
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def test_wan_i2v_replay_state_roundtrip_keeps_conditioning_tensors() -> None:
    transformer = _FakeWanI2VTransformer()
    model = WanI2VDiffusersModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = WanI2VSamplingState(
        latents=torch.zeros(2, 1, 1, 2, 2),
        timesteps=torch.tensor([9.0, 7.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, 3, 4),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        image_embeds=torch.full((2, 2, 3), 0.5),
        condition=torch.full((2, 1, 1, 2, 2), 10.0),
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


def _wan_reference(state: WanT2VSamplingState) -> dict[str, torch.Tensor]:
    td = state.prompt_embeds.dtype
    latent_input = state.latents.to(td)
    t = state.timesteps[0]
    timestep_batch = t.expand(latent_input.shape[0]) if t.ndim == 0 else t
    combined = fake_transformer_output(
        hidden_states=torch.cat([latent_input, latent_input], dim=0),
        timestep=torch.cat([timestep_batch, timestep_batch], dim=0),
        encoder_hidden_states=torch.cat(
            [state.negative_prompt_embeds, state.prompt_embeds],
            dim=0,
        ),
        return_dict=False,
    )
    uncond, cond = combined.chunk(2, dim=0)
    return {
        "noise_pred": uncond + state.guidance_scale * (cond - uncond),
        "noise_pred_cond": cond.to(td),
        "noise_pred_uncond": uncond.to(td),
    }


class _FakeWanI2VTransformer(FakeDiffusionTransformer):
    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (fake_transformer_output(**kwargs)[:, :1],)


def _wan_i2v_reference(state: WanI2VSamplingState) -> dict[str, torch.Tensor]:
    td = state.prompt_embeds.dtype
    latent_input = torch.cat([state.latents.to(td), state.condition.to(td)], dim=1)
    t = state.timesteps[0]
    timestep_batch = t.expand(state.latents.shape[0]) if t.ndim == 0 else t
    combined = fake_transformer_output(
        hidden_states=torch.cat([latent_input, latent_input], dim=0),
        timestep=torch.cat([timestep_batch, timestep_batch], dim=0),
        encoder_hidden_states=torch.cat(
            [state.negative_prompt_embeds, state.prompt_embeds],
            dim=0,
        ),
        encoder_hidden_states_image=torch.cat(
            [state.image_embeds, state.image_embeds],
            dim=0,
        ),
        return_dict=False,
    )[:, :1]
    uncond, cond = combined.chunk(2, dim=0)
    return {
        "noise_pred": uncond + state.guidance_scale * (cond - uncond),
        "noise_pred_cond": cond.to(td),
        "noise_pred_uncond": uncond.to(td),
    }
