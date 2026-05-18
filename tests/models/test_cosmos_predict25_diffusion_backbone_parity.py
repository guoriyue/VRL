from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion_fixtures import FakeDiffusionTransformer, fake_transformer_output
from vrl.models.diffusion.cosmos.predict2_5.model import (
    CosmosPredict25Model,
    CosmosPredict25SamplingState,
)


def test_cosmos_predict25_forward_step_matches_previous_inline_cfg_math() -> None:
    transformer = FakeDiffusionTransformer()
    model = CosmosPredict25Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = _state()

    actual = model.forward_step(state, 0)
    expected = _cosmos25_reference(state, 0)

    assert len(transformer.calls) == 2
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def _state() -> CosmosPredict25SamplingState:
    latents = torch.arange(1.0, 17.0).view(2, 1, 2, 2, 2)
    cond_mask = torch.tensor([[[[[1.0]], [[0.0]]]]])
    cond_indicator = torch.tensor([[[[[1.0]], [[0.0]]]]])
    return CosmosPredict25SamplingState(
        latents=latents,
        timesteps=torch.tensor([0.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.75])),
        prompt_embeds=torch.ones(2, 3, 4),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        guidance_scale=2.0,
        do_cfg=True,
        cond_latent=torch.full_like(latents, 2.0),
        cond_mask=cond_mask,
        cond_indicator=cond_indicator,
        padding_mask=torch.zeros(1, 1, 4, 4),
        seed=0,
        height=4,
        width=4,
        num_frames=2,
        fps=16,
        conditional_frame_timestep=0.1,
    )


def _cosmos25_reference(
    state: CosmosPredict25SamplingState,
    step_idx: int,
) -> dict[str, torch.Tensor]:
    dtype = state.prompt_embeds.dtype
    sigma_t = torch.tensor(state.scheduler.sigmas[step_idx].item()).unsqueeze(0).to(
        dtype=dtype,
    )
    cond_timestep = torch.ones_like(state.cond_indicator) * (
        state.conditional_frame_timestep
    )
    in_latents = (
        state.cond_mask * state.cond_latent + (1 - state.cond_mask) * state.latents
    ).to(dtype)
    in_timestep = state.cond_indicator * cond_timestep + (
        1 - state.cond_indicator
    ) * sigma_t
    raw_cond = fake_transformer_output(
        hidden_states=in_latents,
        condition_mask=state.cond_mask.to(dtype),
        timestep=in_timestep.to(dtype),
        encoder_hidden_states=state.prompt_embeds,
        padding_mask=state.padding_mask,
        return_dict=False,
    )
    gt_velocity = (state.latents - state.cond_latent) * state.cond_mask
    cond = gt_velocity + raw_cond * (1 - state.cond_mask)

    raw_uncond = fake_transformer_output(
        hidden_states=in_latents,
        condition_mask=state.cond_mask.to(dtype),
        timestep=in_timestep.to(dtype),
        encoder_hidden_states=state.negative_prompt_embeds,
        padding_mask=state.padding_mask,
        return_dict=False,
    )
    uncond = gt_velocity + raw_uncond * (1 - state.cond_mask)
    return {
        "noise_pred": cond + state.guidance_scale * (cond - uncond),
        "noise_pred_cond": cond,
        "noise_pred_uncond": uncond,
    }
