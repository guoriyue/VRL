from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion_fixtures import FakeDiffusionTransformer, fake_transformer_output
from vrl.models.diffusion.cosmos.predict2.model import (
    CosmosPredict2Model,
    CosmosPredict2SamplingState,
)


def test_cosmos_predict2_forward_step_matches_previous_inline_cfg_math() -> None:
    transformer = FakeDiffusionTransformer()
    model = CosmosPredict2Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = _state(do_cfg=True)

    actual = model.forward_step(state, 0)
    expected = _cosmos2_reference(state, 0)

    assert len(transformer.calls) == 2
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def _state(*, do_cfg: bool) -> CosmosPredict2SamplingState:
    latents = torch.arange(1.0, 17.0).view(2, 1, 2, 2, 2)
    indicator = torch.tensor([[[[[1.0]], [[0.0]]]]])
    mask = torch.tensor([[[[[0.5]], [[1.0]]]]])
    return CosmosPredict2SamplingState(
        latents=latents,
        timesteps=torch.tensor([0.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([3.0])),
        prompt_embeds=torch.ones(2, 3, 4),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        guidance_scale=2.0,
        do_cfg=do_cfg,
        init_latents=torch.full_like(latents[:1], 5.0),
        cond_mask=mask,
        uncond_mask=mask * 0.25,
        padding_mask=torch.zeros(1, 1, 4, 4),
        cond_indicator=indicator,
        uncond_indicator=1 - indicator,
        fps=16,
        seed=0,
        sigma_conditioning=0.125,
    )


def _cosmos2_reference(
    state: CosmosPredict2SamplingState,
    step_idx: int,
) -> dict[str, torch.Tensor]:
    batch = state.latents.shape[0]
    dtype = state.prompt_embeds.dtype
    sigma = state.scheduler.sigmas[step_idx]
    current_t = sigma / (sigma + 1)
    c_in = 1 - current_t
    c_skip = 1 - current_t
    c_out = -current_t
    timestep = current_t.view(1, 1, 1, 1, 1).expand(
        batch,
        -1,
        state.latents.size(2),
        -1,
        -1,
    )
    t_conditioning = torch.tensor(state.sigma_conditioning).view(
        1,
        1,
        1,
        1,
        1,
    ).expand_as(timestep)
    init = state.init_latents.expand_as(state.latents)
    cond_indicator = state.cond_indicator.expand_as(state.latents)
    uncond_indicator = state.uncond_indicator.expand_as(state.latents)
    cond_mask = state.cond_mask.expand_as(state.latents)
    uncond_mask = state.uncond_mask.expand_as(state.latents)

    cond_latent = cond_indicator * init + (1 - cond_indicator) * (state.latents * c_in)
    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
    raw_cond = fake_transformer_output(
        hidden_states=cond_latent.to(dtype),
        timestep=cond_timestep.to(dtype),
        encoder_hidden_states=state.prompt_embeds,
        fps=state.fps,
        condition_mask=cond_mask,
        padding_mask=state.padding_mask,
        return_dict=False,
    )
    cond = (c_skip * state.latents + c_out * raw_cond.float()).to(dtype)
    cond = cond_indicator * init + (1 - cond_indicator) * cond

    uncond_latent = (
        uncond_indicator * init + (1 - uncond_indicator) * (state.latents * c_in)
    )
    uncond_timestep = uncond_indicator * t_conditioning + (
        1 - uncond_indicator
    ) * timestep
    raw_uncond = fake_transformer_output(
        hidden_states=uncond_latent.to(dtype),
        timestep=uncond_timestep.to(dtype),
        encoder_hidden_states=state.negative_prompt_embeds,
        fps=state.fps,
        condition_mask=uncond_mask,
        padding_mask=state.padding_mask,
        return_dict=False,
    )
    uncond = (c_skip * state.latents + c_out * raw_uncond.float()).to(dtype)
    uncond = uncond_indicator * init + (1 - uncond_indicator) * uncond

    combined = cond + state.guidance_scale * (cond - uncond)
    return {
        "noise_pred": (state.latents - combined) / sigma,
        "noise_pred_cond": cond,
        "noise_pred_uncond": uncond,
    }
