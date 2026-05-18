from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion_fixtures import FakeDiffusionTransformer, fake_transformer_output
from vrl.models.diffusion.sd3_5.model import SD3_5Model, SD3SamplingState


def test_sd3_forward_step_matches_previous_inline_cfg_math() -> None:
    transformer = FakeDiffusionTransformer()
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    state = SD3SamplingState(
        latents=torch.arange(8.0).view(2, 1, 2, 2),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, 3, 4),
        pooled_prompt_embeds=torch.full((2, 4), 2.0),
        negative_prompt_embeds=torch.zeros(2, 3, 4),
        negative_pooled_prompt_embeds=torch.full((2, 4), -1.0),
        guidance_scale=3.0,
        do_cfg=True,
        seed=0,
    )

    actual = model.forward_step(state, 0)
    expected = _sd3_reference(state)

    assert len(transformer.calls) == 1
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def test_sd3_forward_step_matches_previous_inline_single_branch_math() -> None:
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=FakeDiffusionTransformer(), device="cpu"),
        device=torch.device("cpu"),
    )
    state = SD3SamplingState(
        latents=torch.arange(8.0).view(2, 1, 2, 2),
        timesteps=torch.tensor([[5.0, 6.0]]),
        scheduler=None,
        prompt_embeds=torch.ones(2, 3, 4),
        pooled_prompt_embeds=torch.full((2, 4), 2.0),
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
        seed=0,
    )

    actual = model.forward_step(state, 0)
    expected = _sd3_reference(state)

    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def _sd3_reference(state: SD3SamplingState) -> dict[str, torch.Tensor]:
    td = state.prompt_embeds.dtype
    latent_input = state.latents.to(td)
    t = state.timesteps[0]
    timestep_batch = t.expand(latent_input.shape[0]) if t.ndim == 0 else t
    if state.do_cfg:
        combined = fake_transformer_output(
            hidden_states=torch.cat([latent_input, latent_input], dim=0),
            timestep=torch.cat([timestep_batch, timestep_batch], dim=0),
            encoder_hidden_states=torch.cat(
                [state.negative_prompt_embeds, state.prompt_embeds],
                dim=0,
            ),
            pooled_projections=torch.cat(
                [state.negative_pooled_prompt_embeds, state.pooled_prompt_embeds],
                dim=0,
            ),
            return_dict=False,
        )
        uncond, cond = combined.chunk(2, dim=0)
        noise = uncond + state.guidance_scale * (cond - uncond)
    else:
        cond = fake_transformer_output(
            hidden_states=latent_input,
            timestep=timestep_batch,
            encoder_hidden_states=state.prompt_embeds,
            pooled_projections=state.pooled_prompt_embeds,
            return_dict=False,
        ).to(td)
        uncond = torch.zeros_like(cond)
        noise = cond
    return {
        "noise_pred": noise,
        "noise_pred_cond": cond.to(td),
        "noise_pred_uncond": uncond.to(td),
    }
