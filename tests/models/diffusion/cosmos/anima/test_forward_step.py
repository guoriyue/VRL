from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tests.models.diffusion.fixtures import stamp_test_contract
from vrl.models.diffusion.cosmos.anima.model import AnimaModel, AnimaSamplingState


class _ConstantAnimaTransformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=1)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del padding_mask, return_dict
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
            }
        )
        value = encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1, 1)
        return (torch.ones_like(hidden_states) * value,)


def test_anima_forward_step_uses_const_flow_derivative_contract() -> None:
    """Checks Anima forward step uses const flow derivative contract."""
    transformer = _ConstantAnimaTransformer()
    model = AnimaModel(
        transformer=transformer,
        text_encoder=None,
        llm_adapter=torch.nn.Identity(),
        vae=None,
        scheduler=None,
        image_processor=None,
        qwen_tokenizer=None,
        t5_tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    stamp_test_contract(model)
    latents = torch.full((1, 1, 1, 1, 1), 5.0)
    state = AnimaSamplingState(
        latents=latents,
        timesteps=torch.tensor([700.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.7])),
        prompt_embeds=torch.ones(1, 1, 1),
        negative_prompt_embeds=torch.zeros(1, 1, 1),
        guidance_scale=2.0,
        do_cfg=True,
        padding_mask=torch.zeros(1, 1, 8, 8),
    )

    out = model.forward_step(state, 0)

    assert out["noise_pred_cond"] == torch.ones_like(latents)
    assert out["noise_pred_uncond"] == torch.zeros_like(latents)
    assert out["noise_pred"] == torch.full_like(latents, 2.0)
    assert torch.equal(transformer.calls[0]["hidden_states"], latents)
    assert transformer.calls[0]["timestep"] == torch.tensor([0.7])
