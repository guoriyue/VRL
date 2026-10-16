"""The layered SFT recipe's forward adapter and placement step."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vrl.models.host_streaming import is_host_streamed
from vrl.scripts.families.qwen_image_layered.train_sft import (
    layered_forward,
    place_trainable_roots,
)


def test_forward_hands_the_cached_conditioning_to_predict_velocity() -> None:
    seen: dict = {}

    def predict_velocity(latents, timesteps, **kwargs):
        seen.update(kwargs, latents=latents, timesteps=timesteps)
        return latents * 2

    model = SimpleNamespace(predict_velocity=predict_velocity)
    noisy, timesteps = torch.ones(1, 6, 4), torch.tensor([450.0])
    conditioning = {
        "condition_latents": torch.zeros(1, 2, 4),
        "prompt_embeds": torch.zeros(1, 3, 5),
        "prompt_embeds_mask": None,
        "layers": torch.tensor(2),
        "height": 32,
        "width": 32,
        "scene": "scene_a",
    }

    out = layered_forward(model, noisy, timesteps, conditioning)

    torch.testing.assert_close(out, noisy * 2)
    assert seen["layers"] == 2 and isinstance(seen["layers"], int)
    assert (seen["height"], seen["width"]) == (32, 32)
    assert seen["prompt_embeds_mask"] is None and seen["latents"] is noisy
    assert "scene" not in seen


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        for block in self.blocks:
            block.requires_grad_(False)
        self.head = nn.Linear(2, 2)


def test_placement_streams_frozen_blocks_only_when_the_actor_asks() -> None:
    streamed, resident = _Policy(), _Policy()

    place_trainable_roots(
        SimpleNamespace(trainable_modules={"transformer": streamed}),
        SimpleNamespace(stream_frozen_blocks=True),
        device="cpu",
    )
    place_trainable_roots(
        SimpleNamespace(trainable_modules={"transformer": resident}),
        SimpleNamespace(stream_frozen_blocks=None),
        device="cpu",
    )

    assert is_host_streamed(streamed) and not is_host_streamed(resident)
    assert all(parameter.device.type == "cpu" for parameter in streamed.parameters())
