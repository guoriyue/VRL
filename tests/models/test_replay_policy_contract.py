"""Tests for trainer replay policy contracts."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.interfaces import DiffusionPolicy, ReplayPolicy


class _ReplayPolicy:
    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        del batch, timestep_idx
        return {}

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class _DiffusionReplayPolicy(DiffusionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 1, bias=True)
        self.transformer.bias.requires_grad_(False)

    def encode_prompt(self, prompt, negative_prompt=None, **kwargs):
        del prompt, negative_prompt, kwargs
        return {}

    def prepare_sampling(self, request, encoded, **kwargs):
        del request, encoded, kwargs
        return None

    def forward_step(self, state, step_idx):
        del state, step_idx
        return {}

    def decode_latents(self, latents):
        return latents

    @classmethod
    def from_spec(cls, spec):
        del spec
        return cls()

    def apply_lora(self, spec):
        del spec

    def enable_full_finetune(self):
        return None

    def torch_compile_transformer(self, mode: str):
        del mode

    def set_num_steps(self, n: int):
        del n

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return None

    @property
    def backend_handle(self) -> Any:
        return None

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        yield


def test_replay_policy_protocol_accepts_minimal_shape() -> None:
    assert isinstance(_ReplayPolicy(), ReplayPolicy)


def test_diffusion_load_trainable_state_accepts_trainable_only_payload() -> None:
    policy = _DiffusionReplayPolicy()
    new_weight = torch.full_like(policy.transformer.weight, 3.0)

    policy.load_trainable_state({"transformer.weight": new_weight})

    assert torch.equal(policy.transformer.weight, new_weight)


def test_diffusion_load_trainable_state_rejects_frozen_payload() -> None:
    policy = _DiffusionReplayPolicy()

    with pytest.raises(ValueError, match="exactly trainable"):
        policy.load_trainable_state(
            {
                "transformer.weight": torch.ones_like(policy.transformer.weight),
                "transformer.bias": torch.ones_like(policy.transformer.bias),
            },
        )
