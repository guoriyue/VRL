"""Tests for runtime model contracts."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.diffusion import DiffusionModelBase
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeBundle,
    RuntimeModel,
    require_runtime_model,
)


class _MinimalRuntimeModel:
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class _ReplayOnlyModel:
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class _DiffusionModelBaseStub(DiffusionModelBase):
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


def test_runtime_model_protocol_accepts_minimal_shape() -> None:
    assert isinstance(_MinimalRuntimeModel(), RuntimeModel)


def test_require_runtime_model_reports_missing_state_loader() -> None:
    with pytest.raises(TypeError, match="missing: load_trainable_state"):
        require_runtime_model(_ReplayOnlyModel(), owner="test.model")


def test_runtime_bundle_exposes_model_contract() -> None:
    model = _MinimalRuntimeModel()
    bundle = RuntimeBundle(
        model=model,
        trainable_modules={},
        scheduler=None,
        backend_handle=None,
    )

    assert bundle.model is model


def test_diffusion_load_trainable_state_accepts_trainable_only_payload() -> None:
    model = _DiffusionModelBaseStub()
    new_weight = torch.full_like(model.transformer.weight, 3.0)

    model.load_trainable_state({"transformer.weight": new_weight})

    assert torch.equal(model.transformer.weight, new_weight)


def test_diffusion_load_trainable_state_rejects_frozen_payload() -> None:
    model = _DiffusionModelBaseStub()

    with pytest.raises(ValueError, match="exactly trainable"):
        model.load_trainable_state(
            {
                "transformer.weight": torch.ones_like(model.transformer.weight),
                "transformer.bias": torch.ones_like(model.transformer.bias),
            },
        )
