"""Tiny-real NextStep-1 fixtures.

The upstream ``gen_pipeline`` / ``nextstep_model`` packages are not repository
dependencies, so the pipeline and ``unpatchify`` stay package-boundary stand-ins
(mirroring the Janus fixtures). The VAE is a genuine f8 ``AutoencoderKL`` so the
decode geometry a test asserts on is computed by diffusers, not declared.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import build_tiny_autoencoder_kl
from vrl.models.families.nextstep_1.model import NextStep1Model

# NextStep-1-f8ch16-Tokenizer: 8x spatial downsampling, 16 latent channels.
NEXTSTEP_VAE_DOWNSAMPLES = 3
NEXTSTEP_LATENT_CHANNELS = 16


def install_stub_nextstep_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Register a ``gen_pipeline.NextStepPipeline`` stand-in and return its recorded kwargs.

    ``NextStep1Model._load_pipeline`` imports the upstream package by name; the
    stand-in only records what the loader passed it (paths, revisions).
    """

    pipeline_kwargs: dict[str, Any] = {}

    class NextStepPipeline:
        def __init__(self, **kwargs: Any) -> None:
            pipeline_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "gen_pipeline",
        SimpleNamespace(NextStepPipeline=NextStepPipeline),
    )
    return pipeline_kwargs


def build_tiny_nextstep_vae(*, seed: int = 0) -> Any:
    """A real ``AutoencoderKL`` with the f8ch16 tokenizer's geometry."""

    return build_tiny_autoencoder_kl(
        seed=seed,
        downsamples=NEXTSTEP_VAE_DOWNSAMPLES,
        latent_channels=NEXTSTEP_LATENT_CHANNELS,
    )


def build_decode_only_nextstep_model(*, vae: Any) -> NextStep1Model:
    """A ``NextStep1Model`` holding only what ``decode_image_tokens`` reads.

    ``unpatchify`` belongs to the upstream ``nextstep_model`` package and stays a
    stand-in that returns a latent with the VAE's own channel count; the decode
    itself runs through the real ``vae``.
    """

    model = object.__new__(NextStep1Model)
    torch.nn.Module.__init__(model)
    latent_channels = int(vae.config.latent_channels)

    class UpstreamModel:
        @staticmethod
        def unpatchify(tokens: torch.Tensor, *, h: int, w: int) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], latent_channels, h, w)

    object.__setattr__(
        model,
        "_pipeline",
        SimpleNamespace(model=UpstreamModel(), scaling_factor=1.0, shift_factor=0.0),
    )
    model.vae = vae
    return model


__all__ = [
    "NEXTSTEP_LATENT_CHANNELS",
    "NEXTSTEP_VAE_DOWNSAMPLES",
    "build_decode_only_nextstep_model",
    "build_tiny_nextstep_vae",
    "install_stub_nextstep_pipeline",
]
