"""Parsed sampling parameters for full-sequence denoise executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.generation.steps.denoise.config import DenoiseSDEParams
from vrl.generation.steps.denoise.teacache import TeaCacheConfig
from vrl.generation.types import DenoiseRequest


@dataclass(frozen=True, slots=True)
class DenoiseSamplingParams:
    """Parsed denoising sampling fields for one generation request."""

    model_request: DenoiseRequest
    max_sequence_length: int | None
    sde: DenoiseSDEParams
    denoise_mode: str
    teacache: TeaCacheConfig | None = None
    # Re-parses use request-owned randomness so split batches and retries
    # select the same window, including requests without a sampling seed.
    sde_window: tuple[int, int] | None = None

    def text_encode_kwargs(self) -> dict[str, Any]:
        """Build shared prompt-encoder knobs without inventing a text length."""

        kwargs: dict[str, Any] = {
            "guidance_scale": self.model_request.guidance_scale,
        }
        if self.max_sequence_length is not None:
            kwargs["max_sequence_length"] = self.max_sequence_length
        return kwargs


__all__ = [
    "DenoiseSamplingParams",
]
