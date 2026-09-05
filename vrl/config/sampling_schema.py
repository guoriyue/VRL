"""Lightweight public schemas for the YAML ``sampling`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery reject family-inapplicable knobs
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_validator

from vrl.config.base import ConfigBase


class SamplingSection(ConfigBase):
    """Base for family-selected public sampling configuration."""


class DenoiseImageSamplingSection(SamplingSection):
    """Sampling inputs shared by denoise image generators."""

    guidance_scale: Any = None
    height: Any = None
    num_steps: Any = None
    width: Any = None


class TextEncodedImageSamplingSection(DenoiseImageSamplingSection):
    """Image sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class VideoSamplingSection(DenoiseImageSamplingSection):
    """Sampling inputs shared by denoise video generators."""

    fps: Any = None
    num_frames: Any = None


class TextEncodedVideoSamplingSection(VideoSamplingSection):
    """Video sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class EchoSamplingSection(VideoSamplingSection):
    """Echo DMD sampling, whose checkpoint has guidance baked in."""

    guidance_scale: Literal[1.0] | None = None


class MagiSamplingSection(SamplingSection):
    """Inputs mapped into MAGI-1's isolated official inference runtime."""

    fps: Any = None
    height: Any = None
    num_frames: Any = None
    num_steps: Any = None
    width: Any = None


class ARSamplingSection(SamplingSection):
    """Request-local scheduler controls shared by autoregressive generators."""

    ar_scheduler_batch_size: int | None = None

    @field_validator("ar_scheduler_batch_size", mode="before")
    @classmethod
    def _validate_ar_scheduler_batch_size(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "sampling.ar_scheduler_batch_size must be a positive integer or null",
            )
        return value


class TextEncodedARSamplingSection(ARSamplingSection):
    """AR sampling whose prompt encoder exposes a sequence-length knob."""

    max_text_length: Any = None


class SharedAttentionARSamplingSection(TextEncodedARSamplingSection):
    """AR sampling for families using the shared selectable attention adapter."""

    attention_backend: Literal["vllm_paged", "torch_native"] | None = None


class JanusProSamplingSection(SharedAttentionARSamplingSection):
    """Janus-Pro discrete image-token sampling."""

    guidance_scale: Any = None
    image_size: Any = None
    image_token_num: Any = None
    temperature: Any = None


class JanusProR1SamplingSection(JanusProSamplingSection):
    """Janus-Pro reflective sampling additions."""

    max_reflect_len: Any = None


class NextStepSamplingSection(SharedAttentionARSamplingSection):
    """NextStep continuous image-token sampling."""

    guidance_scale: Any = None
    image_size: Any = None
    image_token_num: Any = None
    num_steps: Any = None


class Emu3SamplingSection(SharedAttentionARSamplingSection):
    """Emu3 latent-grid-derived image sampling."""

    guidance_scale: Any = None
    image_area: Any = None
    ratio: Any = None
    temperature: Any = None


class GlmImageSamplingSection(TextEncodedARSamplingSection):
    """GLM-Image native-cache AR prior and frozen DiT decode controls."""

    decode_guidance_scale: Any = None
    decode_num_inference_steps: Any = None
    image_height: Any = None
    image_width: Any = None
    temperature: Any = None
    top_p: Any = None


class LlamaGenSamplingSection(ARSamplingSection):
    """LlamaGen native-cache discrete image-token sampling."""

    guidance_scale: Any = None
    temperature: Any = None
    top_k: Any = None
    top_p: Any = None


__all__ = [
    "ARSamplingSection",
    "DenoiseImageSamplingSection",
    "EchoSamplingSection",
    "Emu3SamplingSection",
    "GlmImageSamplingSection",
    "JanusProR1SamplingSection",
    "JanusProSamplingSection",
    "LlamaGenSamplingSection",
    "MagiSamplingSection",
    "NextStepSamplingSection",
    "SamplingSection",
    "SharedAttentionARSamplingSection",
    "TextEncodedARSamplingSection",
    "TextEncodedImageSamplingSection",
    "TextEncodedVideoSamplingSection",
    "VideoSamplingSection",
]
