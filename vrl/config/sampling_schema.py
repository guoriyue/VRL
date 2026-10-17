"""Lightweight public schemas for the YAML ``sampling`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery reject family-inapplicable knobs
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt

from vrl.config.base import ConfigBase


class SamplingSection(ConfigBase):
    """Base for family-selected public sampling configuration.

    The declared fields are also the per-prompt ``request_overrides`` vocabulary:
    the collector revalidates every override against the selected family class
    before it reaches a ``GenerationRequest``.
    """

    # Usually a per-prompt request override (paired rollouts/evals); a YAML value
    # seeds every request identically. The online collector draws unspecified
    # seeds from the checkpointed driver RNG; direct generation owns its policy.
    seed: StrictInt | None = None


class TeaCacheSection(ConfigBase):
    """``sampling.teacache`` mapping form; ``teacache: true`` selects the defaults.

    reader: vrl/generation/steps/denoise/teacache.py TeaCacheConfig.from_sampling
    (the request boundary), via the collector's request projection.
    """

    enabled: StrictBool = True
    threshold: float | None = None
    warmup_steps: StrictInt | None = None


class DenoiseImageSamplingSection(SamplingSection):
    """Sampling inputs shared by denoise image generators."""

    guidance_scale: Any = None
    height: Any = None
    num_steps: Any = None
    width: Any = None
    negative_prompt: str | None = None
    # Rollout-only forward approximation (skips denoise steps on a cached
    # noise_pred). A request-scoped drift source: config validation refuses it
    # unless an importance-sampling correction is armed.
    teacache: StrictBool | TeaCacheSection | None = None


class TextEncodedImageSamplingSection(DenoiseImageSamplingSection):
    """Image sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class QwenImage21SamplingSection(DenoiseImageSamplingSection):
    """Qwen-Image-2.1 reference preprocessing and decoded channel selection.

    No ``max_sequence_length``: the Qwen3-VL prompt template is tokenized whole,
    so the knob is not a pipeline input (the model ignores it if passed).
    """

    reference_resolution: StrictInt = Field(default=1024, ge=32)
    output_mode: Literal["rgb", "rgba"] = "rgb"


class QwenImageLayeredSamplingSection(TextEncodedImageSamplingSection):
    """Qwen-Image-Layered layer budget and source-image channel handling.

    ``layers`` is the number of RGBA layers generated per source image (the
    checkpoint was trained around 4; 8 is the practical ceiling). ``input_mode``
    chooses whether a source PNG's alpha reaches the four-channel VAE or is
    replaced by opaque, matching the upstream decomposition script.
    """

    layers: StrictInt = Field(default=4, ge=1, le=8)
    input_mode: Literal["rgb", "rgba"] = "rgb"


class VideoSamplingSection(DenoiseImageSamplingSection):
    """Sampling inputs shared by denoise video generators."""

    fps: Any = None
    num_frames: Any = None


class TextEncodedVideoSamplingSection(VideoSamplingSection):
    """Video sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class MiniMaxH3SamplingSection(TextEncodedVideoSamplingSection):
    """MiniMax-H3 sampling: guidance-distilled, ``max_sequence_length`` bounds the
    text rows of its packed sequence (``fps`` is fixed at 24 by the checkpoint)."""

    guidance_scale: Literal[1.0] | None = None


__all__ = [
    "DenoiseImageSamplingSection",
    "MiniMaxH3SamplingSection",
    "QwenImage21SamplingSection",
    "SamplingSection",
    "TeaCacheSection",
    "TextEncodedImageSamplingSection",
    "TextEncodedVideoSamplingSection",
    "VideoSamplingSection",
]
