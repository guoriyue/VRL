"""Lightweight public schemas for the YAML ``sampling`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery reject family-inapplicable knobs
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import StrictBool, StrictInt

from vrl.config.base import ConfigBase


class SamplingSection(ConfigBase):
    """Base for family-selected public sampling configuration.

    The declared fields are also the per-prompt ``request_overrides`` vocabulary:
    the collector validates every override against the selected family class
    before it reaches a ``GenerationRequest`` (``SamplingSection.require_overrides``).
    """

    # Usually a per-prompt request override (paired rollouts/evals); a YAML value
    # seeds every request identically. The online collector draws unspecified
    # seeds from the checkpointed driver RNG; direct generation owns its policy.
    seed: StrictInt | None = None

    @classmethod
    def require_overrides(cls, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Return per-prompt ``request_overrides`` validated against this family's fields.

        The override vocabulary is this class's fields, so a typo fails with the
        same ``unknown sampling.<key>`` message as a YAML typo instead of riding
        the wire to a runtime that ignores it. Returns the validated values.
        """

        section = cls.revalidate(dict(overrides), section="sampling")
        return section.model_dump(mode="python", exclude_unset=True)


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
    # unless a drift guard or importance-sampling correction is armed.
    teacache: StrictBool | TeaCacheSection | None = None


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


class MiniMaxH3SamplingSection(TextEncodedVideoSamplingSection):
    """MiniMax-H3 sampling: guidance-distilled, ``max_sequence_length`` bounds the
    text rows of its packed sequence (``fps`` is fixed at 24 by the checkpoint)."""

    guidance_scale: Literal[1.0] | None = None


class MagiSamplingSection(SamplingSection):
    """Inputs mapped into MAGI-1's isolated official inference runtime."""

    fps: Any = None
    height: Any = None
    num_frames: Any = None
    num_steps: Any = None
    width: Any = None


__all__ = [
    "DenoiseImageSamplingSection",
    "EchoSamplingSection",
    "MagiSamplingSection",
    "MiniMaxH3SamplingSection",
    "SamplingSection",
    "TeaCacheSection",
    "TextEncodedImageSamplingSection",
    "TextEncodedVideoSamplingSection",
    "VideoSamplingSection",
]
