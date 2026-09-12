"""Request sampling parameters for full-sequence denoise executors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from vrl.generation.steps.denoise.config import DenoiseRequestOptions, DenoiseSDEParams
from vrl.generation.steps.denoise.teacache import TeaCacheConfig
from vrl.generation.types import (
    DenoiseRequest,
    GenerationRequest,
)
from vrl.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class DiffusionSamplingParams:
    """Parsed diffusion sampling fields for one generation request."""

    model_request: DenoiseRequest
    max_sequence_length: int | None
    sde: DenoiseSDEParams
    sde_window_size: int
    sde_window_range: tuple[int, int]
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


class DiffusionRequestLayout:
    """Prompt-major request parser owned by a diffusion executor.

    The fallback values have NO defaults: the executor is their single source
    and always supplies its resolved values. A default here would be a silent
    second source that drifts from the executor. Batch width is deliberately
    absent — it belongs to planning (``EnginePlan.from_request``'s single fallback),
    not to request parsing.
    """

    __slots__ = (
        "default_fps",
        "default_max_sequence_length",
        "default_num_frames",
        "sde_type",
    )

    def __init__(
        self,
        *,
        default_num_frames: int,
        default_fps: int | None,
        default_max_sequence_length: int | None,
        sde_type: str,
    ) -> None:
        self.default_num_frames = default_num_frames
        self.default_fps = default_fps
        self.default_max_sequence_length = default_max_sequence_length
        self.sde_type = sde_type

    def parse_sampling_params(self, request: GenerationRequest) -> DiffusionSamplingParams:
        """Parse shared diffusion sampling fields from GenerationRequest."""

        sampling = request.sampling
        options = request.denoise if request.denoise is not None else DenoiseRequestOptions()
        max_sequence_length = sampling.get(
            "max_sequence_length",
            self.default_max_sequence_length,
        )
        negative_prompt = sampling.get("negative_prompt")
        model_request = DenoiseRequest(
            num_steps=sampling["num_steps"],
            guidance_scale=float(sampling["guidance_scale"]),
            height=sampling["height"],
            width=sampling["width"],
            frame_count=sampling.get(
                "num_frames", sampling.get("frame_count", self.default_num_frames)
            ),
            fps=sampling.get("fps", self.default_fps),
            negative_prompt="" if negative_prompt is None else negative_prompt,
            seed=sampling.get("seed"),
        )
        # The typed options carry the rollout-owned knobs; only the two values
        # that depend on the executor or the schedule resolve here.
        sde_window_range = options.resolve_sde_window_range(model_request.num_steps)
        sde = DenoiseSDEParams(
            noise_level=options.noise_level,
            sde_type=options.sde_type or self.sde_type,
            return_kl=options.return_kl,
            return_prev_sample_mean=options.return_prev_sample_mean,
            cache_ref_noise_pred=options.cache_ref_noise_pred,
        )
        if max_sequence_length is not None:
            max_sequence_length = require_int(
                max_sequence_length,
                path="sampling.max_sequence_length",
                minimum=1,
            )

        # Preserve the existing seeded stream; unseeded requests carry their
        # fallback seed across worker serialization and batch retries.
        sde_window = None
        window_size = options.sde_window_size
        if window_size > 0:
            lo, hi = sde_window_range
            seed = model_request.seed
            if seed is None:
                seed = request.sde_window_seed
                if seed is None:
                    raise ValueError("SDE window requires request-owned sde_window_seed")
            # Fixed stream salt: vary with the request seed while keeping window
            # selection separate from other uses of that seed. Preserve this
            # value so existing seeded runs retain their window selection.
            rng = random.Random(seed ^ 0x5DE317D0)
            start = rng.randint(lo, hi - window_size)
            sde_window = (start, start + window_size)

        return DiffusionSamplingParams(
            model_request=model_request,
            max_sequence_length=max_sequence_length,
            sde=sde,
            sde_window_size=options.sde_window_size,
            sde_window_range=sde_window_range,
            denoise_mode=options.denoise_mode,
            teacache=options.teacache,
            sde_window=sde_window,
        )


__all__ = [
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
]
