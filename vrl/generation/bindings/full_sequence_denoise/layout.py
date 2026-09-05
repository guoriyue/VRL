"""Request layout helpers shared by full-sequence denoise executors and gatherers."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from vrl.generation.execution.sample_batches import ordered_covering_batches
from vrl.generation.steps.denoise.config import DenoiseSDEParams
from vrl.generation.steps.denoise.teacache import TeaCacheConfig
from vrl.generation.types import (
    DenoiseRequest,
    GenerationRequest,
    GenerationSampleRow,
)

TChunk = TypeVar("TChunk")


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
    # The RESOLVED stochastic window, drawn once at parse time (see
    # select_sde_window). Every sample batch of the request reads this field, so
    # chunked groups share one window — Flash-GRPO's iso-temporal grouping.
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
        num_steps = int(sampling["num_steps"])
        fps_value = sampling.get("fps", self.default_fps)
        max_sequence_length = sampling.get(
            "max_sequence_length",
            self.default_max_sequence_length,
        )
        seed = sampling.get("seed")
        model_request_kwargs: dict[str, Any] = {
            "num_steps": num_steps,
            "guidance_scale": float(sampling["guidance_scale"]),
            "height": int(sampling["height"]),
            "width": int(sampling["width"]),
            "frame_count": int(
                sampling.get(
                    "num_frames",
                    sampling.get("frame_count", self.default_num_frames),
                )
            ),
        }
        if fps_value is not None:
            model_request_kwargs["fps"] = int(fps_value)
        if sampling.get("negative_prompt") is not None:
            model_request_kwargs["negative_prompt"] = sampling["negative_prompt"]
        if seed is not None:
            model_request_kwargs["seed"] = int(seed)
        denoise_mode = self._parse_denoise_mode(sampling.get("denoise_mode", "sde"))
        sde_window_range = self._parse_sde_window_range(
            sampling.get("sde_window_range", (0, num_steps)),
            num_steps=num_steps,
        )
        sde_window_size = int(sampling.get("sde_window_size", 0))
        self._validate_sde_window_size(sde_window_size, sde_window_range)
        sde = DenoiseSDEParams(
            noise_level=float(sampling.get("noise_level", 1.0)),
            sde_type=self._parse_sde_type(sampling.get("sde_type", self.sde_type)),
            return_kl=bool(sampling.get("return_kl", False)),
            return_prev_sample_mean=bool(
                sampling.get("return_prev_sample_mean", False),
            ),
            cache_ref_noise_pred=bool(
                sampling.get("cache_ref_noise_pred", False),
            ),
        )
        params = DiffusionSamplingParams(
            model_request=DenoiseRequest(**model_request_kwargs),
            max_sequence_length=(
                None if max_sequence_length is None else int(max_sequence_length)
            ),
            sde=sde,
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            denoise_mode=denoise_mode,
            teacache=TeaCacheConfig.from_sampling(sampling.get("teacache")),
        )
        # Resolve the stochastic window HERE, once per request, so every sample
        # batch built from these params shares it (see select_sde_window).
        return dataclasses.replace(params, sde_window=self.select_sde_window(params))

    def repeat_batch(self, value: Any, count: int) -> Any:
        """Repeat a singleton tensor batch or accept an already-sized batch."""

        if count < 1:
            raise ValueError("count must be >= 1")
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim == 0:
            return value
        batch = int(value.shape[0])
        if batch == count:
            return value
        if batch != 1:
            raise ValueError(
                f"cannot repeat tensor batch={batch} to batch sample count {count}",
            )
        repeat_shape = (count,) + (1,) * (value.ndim - 1)
        return value.repeat(*repeat_shape)

    @staticmethod
    def ordered_batches(
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[TChunk],
    ) -> list[TChunk]:
        """Sort diffusion batches and check they exactly cover the sample rows.

        A ``@staticmethod`` so the gatherer can sort without building a throwaway
        layout, while the ordering stays grouped with the parser it validates
        for. Reads no parsing fallback; every row-bearing batch tensor must carry
        exactly ``sample_count`` leading rows.
        """

        return ordered_covering_batches(
            request,
            sample_rows,
            batches,
            row_fields=("observations", "actions", "log_probs", "timesteps", "kl", "video"),
        )

    def select_sde_window(
        self,
        params: DiffusionSamplingParams,
    ) -> tuple[int, int] | None:
        """Pick the stochastic denoise-step window for a request.

        Drawn once per REQUEST (parse_sampling_params stores the result on the
        params), not per sample batch: all chunks of a request — and therefore
        all G samples of a prompt group — share one window, which is what makes
        a group's stochastic step land on the same timestep (Flash-GRPO's
        iso-temporal grouping; group advantages are then never confounded by
        timestep difficulty).

        When the request carries a seed the draw is derived from it, so
        multi-rank engines and re-parses agree deterministically. Integer
        arithmetic, not tuple hashing — hash() is process-randomized and would
        silently break cross-rank agreement. Without a seed it falls back to
        the module RNG (rank-coherent only via the worker RNG sync).
        """

        sde_window_size = params.sde_window_size
        if sde_window_size <= 0:
            return None
        lo, hi = params.sde_window_range
        seed = params.model_request.seed
        if seed is not None:
            rng = random.Random(int(seed) ^ 0x5DE317D0)
            start = rng.randint(lo, hi - sde_window_size)
        else:
            start = random.randint(lo, hi - sde_window_size)
        return (start, start + sde_window_size)

    @staticmethod
    def _parse_sde_window_range(value: Any, *, num_steps: int) -> tuple[int, int]:
        try:
            lo = int(value[0])
            hi = int(value[1])
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(
                "sampling.sde_window_range must contain two integer values",
            ) from exc
        if lo < 0 or hi <= lo or hi > num_steps:
            raise ValueError(
                "sampling.sde_window_range must satisfy 0 <= lo < hi <= num_steps",
            )
        return lo, hi

    @staticmethod
    def _validate_sde_window_size(
        sde_window_size: int,
        sde_window_range: tuple[int, int],
    ) -> None:
        if sde_window_size < 0:
            raise ValueError("sampling.sde_window_size must be >= 0")
        lo, hi = sde_window_range
        if sde_window_size > hi - lo:
            raise ValueError(
                "sampling.sde_window_size cannot exceed sampling.sde_window_range",
            )

    @staticmethod
    def _parse_sde_type(value: Any) -> str:
        sde_type = str(value)
        if sde_type not in {"flow_grpo", "cps", "ddim"}:
            raise ValueError(
                "sampling.sde_type must be 'flow_grpo', 'cps', or 'ddim'",
            )
        return sde_type

    @staticmethod
    def _parse_denoise_mode(value: Any) -> str:
        denoise_mode = str(value).strip().lower()
        if denoise_mode not in {"native", "sde"}:
            raise ValueError(
                "sampling.denoise_mode must be 'native' or 'sde'",
            )
        return denoise_mode


__all__ = [
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
]
