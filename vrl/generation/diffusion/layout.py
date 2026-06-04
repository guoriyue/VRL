"""Request layout helpers shared by diffusion executors and gatherers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from vrl.generation.types import GenerationRequest, GenerationSampleRow

TChunk = TypeVar("TChunk")


@dataclass(slots=True)
class VideoGenerationRequest:
    """Per-request diffusion inference parameters."""

    prompt: str = ""
    negative_prompt: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    width: int = 1024
    height: int = 640
    frame_count: int = 16
    num_steps: int = 35
    guidance_scale: float = 5.0
    seed: int | None = None
    model_name: str = ""
    fps: int = 16
    shift: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiffusionBaseParams:
    """Common parsed sampling fields every diffusion executor needs."""

    num_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    fps: int | None
    sample_batch_size: int
    max_sequence_length: int
    seed: int | None
    negative_prompt: str | None


@dataclass(frozen=True, slots=True)
class DiffusionSDEParams:
    """Parsed SDE rollout knobs for diffusion executors that collect logprobs."""

    noise_level: float
    sde_type: str
    sde_window_size: int
    sde_window_range: tuple[int, int]
    same_latent: bool
    return_kl: bool


@dataclass(frozen=True, slots=True)
class DiffusionSamplingParams:
    """Parsed diffusion sampling fields for one generation request."""

    base: DiffusionBaseParams
    sde: DiffusionSDEParams | None
    denoise_mode: str


@dataclass(frozen=True, slots=True)
class DiffusionRequestLayout:
    """Prompt-major request layout shared by diffusion executors and gatherers."""

    default_sample_batch_size: int = 1
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512
    sde_type: str = "sde"

    def parse_sampling_params(self, request: GenerationRequest) -> DiffusionSamplingParams:
        """Parse shared diffusion sampling fields from GenerationRequest."""

        sampling = request.sampling
        num_steps = int(sampling["num_steps"])
        fps_value = sampling.get("fps", self.default_fps)
        seed = sampling.get("seed")
        base = DiffusionBaseParams(
            num_steps=num_steps,
            guidance_scale=float(sampling["guidance_scale"]),
            height=int(sampling["height"]),
            width=int(sampling["width"]),
            num_frames=int(
                sampling.get(
                    "num_frames",
                    sampling.get("frame_count", self.default_num_frames),
                )
            ),
            fps=None if fps_value is None else int(fps_value),
            sample_batch_size=max(
                1,
                int(
                    sampling.get(
                        "sample_batch_size",
                        self.default_sample_batch_size,
                    )
                ),
            ),
            max_sequence_length=int(
                sampling.get(
                    "max_sequence_length",
                    self.default_max_sequence_length,
                )
            ),
            seed=None if seed is None else int(seed),
            negative_prompt=sampling.get("negative_prompt"),
        )
        denoise_mode = self._parse_denoise_mode(sampling.get("denoise_mode", "sde"))
        sde_window_range = self._parse_sde_window_range(
            sampling.get("sde_window_range", (0, num_steps)),
            num_steps=num_steps,
        )
        sde_window_size = int(sampling.get("sde_window_size", 0))
        self._validate_sde_window_size(sde_window_size, sde_window_range)
        sde = DiffusionSDEParams(
            noise_level=float(sampling.get("noise_level", 1.0)),
            sde_type=self._parse_sde_type(sampling.get("sde_type", self.sde_type)),
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            same_latent=bool(sampling.get("same_latent", False)),
            return_kl=bool(sampling.get("return_kl", False)),
        )
        return DiffusionSamplingParams(base=base, sde=sde, denoise_mode=denoise_mode)

    def repeat_encoded_batch(self, encoded: dict[str, Any], count: int) -> dict[str, Any]:
        """Repeat singleton-batch encoded tensors for a chunk sample count."""

        return {key: self.repeat_batch(value, count) for key, value in encoded.items()}

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
                f"cannot repeat tensor batch={batch} to chunk sample count {count}",
            )
        repeat_shape = (count,) + (1,) * (value.ndim - 1)
        return value.repeat(*repeat_shape)

    def ordered_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[TChunk],
    ) -> list[TChunk]:
        """Sort diffusion chunks and ensure they exactly cover sample rows."""

        if not chunks:
            raise ValueError("chunks must be non-empty")
        ordered = sorted(
            chunks,
            key=lambda chunk: (int(chunk.prompt_index), int(chunk.sample_start)),
        )
        expected = [(row.prompt_index, row.sample_index) for row in sample_rows]
        actual: list[tuple[int, int]] = []
        for chunk in ordered:
            prompt_index = int(chunk.prompt_index)
            sample_start = int(chunk.sample_start)
            sample_count = int(chunk.sample_count)
            self._validate_chunk_range(
                request,
                prompt_index=prompt_index,
                sample_start=sample_start,
                sample_count=sample_count,
            )
            self._require_rows("observations", chunk.observations, sample_count)
            self._require_rows("actions", chunk.actions, sample_count)
            self._require_rows("log_probs", chunk.log_probs, sample_count)
            self._require_rows("timesteps", chunk.timesteps, sample_count)
            self._require_rows("kl", chunk.kl, sample_count)
            self._require_rows("video", chunk.video, sample_count)
            actual.extend(
                (prompt_index, sample_index)
                for sample_index in range(sample_start, sample_start + sample_count)
            )
        if actual != expected:
            raise ValueError(
                "Diffusion chunks do not cover sample_rows in prompt-major order",
            )
        return ordered

    def max_peak_memory_mb(self, chunks: Sequence[Any]) -> float | None:
        """Return the maximum non-null peak memory metric across chunk results."""

        peaks = [chunk.peak_memory_mb for chunk in chunks if chunk.peak_memory_mb is not None]
        return max(peaks) if peaks else None

    def select_sde_window(
        self,
        sde_window_size: int,
        sde_window_range: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Pick the stochastic denoise-step window for a request."""

        if sde_window_size <= 0:
            return None
        lo, hi = sde_window_range
        start = random.randint(lo, hi - sde_window_size)
        return (start, start + sde_window_size)

    @staticmethod
    def _require_rows(name: str, value: Any, count: int) -> None:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 1:
            raise ValueError(f"chunk {name} must have a leading batch dimension")
        if int(shape[0]) != count:
            raise ValueError(f"chunk {name} has {shape[0]} rows, expected {count}")

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
        if sde_type not in {"sde", "cps"}:
            raise ValueError("sampling.sde_type must be 'sde' or 'cps'")
        return sde_type

    @staticmethod
    def _parse_denoise_mode(value: Any) -> str:
        denoise_mode = str(value).strip().lower()
        if denoise_mode not in {"native", "sde"}:
            raise ValueError(
                "sampling.denoise_mode must be 'native' or 'sde'",
            )
        return denoise_mode

    @staticmethod
    def _validate_chunk_range(
        request: GenerationRequest,
        *,
        prompt_index: int,
        sample_start: int,
        sample_count: int,
    ) -> None:
        if prompt_index < 0 or prompt_index >= len(request.prompts):
            raise ValueError(f"chunk.prompt_index={prompt_index} is out of range")
        sample_end = sample_start + sample_count
        if sample_start < 0 or sample_count < 1:
            raise ValueError(
                "chunk sample range must have non-negative start and positive count",
            )
        if sample_end > request.samples_per_prompt:
            raise ValueError(
                "chunk sample range exceeds request.samples_per_prompt: "
                f"{sample_start}:{sample_end} > {request.samples_per_prompt}",
            )


__all__ = [
    "DiffusionBaseParams",
    "DiffusionRequestLayout",
    "DiffusionSDEParams",
    "DiffusionSamplingParams",
    "VideoGenerationRequest",
]
