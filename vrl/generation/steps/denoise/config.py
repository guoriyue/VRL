"""Configuration owned by the continuous denoise-step loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from vrl.generation.steps.denoise.teacache import TeaCacheConfig

DenoiseMode = Literal["native", "sde"]
SdeType = Literal["flow_grpo", "cps", "ddim"]


@dataclass(frozen=True, slots=True)
class DenoiseRequestOptions:
    """Rollout-owned denoise knobs carried by one ``GenerationRequest``.

    Projected once by the collector from ``rollout.*`` / ``rollout.sde.*`` (the
    YAML declares no defaults for them, so the effective defaults live here and
    nowhere else). ``sde_type=None`` defers to the executor family's default and
    ``sde_window_range=None`` means the whole schedule; both resolve against the
    request's step count in ``DiffusionRequestLayout.parse_sampling_params``.
    """

    denoise_mode: DenoiseMode = "sde"
    noise_level: float = 1.0
    sde_type: SdeType | None = None
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] | None = None
    return_kl: bool = False
    return_prev_sample_mean: bool = False
    cache_ref_noise_pred: bool = False
    teacache: TeaCacheConfig | None = None

    def __post_init__(self) -> None:
        if self.denoise_mode not in get_args(DenoiseMode):
            raise ValueError(
                f"rollout.denoise_mode must be one of {get_args(DenoiseMode)}; "
                f"got {self.denoise_mode!r}",
            )
        if self.sde_type is not None and self.sde_type not in get_args(SdeType):
            raise ValueError(
                f"rollout.sde.type must be one of {get_args(SdeType)}; got {self.sde_type!r}",
            )
        if self.sde_window_size < 0:
            raise ValueError("rollout.sde.window_size must be >= 0")
        if self.sde_window_range is not None:
            try:
                lo, hi = (int(self.sde_window_range[0]), int(self.sde_window_range[1]))
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError(
                    "rollout.sde.window_range must contain two integer values",
                ) from exc
            if lo < 0 or hi <= lo:
                raise ValueError("rollout.sde.window_range must satisfy 0 <= lo < hi")
            if self.sde_window_size > hi - lo:
                raise ValueError(
                    "rollout.sde.window_size cannot exceed rollout.sde.window_range",
                )
            object.__setattr__(self, "sde_window_range", (lo, hi))

    def resolve_sde_window_range(self, num_steps: int) -> tuple[int, int]:
        """The window bounds for a request with ``num_steps`` denoise steps."""

        if self.sde_window_range is None:
            if self.sde_window_size > num_steps:
                raise ValueError(
                    "rollout.sde.window_size cannot exceed sampling.num_steps",
                )
            return (0, num_steps)
        lo, hi = self.sde_window_range
        if hi > num_steps:
            raise ValueError(
                "rollout.sde.window_range must satisfy hi <= sampling.num_steps",
            )
        return (lo, hi)


@dataclass(frozen=True, slots=True)
class DenoiseSDEParams:
    """Parsed SDE knobs used to sample and score denoise transitions."""

    noise_level: float
    sde_type: str
    return_kl: bool
    return_prev_sample_mean: bool = False
    cache_ref_noise_pred: bool = False


@dataclass(frozen=True, slots=True)
class DenoiseLoopConfig:
    """Runtime inputs for one sample batch's denoise loop."""

    sample_start: int
    sample_count: int
    seed: int | None
    sde: DenoiseSDEParams
    sde_window: tuple[int, int] | None
    denoise_mode: str = "sde"
    teacache: TeaCacheConfig | None = None
    execute_steps: int | None = None


__all__ = [
    "DenoiseLoopConfig",
    "DenoiseMode",
    "DenoiseRequestOptions",
    "DenoiseSDEParams",
    "SdeType",
]
