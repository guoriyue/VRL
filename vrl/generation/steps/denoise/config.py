"""Configuration owned by the continuous denoise-step loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from vrl.generation.steps.denoise.teacache import TeaCacheConfig
from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.config.sampling_schema import SamplingSection
    from vrl.config.schema import RolloutConfig


DenoiseMode = Literal["native", "sde"]
SdeType = Literal["flow_grpo", "cps", "ddim"]


@dataclass(frozen=True, slots=True)
class DenoiseRequestOptions:
    """Rollout-owned denoise knobs carried by one ``GenerationRequest``.

    Projected once by the collector from ``rollout.*`` / ``rollout.sde.*`` (the
    YAML declares no defaults for them, so the effective defaults live here and
    nowhere else). ``sde_type=None`` defers to the executor family's default and
    ``sde_window_range=None`` means the whole schedule; both resolve against the
    request's step count in ``DenoiseRequestLayout.parse_sampling_params``.
    """

    denoise_mode: DenoiseMode = "sde"
    noise_level: float = 1.0
    sde_type: SdeType | None = None
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] | None = None
    return_prev_sample_mean: bool = False
    group_shared_noise: bool = False
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
        require_int(self.sde_window_size, path="rollout.sde.window_size", minimum=0)
        if self.sde_window_range is not None:
            if (
                not isinstance(self.sde_window_range, (tuple, list))
                or len(self.sde_window_range) != 2
            ):
                raise ValueError("rollout.sde.window_range must contain two integer values")
            lo, hi = self.sde_window_range
            require_int(lo, path="rollout.sde.window_range[0]")
            require_int(hi, path="rollout.sde.window_range[1]")
            if lo < 0 or hi <= lo:
                raise ValueError("rollout.sde.window_range must satisfy 0 <= lo < hi")
            if self.sde_window_size > hi - lo:
                raise ValueError(
                    "rollout.sde.window_size cannot exceed rollout.sde.window_range",
                )
            object.__setattr__(self, "sde_window_range", (lo, hi))

    @classmethod
    def from_sections(
        cls,
        rollout: RolloutConfig | None,
        sampling: SamplingSection | None,
    ) -> DenoiseRequestOptions:
        """Project rollout.* / rollout.sde.* / sampling.teacache into the typed options.

        Only YAML-declared values are passed, so the option defaults stay the single
        source.
        """

        values: dict[str, Any] = {}
        if rollout is not None:
            for name in (
                "denoise_mode",
                "noise_level",
                "return_prev_sample_mean",
                "group_shared_noise",
            ):
                value = getattr(rollout, name)
                if value is not None:
                    values[name] = value
            sde = rollout.sde
            if sde is not None:
                values["sde_type"] = sde.type
                if sde.window_size is not None:
                    values["sde_window_size"] = sde.window_size
                if sde.window_range is not None:
                    values["sde_window_range"] = tuple(sde.window_range)
        teacache = getattr(sampling, "teacache", None)
        if teacache is not None:
            # Bool flows as-is; the mapping form is the section minus unset keys, so
            # TeaCacheConfig.from_sampling sees exactly what the YAML declared.
            values["teacache"] = TeaCacheConfig.from_sampling(
                teacache
                if isinstance(teacache, bool)
                else teacache.model_dump(mode="python", exclude_none=True, exclude_unset=True),
            )
        return cls(**values)

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
    return_prev_sample_mean: bool = False


@dataclass(frozen=True, slots=True)
class DenoiseLoopConfig:
    """Runtime inputs for one sample batch's denoise loop."""

    sample_start: int
    sample_count: int
    seed: int | None
    sde: DenoiseSDEParams
    sde_window: tuple[int, int] | None
    denoise_mode: DenoiseMode = "sde"
    teacache: TeaCacheConfig | None = None
    # Set when the request shares one initial latent per prompt group: the
    # generator seed every batch of this prompt draws from (no batch offset).
    initial_noise_seed: int | None = None
    # Memory probes may execute fewer steps while retaining full buffer allocation.
    execute_steps: int | None = None

    def __post_init__(self) -> None:
        require_int(self.sample_start, path="sample_start", minimum=0)
        require_int(self.sample_count, path="sample_count", minimum=1)
        if self.denoise_mode not in get_args(DenoiseMode):
            raise ValueError(
                f"denoise_mode must be one of {get_args(DenoiseMode)}; got {self.denoise_mode!r}"
            )


__all__ = [
    "DenoiseLoopConfig",
    "DenoiseMode",
    "DenoiseRequestOptions",
    "DenoiseSDEParams",
    "SdeType",
]
