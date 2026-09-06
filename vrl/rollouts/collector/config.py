"""Rollout config projection from the parsed public config.

The one place validated public config becomes generation-request wire state:
``RolloutCollectorConfig`` splits the ``rollout``/``sampling`` sections into
the per-request sampling payload, the typed engine-level request fields
(batch width, trainable segments, trajectory storage, denoise options), and the
collector-local KL reward coefficient. Projection is fail-closed — accepted
keys are derived from the schema types (``generation_request_rollout_fields``,
the family-selected ``SamplingSection``), never from a hand-maintained list, so
a new schema field flows through without a second vocabulary to update.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

from vrl.config.algorithm import resolve_kl_reward_coef
from vrl.config.schema import generation_request_rollout_fields
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.steps.denoise.teacache import TeaCacheConfig
from vrl.trajectory import TrajectoryStoragePolicy

if TYPE_CHECKING:
    from vrl.config.base import ConfigBase
    from vrl.config.sampling_schema import SamplingSection
    from vrl.config.schema import RolloutConfig, RootConfig


@dataclass(frozen=True, slots=True)
class RolloutCollectorConfig:
    """Collector-local policy plus the fail-closed generation request projection."""

    request_sampling: dict[str, Any] = field(default_factory=dict)
    samples_per_generation_batch: int | Literal["auto"] | None = None
    train_segments: dict[str, bool] | None = None
    denoise: DenoiseRequestOptions | None = None
    kl_reward_coef: float = 0.0
    trajectory_storage: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )

    @classmethod
    def from_root(cls, root: RootConfig) -> RolloutCollectorConfig:
        """Project the parsed public sections into collector-local and request state."""

        rollout = root.rollout
        sampling = root.sampling
        algorithm = root.algorithm

        request_sampling: dict[str, Any] = {}
        # The planner batch width and the denoise options are GenerationRequest
        # fields, not sampling keys; only the remaining rollout scalars flatten.
        _merge_flat_section_values(
            request_sampling,
            rollout,
            "rollout",
            allowed=generation_request_rollout_fields()
            - _DENOISE_OPTION_FIELDS
            - {"samples_per_generation_batch"},
        )
        samples_per_generation_batch = (
            rollout.samples_per_generation_batch if rollout is not None else None
        )
        _merge_flat_section_values(
            request_sampling,
            sampling,
            "sampling",
            allowed=frozenset(type(sampling).model_fields)
            if sampling is not None
            else frozenset(),
        )
        hyperparameters = algorithm.hyperparameters if algorithm is not None else None
        train_segments = getattr(hyperparameters, "train_segments", None)
        kl_reward_coef = resolve_kl_reward_coef(
            algorithm.kl_reward_coef if algorithm is not None else None,
        )
        trajectory_storage = (
            rollout.trajectory_storage if rollout is not None else None
        ) or TrajectoryStoragePolicy()
        return cls(
            request_sampling=request_sampling,
            samples_per_generation_batch=samples_per_generation_batch,
            train_segments=None if train_segments is None else dict(train_segments),
            denoise=_denoise_options(rollout, sampling, kl_reward_coef=kl_reward_coef),
            kl_reward_coef=kl_reward_coef,
            trajectory_storage=trajectory_storage,
        )


# Derived from the typed options, so a knob added to DenoiseRequestOptions is
# automatically kept off the flat sampling dict.
_DENOISE_OPTION_FIELDS = frozenset(item.name for item in fields(DenoiseRequestOptions))


def _denoise_options(
    rollout: RolloutConfig | None,
    sampling: SamplingSection | None,
    *,
    kl_reward_coef: float,
) -> DenoiseRequestOptions:
    """Project rollout.* / rollout.sde.* / sampling.teacache into the typed options.

    Only YAML-declared values are passed, so the option defaults stay the single
    source. ``return_kl`` is derived: KL rollout signals are recorded exactly
    when an SDE block exists and the KL reward coefficient is on.
    """

    values: dict[str, Any] = {}
    if rollout is not None:
        for name in (
            "denoise_mode",
            "noise_level",
            "return_prev_sample_mean",
            "cache_ref_noise_pred",
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
            values["return_kl"] = kl_reward_coef > 0.0
    teacache = getattr(sampling, "teacache", None)
    if teacache is not None:
        # Bool flows as-is; the mapping form is the section minus unset keys, so
        # TeaCacheConfig.from_sampling sees exactly what the YAML declared.
        values["teacache"] = TeaCacheConfig.from_sampling(
            teacache if isinstance(teacache, bool) else _section_values(teacache),
        )
    return DenoiseRequestOptions(**values)


def _section_values(section: ConfigBase | None) -> dict[str, Any]:
    """The keys a parsed section was actually given, as plain python values."""

    if section is None:
        return {}
    return section.model_dump(mode="python", exclude_none=True, exclude_unset=True)


def _merge_flat_section_values(
    values: dict[str, Any],
    section: ConfigBase | None,
    name: str,
    *,
    allowed: frozenset[str],
) -> None:
    for key, value in _section_values(section).items():
        if key not in allowed:
            continue
        # Nested blocks (sde, trajectory_storage, torch_profiler) have their own
        # projection or no wire presence at all; only scalars flatten here.
        if isinstance(value, dict) or is_dataclass(value):
            continue
        if key in values:
            raise ValueError(
                f"rollout request key {key!r} has multiple config owners; "
                f"remove the duplicate from {name}",
            )
        values[key] = value


__all__ = [
    "RolloutCollectorConfig",
]
