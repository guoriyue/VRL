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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vrl.config.schema import generation_request_rollout_fields
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.trajectory.storage import TrajectoryStoragePolicy

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


@dataclass(frozen=True, slots=True)
class RolloutCollectorConfig:
    """Collector-local policy plus the fail-closed generation request projection."""

    request_sampling: dict[str, Any] = field(default_factory=dict)
    samples_per_generation_batch: int | Literal["auto"] | None = None
    denoise: DenoiseRequestOptions | None = None
    # rollout.group_shared_noise: every sample of a prompt group starts the
    # denoise from one latent (DanceGRPO on video). A rollout allocation rule,
    # so it becomes per-sample initial-noise seeds on the request; the
    # execution layer never sees a group.
    group_shared_noise: bool = False
    trajectory_storage: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )
    # Where edit-chain collection writes the parent image each later step is
    # conditioned on: <trainer.output_dir>/edit_chains. Chains cannot be
    # collected without it; one-shot prompts never touch it.
    edit_chain_media_dir: Path | None = None

    @classmethod
    def from_root(cls, root: RootConfig) -> RolloutCollectorConfig:
        """Project the parsed public sections into collector-local and request state."""

        rollout = root.rollout
        sampling = root.sampling

        request_sampling: dict[str, Any] = {}
        # The planner batch width and the denoise options are GenerationRequest
        # fields, and group_shared_noise is this collector's allocation rule
        # (per-sample seeds); only the remaining rollout scalars flatten.
        rollout_fields = (
            generation_request_rollout_fields()
            - _DENOISE_OPTION_FIELDS
            - {"samples_per_generation_batch", "group_shared_noise"}
        )
        for name, section, allowed in (
            ("rollout", rollout, rollout_fields),
            ("sampling", sampling, type(sampling).model_fields if sampling is not None else ()),
        ):
            if section is None:
                continue
            declared = section.model_dump(mode="python", exclude_none=True, exclude_unset=True)
            for key, value in declared.items():
                # Nested blocks have their own projection; only scalars flatten.
                if key not in allowed or isinstance(value, dict) or is_dataclass(value):
                    continue
                if key in request_sampling:
                    raise ValueError(
                        f"rollout request key {key!r} has multiple config owners; "
                        f"remove the duplicate from {name}",
                    )
                request_sampling[key] = value
        samples_per_generation_batch = (
            rollout.samples_per_generation_batch if rollout is not None else None
        )
        trajectory_storage = (
            rollout.trajectory_storage if rollout is not None else None
        ) or TrajectoryStoragePolicy()
        output_dir = root.trainer.output_dir if root.trainer is not None else None
        return cls(
            request_sampling=request_sampling,
            samples_per_generation_batch=samples_per_generation_batch,
            denoise=DenoiseRequestOptions.from_sections(rollout, sampling),
            group_shared_noise=bool(getattr(rollout, "group_shared_noise", None)),
            trajectory_storage=trajectory_storage,
            edit_chain_media_dir=Path(output_dir) / "edit_chains" if output_dir else None,
        )


# Derived from the typed options, so a knob added to DenoiseRequestOptions is
# automatically kept off the flat sampling dict.
_DENOISE_OPTION_FIELDS = frozenset(item.name for item in fields(DenoiseRequestOptions))


__all__ = [
    "RolloutCollectorConfig",
]
