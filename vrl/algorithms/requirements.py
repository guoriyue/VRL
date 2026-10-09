"""Import-light algorithm requirements for validation and model construction."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AlgorithmRequirements:
    """What an objective needs from the run: declared behavior, never YAML knobs.

    Every objective config carries one as a ClassVar. Config resolution reads it
    to reject a run the objective cannot be sound on, and the trainer reads it
    to decide which policies to hold.
    """

    # The objective trains on reverse-SDE transitions, so generation must run
    # the stochastic sampler (``rollout.sde``).
    needs_sde_rollout: bool
    # Where a clean-target SFT term takes its targets: precomputed latents
    # (``data.sft_latents``), the preference winner (offline DPO), or no such
    # term at all.
    sft_source: Literal["unsupported", "latents", "preference_winner"]
    # The objective scores the trainable policy against theta_old, the detached
    # current prediction, through the full-sequence replay forward. Such an
    # objective cannot train on samples an older policy generated, so it also
    # rules out continuous scheduling's bounded staleness.
    requires_previous_policy: bool = False
    # The loss reads the pre-training weights unconditionally (not only behind a
    # KL weight).
    requires_reference_policy: bool = False
    # The loss is *defined* by a clipped/guarded ratio against the rollout
    # policy (Flow-DPPO / GRPO-Guard), so it needs a behaviour policy that
    # differs from the trained one and the rollout's stored proposal mean.
    # Plain GRPO's clip is only a safety rail and leaves this False.
    requires_active_trust_region: bool = False
    # Config sections an offline objective consumes: a listed section allows
    # only the named fields (an empty set permits only an empty section, None
    # forbids the section); unlisted sections and ``None`` are unrestricted.
    consumed_sections: tuple[tuple[str, frozenset[str] | None], ...] | None = None
