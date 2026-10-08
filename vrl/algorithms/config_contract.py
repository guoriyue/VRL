"""Import-light algorithm requirements for validation and model construction."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AlgorithmConfigContract:
    """Declared behavior, not user-overridable hyperparameters.

    Every independent algorithm config must choose its rollout, reward-shaping,
    and SFT semantics. A restricted surface lists the fields each named section
    consumes. An empty set allows only an empty section; a None entry forbids
    the entire section. Sections not listed have no restriction from this
    contract, and consumed_sections=None imposes no surface restrictions.
    """

    needs_sde_rollout: bool
    sft_source: Literal["unsupported", "latents", "preference_winner"]
    # Objective requirements on the model. ``requires_previous_policy``: the
    # objective scores the trainable policy against theta_old (the detached
    # current prediction) through the full-sequence replay forward;
    # ``requires_reference_policy``: the pre-training weights. Never a family
    # default.
    requires_previous_policy: bool = False
    requires_reference_policy: bool = False
    # Whether the objective stays sound on samples an older policy version
    # generated (continuous scheduling's bounded lag). An objective whose
    # behaviour policy is the current weights cannot absorb that lag.
    tolerates_off_policy_staleness: bool = False
    # The loss is *defined* by a clipped/guarded ratio against the rollout
    # policy (Flow-DPPO / GRPO-Guard), so it needs a behaviour policy that
    # differs from the trained one and the rollout's stored proposal mean.
    # Plain GRPO's clip is only a safety rail and leaves this False.
    requires_active_trust_region: bool = False
    consumed_sections: tuple[tuple[str, frozenset[str] | None], ...] | None = None
