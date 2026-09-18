"""Cross-section rules over a parsed ``RootConfig`` (validation tier 2).

The config package validates in three tiers, one module each:

1. **Section shape** — ``schema.py``. Pydantic models: closed keys, types,
   per-section invariants. Runs inside ``parse_config``.
2. **Cross-section rules** — this module's ``check_cross_section_rules``: the
   checks that relate two or more parsed sections (``algorithm.kind`` against
   ``rollout``, ``algorithm.sft_weight`` against ``data``, …). No I/O, no torch,
   no runtime modules: every entrypoint that parses a config pays for them, so
   they must stay import-light. ``RootConfig``'s ``model_validator`` calls it,
   so the checks fire on every construction path (``parse_config`` and direct
   ``RootConfig.model_validate``).
3. **Launch gates** — ``validation.py``. Checks that need the resolved
   precision policy, runtime modules or the filesystem; only training
   launches pay for them (``require_training_config``).

Adding a check: append a block below that raises ``ValueError`` naming the
offending dotted path. Every check needs the selected algorithm, which is
resolved once at the top. A check that needs a runtime module belongs in
tier 3; one that reads a single section belongs in tier 1.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vrl.config.algorithm import resolve_kl_reward_coef

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


def check_cross_section_rules(root: RootConfig) -> None:
    """Run every cross-section check against ``root``; the first violation raises."""

    if (
        root.model is not None
        and root.model.lora is not None
        and root.model.lora.parameter_dtype == "float32"
    ):
        training = root.distributed.training if root.distributed is not None else None
        if (
            training is not None
            and training.strategy == "fsdp"
            and (training.fsdp is None or training.fsdp.precision_policy != "none")
        ):
            raise ValueError(
                "model.lora.parameter_dtype=float32 requires "
                "distributed.training.fsdp.precision_policy=none to preserve frozen dtypes",
            )

    algo = root.algorithm
    if algo is None:
        return
    contract = algo.hyperparameters.config_contract
    kind = algo.kind
    rollout = root.rollout

    # ── algorithm.kl_reward_coef shapes rewards with the collected per-step KL.
    # An objective whose trajectory records no per-step KL would silently
    # ignore a positive coefficient.
    if resolve_kl_reward_coef(algo.kl_reward_coef) > 0.0 and not contract.supports_step_kl_reward:
        raise ValueError(
            "algorithm.kl_reward_coef > 0 requires a diffusion rollout trajectory "
            f"with collected per-step KL; algorithm.kind={kind!r} does not provide one",
        )

    # ── Sections and explicit fields outside the algorithm's declared surface.
    # An empty allowed set permits only an empty section; a None entry forbids
    # the section outright; consumed_sections=None imposes no restriction.
    for section_name, allowed in contract.consumed_sections or ():
        section = getattr(root, section_name)
        if section is None:
            continue
        if allowed is None:
            raise ValueError(f"{kind} does not consume the {section_name} config section")
        unsupported = sorted(section.model_fields_set - allowed)
        if unsupported:
            fields = ", ".join(f"{section_name}.{name}" for name in unsupported)
            raise ValueError(f"{kind} does not consume config field(s): {fields}")

    # ── SFT weights must be valid and backed by the data source their owner declares.
    raw = getattr(algo.hyperparameters, "sft_weight", None)
    if raw is not None:
        try:
            sft_weight = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("algorithm.sft_weight must be a finite number >= 0") from exc
        if not math.isfinite(sft_weight) or sft_weight < 0:
            raise ValueError("algorithm.sft_weight must be a finite number >= 0")
        if sft_weight > 0:
            if contract.sft_source == "unsupported":
                raise ValueError(
                    f"algorithm.sft_weight > 0 is not supported by algorithm.kind={kind!r}",
                )
            if contract.sft_source == "latents" and (
                root.data is None or not root.data.sft_latents
            ):
                raise ValueError(
                    "algorithm.sft_weight > 0 requires data.sft_latents "
                    "(the precomputed clean-latents shard; see "
                    "vrl/scripts/denoise/encode_targets.py)",
                )

    # ── Objectives that collect through the stochastic sampler need rollout.sde.
    # Membership of sde.type itself is the SdeConfig Literal's job; only the
    # presence of the block is a cross-section fact.
    if contract.needs_sde_rollout and (rollout is None or rollout.sde is None):
        raise ValueError("config missing required field: rollout.sde.type")


__all__ = ["check_cross_section_rules"]
