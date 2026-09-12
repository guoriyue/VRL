# Advantage normalization and package boundary

Following 9def7a85b, reviewed advantages.py and algorithms/__init__.py, their
algorithm/trainer callers, GRPO config construction and factory reward-weight
injection. The package root deliberately exports nothing so importing lightweight
algorithm configuration does not eagerly import Torch implementations.

Change: describe GroupAdvantageEstimator's existing ownership (component weights,
normalization settings and selected strategy), removing the speculative claim
that future state would justify it. No numerical implementation changes.

Retain and why:

- all_reduce_sufficient_stats shares one collective mechanism across advantage
  normalization, trainer reward reporting and Flash-GRPO rollout statistics.
  It returns sums, squared sums and count; each caller owns its reduction formula.
  Empty local values must participate in a distributed call rather than return
  early. This is a shared communication boundary, not a function to inline.
- _population_std_across_ranks owns the global denominator. Local execution uses
  var_mean; distributed execution uses reduced sufficient statistics. These are
  population statistics but are not numerically identical implementations.
- _standardize_group_rewards centers each prompt group; group_relative_advantages
  adds the final clamp. The unclamped helper is also needed for component-first
  combination so clipping happens after weights and summation, not per component.
- Low-precision rewards are promoted for mean/variance and epsilon arithmetic,
  then advantages return to the original dtype. The constant decimal reward test
  exercises a real rounding issue, not an invented invalid input.
- GroupAdvantageEstimator holds per-algorithm settings and reward weights. Config
  selects the strategy, while scripts/common/factory.py supplies reward.weights.
  Keep the owner and existing constructor; no additional estimator base class.
- _STRATEGIES is a two-entry dispatch/validation table isolated inside that owner.
  DEFAULT_STRATEGY is shared with the public GRPO config default. Retain both:
  this is a deliberate configuration taxonomy, not duplicated per-algorithm facts
  embedded in trainer control flow. Do not extract another file just for two keys.
- Strategy validation has both config and direct estimator construction callers.
  Component key agreement compares runtime rewards against configured weights.
  Sorted component order keeps cross-rank collectives aligned. Keep these checks
  rather than assuming every public construction goes through one factory.
- __all__ declarations express package/module API boundaries. An empty root is
  useful import isolation; it does not need algorithm re-exports for symmetry.

Non-goals: alter normalization/clipping, change accumulation dtype or collective
backend, vectorize group loops, remove supported strategies, or add checker
classes/tests. Broad numerical redesign would exceed this clarity cleanup.

Limits: per-group normalization assumes each local rank owns complete groups;
only the denominator is global when requested. Sorted component keys cannot
repair different key sets across ranks, and the local/global variance formulas
need not be bitwise equal. The sufficient-statistics helper itself does not
promote dtype; that is explicit at its normalization caller. No stronger
precision or fault-recovery contract is claimed.

Validation: 27 existing advantage-combine, global-std Gloo and GRPO tests passed
on CPU. Ruff check and format check
passed. No tests or runtime guards added. Coverage: 193 reviewed, 306 pending.
