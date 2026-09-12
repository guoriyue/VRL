# Algorithm loss and input ownership

Reviewed complete modules: `vrl/algorithms/{base,dpo,trajectory}.py` and
`vrl/algorithms/grpo/{__init__,continuous,token}.py`. Read their config dispatch,
the prebuilt algorithm section path, online replay/advantage dispatch, offline
DPO/SFT loss consumption, and relevant existing algorithm tests. Excerpts from
the large trainers and config schema do not count as full-module review.

## Changes

- Move both DPO loss descriptions ahead of their local imports. They were inert
  string expressions rather than Python docstrings. `help()`/introspection now
  sees the API documentation; torch remains a call-time dependency.
- Use `dataclasses.replace` when `AlgorithmAdapter` supplies missing advantages.
  Only `advantages` changes, while the original input stays untouched. The old
  constructor manually forwarded all other fields, making every new input field
  require an unrelated adapter edit. There are no AlgorithmInput subclasses in
  repository callers; external subclasses would now retain their class and run
  that class's dataclass construction hooks, whereas the old path made a base
  AlgorithmInput. No tensor copy or training formula change.

## Retained boundaries and non-goals

- `Algorithm` and `ComponentAdvantageAlgorithm` are actual trainer interfaces.
  The latter is checked at runtime when component rewards are available; keep
  its separate optional capability rather than growing the mandatory protocol.
- `AlgorithmAdapter` owns dispatch-time input requirements and derivation of
  missing advantages. The fallback from explicit group IDs to signal group IDs
  reads another declared representation of the same grouping; it is not an
  inference from filenames or unrelated progress counters.
- `diffusion_dpo_loss` is stateless paired-preference math used directly by the
  offline trainer, outside the online reward/advantage protocol. Keep it free.
  `diffusion_sft_loss` is its exported winner-loss API and owns FP32 reduction.
  Do not introduce an objective class simply to house these two functions.
- Keep DPO shape equality and even-batch checks: prediction/target arithmetic
  can otherwise broadcast or split unequal winner/loser halves. The ordinary
  tensor annotation does not declare those runtime dimensions.
- Keep `_require_trust_region_signals`: two distinct objectives share the SDE
  input boundary, including mandatory recorded rollout mean and dt. Keep the
  named rectification guard as the corresponding Flash-GRPO signal boundary;
  it keeps missing-input diagnostics separate from the weight formula. Neither
  becomes a generic validator class or loses its conditional requirements.
- Keep `_cross_rank_mean`: distributed sum/count reduction is different from
  local `.mean()`, including an empty local rank. The local `_per_sample` shares
  three coefficient reductions inside Flash-GRPO, and `_token_kl_per_token`
  isolates the three supported mathematical estimators. These numerical helpers
  do not need their own classes or a dispatch table of lambdas.
- GRPO's `_loss_weight` is the subclass hook actually overridden by Flash-GRPO.
  Shared advantage/precision initialization and broadcasting have several
  objective consumers. Similar-looking objective bodies preserve different
  trust-region masks, denominators and gradient scales; merging those formulas
  is not a clarity-only refactor.
- Algorithm ClassVar contracts, required signal/data key tuples, and `__all__`
  are protocol/schema/export declarations. There is no workflow-local ALL_CAPS
  business vocabulary to relocate in these modules. Keep the empty GRPO package
  facade rather than eagerly importing every objective during config dispatch.

## SFT validation ownership: precise deferral

The previous rules review identified a single-field check in a cross-section
module. GRPOConfig and DiffusionDPOConfig are mutable standard dataclasses;
FlashGRPOConfig inherits GRPOConfig, while FlowDPPOConfig and GRPOGuardConfig
do not inherit its SFT field. AlgorithmConfig accepts an existing config instance
by identity, bypassing TypeAdapter, and a prebuilt AlgorithmConfig can also be
passed to RootConfig.

Verified directly: construct DiffusionDPOConfig, place it in AlgorithmConfig,
then set its sft_weight to -0.1. RootConfig currently rejects both the prebuilt
section and a mapping containing that config. Moving the numeric check only to
dataclass __post_init__ would lose both checks. Moving it only to a section's
before-validator would miss the existing-section path. Direct dataclass
construction currently has no numeric SFT validation at all.

Retain the numeric guard until the mutable/prebuilt revalidation contract is
settled across configuration owners. A coherent move must cover those paths
without adding a second numeric guard or silently freezing public configs. The
dataset requirement stays cross-section regardless. This is a documented
ownership debt, not a claim that the check is redundant or that all configs have
already validated their scalar values.

## Verification

- 248 existing tests passed: DPO, GRPO, Flash-GRPO, Flow-DPPO/GRPO-Guard,
  Dance-GRPO, algorithm declarations, config schema and offline DPO timesteps.
  Two dependency deprecation warnings were emitted.
- After changing the input adapter, 34 existing input-contract, token-GRPO and
  online reward/update tests passed.
- A direct adapter exercise with real GRPO derived advantages using signal group
  IDs, preserved all other fields and the original input, and completed backward.
- Direct introspection verified both docstrings and torch-free DPO import.
  Ruff check and format check passed on the two modified Python files.

All runs used CPU with the isolated worktree on PYTHONPATH. No new tests for
docstring position or dataclass mechanics; no GPU/distributed-training claim.
Previous audit commit: `f9252df3b`. This commit records the implementation and
coverage together in the isolated branch's history.
