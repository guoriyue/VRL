# Denoise configuration, buffers and TeaCache

Reviewed complete modules under `vrl/generation/steps/denoise`: `__init__.py`,
`config.py`, `loop.py` and `teacache.py`. Read cache tests, relevant preallocation
and probe tests, collector configuration projection, layout window resolution,
and executor probe setup. Larger executor/layout modules remain pending review.

## Change

Move TeaCache's torch import under TYPE_CHECKING. Every torch reference in that
module is a postponed tensor annotation; runtime calculations use methods on
the input tensors. DenoiseRequestOptions imports TeaCacheConfig, so the old
import loaded torch even when only constructing scalar configuration.

A fresh interpreter confirmed torch was loaded before the change, and another
fresh interpreter confirmed constructing DenoiseRequestOptions with enabled
TeaCache defaults no longer loads it afterward. No new lazy-import wrapper or
separate config module. Callers executing tensor operations already provide
tensors; no cache mathematics or state transition changed.

## Retained boundaries and non-goals

- Config dataclasses own defaults and request/window constraints. The request
  constructor validates the window's own shape; `resolve_sde_window_range`
  checks it against the actual schedule length. Those are different facts.
  DenoiseMode/SdeType Literals and export lists are schema/API boundaries, not
  workflow business tables.
- `from_sections` preserves explicitly declared scalar values and separately
  projects bool/mapping TeaCache settings. Keep it on the request-options owner.
  Do not replace mapping settings with a boolean or duplicate their defaults.
- The memory-probe executor validates execute_steps before encoding. Loop config
  retains the bound, and the loop runs that prefix while keeping full buffer
  allocation for sizing. Do not add another identical integer check to the loop.
  A probe result's unwritten tail is not a completed training trajectory; keep
  that distinction when reviewing the executor/worker consumers.
- DenoiseTrajectoryBuffers owns allocation, dtype/device selection and writes.
  `_expand_timestep` adapts a schedule element to the allocated batch dimension;
  scalar and per-sample schedules have different shapes. Avoid replacing this
  with a general tensor conversion without comparing its shape contract.
- `run_denoise_loop` is the shared execution boundary used by family executors.
  Its result is a data container, not a reason to turn the loop into a method on
  the output. Native scheduler advancement and stochastic window advancement
  intentionally differ; no unification of their mathematics.
- TeaCacheState is a real per-batch state owner. The loop constructs it once,
  passes a cloned pre-step latent signal, and feeds back real predictions.
  Detaching signals/predictions avoids gradient retention; cloning the signal
  in the loop prevents later latent replacement from changing the saved sample.
- `relative_l1_change` is shared by the runtime and offline drift probe. Its
  FP32 reduction prevents half-precision overflow, and its scalar result causes
  host synchronization on CUDA. Keep this shared numerical API, including the
  zero-denominator behavior; do not hide it inside a private state method.
- Warmup, final-step, missing-cache and nonfinite-change behavior have explicit
  tests. Skip counters are observations, not a promise of training acceleration.
  Cache thresholds and direct mapping validation are genuine configuration
  boundaries. No new checker class or speculative tensor-type tests.
- Keep the empty package facade and existing profile/engine counter names as
  import and telemetry boundaries. No ALL_CAPS business vocabulary to relocate.

## Follow-up: the rollout field called KL

The loop's `record_step` writes `abs(sde_result.log_prob)` into `kl` when
return_kl is enabled. The collector batch builder sums that tensor and subtracts
`kl_reward_coef * sum` from rewards. This recorded quantity is not in general a
KL divergence against a reference policy. Existing buffer tests explicitly
expect the absolute log-probability. Preserve behavior in this import cleanup,
but review the reward-shaping contract before renaming or changing the formula;
the reference-prediction buffer is separately optional. This is an open semantic
finding, not evidence that a naming-only change fixes the objective.

## Validation

132 existing denoise-step, config drift/compile and collector-runtime tests
passed; 3 CUDA cases skipped with CUDA disabled. Includes TeaCache transitions,
nonfinite signals, FP16 reduction, buffer writes, probe limits and drift gates.
Cold-interpreter configuration construction verifies the import change. Ruff
check/format and whitespace checks passed. No GPU memory/performance claim and
no new tests for a type-only import.

Previous isolated audit commit: `b94e6a624`.
