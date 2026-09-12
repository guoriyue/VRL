# Denoise mathematics

Following 9decfed2a, reviewed all four math/denoise modules and the generation
loop, SDE evaluator, CausVid runner/replay call sites and existing mathematical
regressions. All baseline math modules now have a reviewed disposition.

Change: move the duplicated generator/recorded-action exclusion from both CPS
and flow branches into their common path. The same check/message remains; no
random draw or formula changes. Invalid calls are rejected before branch-specific
mean calculation. DDIM retains its own check because it is a public callable and
dispatch returns before the shared flow body. Broaden SDEStepResult's docstring
to name both flow and DDIM, which already share that result.

Retain and why:

- sde_step_with_logprob is a shared generation/evaluation entry point with a lazy
  DDIM adapter. Keeping one callable shape prevents callers from duplicating
  scheduler-family dispatch. SDEStepResult expresses real named tensor outputs.
- DDIM's alpha ladder and epsilon/v-prediction conversion differ from flow sigma
  math. Zero-variance terminal/deterministic transitions use the existing squared
  error convention; they are not ordinary finite Gaussian densities. Keep that
  branch and its tests rather than hiding it in a generic Gaussian wrapper.
- Flow's runtime sigma-domain detection addresses the actual distinction between
  EDM tables and already-normalized flow tables. Config sigma_max alone is not
  authoritative. Returned actions convert back to the scheduler domain, while
  scoring and KL ingredients remain in the flow domain.
- step_index supplied by the generation loop avoids table lookup; replay can use
  scheduler lookup for supplied timesteps. These are two supported callers,
  unlike guessing checkpoint progress from filenames.
- compute_kl_divergence shares reduction math, with optional sqrt(-dt) for the
  different noise parameterizations. Do not merge DDIM and flow denominators.
- diffusion_pretraining_pair adapts actual scale_noise/add_noise scheduler APIs
  and prediction targets. Existing tests use FlowMatch, UniPC flow prediction and
  DDPM epsilon/v-prediction. Offline DPO's indexed sigma fallback remains its
  separately reviewed path; this is not a blanket scheduler utility replacement.
- RenoiseStepResult and renoise_step_with_logprob define CausVid's distribution
  shared by sampling and replay. The runner supplies per-sample noise and stores
  the returned action; model replay scores that action. Quantizing it before
  scoring and detaching it in the density preserve the behavior/replay contract.
- Scalar/sample sigma shape and action/noise shape checks guard tensor alignment
  across those callers. Positive sigma distinguishes stochastic transitions from
  terminal deterministic work handled by the family. No extra dtype gates added.
- Empty package exports preserve lightweight submodule imports. __all__ names
  are public API boundaries; no ALL_CAPS workflow vocabulary needs extraction.

Non-goals: combine different diffusion distributions, change noise consumption,
remove pseudo-log-prob conventions, add scheduler wrapper classes, alter clipping
or rewrite gradient formulas. Shared numerical functions are useful abstractions.

Limits: EDM-domain caching is attached to the scheduler and assumes its domain
does not change during that object's lifetime. Flow deterministic overrides still
follow the existing draw sequence; this cleanup does not claim RNG savings.
Step-index forwarding removes lookup syncs, not every possible scheduler/device
operation. No GPU performance or full-model parity claim is made from CPU tests.

Validation: 75 existing flow/EDM, DDIM, re-noise and scheduler-parity tests passed
on CPU, with eight dependency warnings. Coverage includes real Diffusers DDIM
mean comparisons, terminal transitions, sample/replay equality and gradients
remaining on the current prediction. Ruff check and format check passed. No tests
or runtime gates added. Coverage: 202 reviewed, 297 pending baseline modules.
