# Forward-process and multisegment objectives

Reviewed complete modules: `vrl/algorithms/{diffusion_nft,v_grpo,types}.py`,
`vrl/algorithms/grpo/multisegment.py`, and `vrl/rollouts/evaluators/types.py`.
Read corresponding algorithm tests, base NFT/V-GRPO presets, trainer metric
aggregation and optimizer hooks, timestep-buffer writes, and the FLUX, SD3.5 and
Cosmos forward-input hooks. These family/trainer excerpts are dependency evidence,
not full-module coverage.

## Change

Iterate the multisegment signal dictionary's items directly instead of copying
its keys and looking each value up again. The objective does not mutate the
signal dictionary during the synchronous loop. Retain the explicit None-value
diagnostic for a mutated signal object.

Remove the local weighted-average helper's empty-list fallback. Every selected
segment appends all three metric lists together with its loss. If none is
selected, the earlier differentiable-zero branch returns before this helper is
called. This does not apply to the public PolicyUpdateStats reducer or trainer
aggregator, which have independent callers and legitimate empty inputs.

## Retained ownership and non-goals

- Multisegment config owns its default segment names, weights and training flags.
  Keep these declared defaults here rather than inventing a global vocabulary.
  Missing explicitly weighted active segments must still fail. A named advantage
  or explicit `__default__` entry is a supported input choice, not inference from
  unrelated state; the normal trainer currently passes shared tensor advantages.
- Keep TokenGRPO delegation per segment. It shares the actual clipped objective
  and KL/mask behavior, while multisegment owns weighting across unequal token
  lengths. The small local weighted average shares three metric reductions and
  does not warrant a separate aggregation object.
- NFT and V-GRPO `compute_loss` methods are uniform Algorithm protocol adapters;
  their timestep methods also serve direct tests and first-step diagnostics.
  Keep the adapters rather than flattening this cross-objective interface.
- `normalized_mse` is genuine stateless shared mathematics, used by both NFT and
  V-GRPO. Its detached double-precision normalization and FP32 callers have
  numerical meaning. Retain the shared function; moving it to a new one-function
  module or wrapping it in a class would add structure without reducing coupling
  in this import-light algorithm layer.
- Keep independent NFT and V-GRPO first-step invariants. NFT compares equal
  losses under an advantage flip; V-GRPO compares opposite losses. Their local
  loss helpers reuse RNG scopes within each diagnostic. A generic checker would
  hide which invariant each objective promises.
- Keep model capability checks and replay batch/timestep bounds. These timestep
  methods are called directly as well as through AlgorithmAdapter, and model
  annotations do not establish optional forward-process hooks. None of this
  review introduces float-token validation or additional general checker APIs.
- `_flow_time` currently isolates V-GRPO's timestep conversion. Keep its guard
  until the timestep contract is corrected below; deleting it would silently
  permit unsupported grids. Do not unify it with NFT merely because the code is
  similar: NFT casts normalized t to latent dtype, V-GRPO retains FP32.
- `_SEED_UPDATE`, `_SEED_GROUP`, and `_SEED_INDEX` are deterministic random-stream
  coefficients, a reproducibility boundary rather than business vocabulary.
  Changing or consolidating them would change training draws. They remain here.
- Metric dataclasses distinguish objective updates from initial replay parity.
  Only the latter's max/finite values control the parity gate. Keep the names,
  immutable initial snapshot and separate reductions; do not average all fields
  of TrainStepMetrics generically. `PolicyUpdateStats.weighted_mean` derives its
  homogeneous numeric schema from dataclass fields and serves both trainer and
  multisegment callers.
- Signal types are a shared evaluator/algorithm boundary. Keep nonempty segment,
  key/name agreement, mask and same-shape checks there. `_require_same_shape`
  shares the two shape comparisons without a validator wrapper class. The
  mutable dataclasses are not permanent proof of validity after construction.
  `SignalRequest` expresses optional reference/KL work uniformly across families.
- Other tuples and `__all__` in these modules are declared input/export schemas;
  there is no additional ALL_CAPS business table to relocate.

## Open findings with concrete next steps

1. **Timestep units are inferred from values.** NFT and V-GRPO divide a selected
   timestep slice by 1000 if any value exceeds 1. The denoise loop records raw
   state timesteps without a unit tag. FLUX's forward-input hook repeats the
   threshold conversion; SD3.5 consumes the raw grid. The guard rejects 80000
   but cannot distinguish normalized time 0.5 from raw time 0.5 near the end of
   a 1000-scaled schedule. Removing the heuristic requires producer-owned time
   units or an explicit normalized flow-time tensor, propagated through replay
   and family conditioning. Do not silently reinterpret existing trajectories
   or derive the scale from one selected slice.
2. **Image/video context fallback is not a general layout contract.** Both
   objectives fall back to `x0.shape[2]` for frame count when ndim >= 3, which
   includes packed FLUX and 4D image latents. FLUX currently discards num_frames,
   SD3.5 absorbs it, while Cosmos uses it to size a condition mask. Before
   replacing defaults, inspect each family's context export and trajectory
   fixtures; latent axis lengths are not interchangeable with decoded dimensions.
3. **V-GRPO resume noise state needs an explicit owner.** `_update_counter` starts
   at zero and is changed only by `after_optimizer_step(global_step + 1)`; a
   repository search finds no restore assignment. The inspected OnlineTrainer
   snapshot includes step/global_step, optimizer, scaler and EMA, but no algorithm
   state. Its restore path does not call the optimizer hook. A new objective on
   resume can therefore use the initial seed stream for its first update. Review
   checkpoint creation/restoration together with scaler-skipped updates before
   implementing a fix: simply invoking the optimizer hook at load would also
   copy/decay the previous-policy adapter, an unwanted side effect. This audit
   has not yet run an end-to-end resumed V-GRPO training reproduction.

These are unresolved behavior/ownership findings, not harmless functions to
delete. Reviewed coverage means source inspection, not closure of these items.

## Validation

93 existing multisegment/NFT/V-GRPO, metrics I/O and trainer diagnostics tests
passed on CPU. They exercise real tiny transformer/LoRA objectives, group-shared
noise, timestep bounds, loss weighting and metrics. The 22 warnings were existing
dependency/adapter and tensor-to-scalar warnings. No GPU or distributed claim.

A direct multisegment check also confirmed differentiable zero loss when all
segments have zero weights or are disabled. Ruff check, format check and diff
whitespace check passed. No new tests mirroring the simplified loop/helper.
Previous isolated audit commit: `336efeeaa`; this review and its implementation
are committed together on the audit branch.
