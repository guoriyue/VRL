# Log-probability mismatch and correction

Following 915db8a35, reviewed all of algorithms/logprob_mismatch.py, the continuous
and token loss consumers, trainer drift guard and existing correction tests.

Changes:

- Initialize immutable scalar/string config defaults directly instead of seven
  field(default=...) calls with no metadata or other options. Dataclass fields,
  defaults and validation remain the same; remove the unused field import.
- Correct the config docstring's claim that guard and RS read identical stats
  and cannot disagree. The guard computes reduced LogprobMismatchStats and checks
  maximum differences; RS reads raw log-ratios in the surrogate and can accept a
  sequence whose mean offsets opposing deviations. Their decisions are distinct.

Retain and why:

- LogprobMismatchStats owns detached FP32 measurement and aggregation. Its maxima
  and finiteness are not weighted means; preserve these different reductions.
  The existing weighted aggregation test verifies that difference.
- PrecisionCorrectionConfig is reachable without importing Torch. TYPE_CHECKING
  tensor annotations and call-time imports maintain the config parsing boundary.
- Config mode/range validation handles public configuration choices. Reserved
  recompute='on' explicitly rejects an unimplemented behavior rather than silently
  enabling a no-op. No supported modes or checks removed in this pass.
- TIS and RS functions are shared numerical operations used by continuous,
  token and trust-region algorithms. TIS modifies importance weights or supplies
  an element mask; RS supplies a sample/sequence mask. Merging them into the
  dataclass would couple runtime tensor operations to configuration ownership.
- combine_keep_masks shares validity/TIS/RS intersection and broadcasting. It
  removes real duplication across losses. The caller uses the combined mask in
  the denominator; zeroing loss without that denominator change is not equivalent.
- For a per-sample scalar log-ratio, the two sequence modes coincide. Token
  sequences use last-axis mean or extrema, excluding padding. Keep that actual
  shape distinction rather than flattening every trajectory into one scalar.
- The nested aggregate mean helper shares weights across diagnostic fields;
  it does not need a separate class. __all__ is the only module ALL_CAPS data and
  expresses the public API. Mode literals are a small config vocabulary local to
  their owner, not a workflow taxonomy to extract into another file.

Non-goals: change importance/PPO formulas, introduce old-policy recomputation,
alter padding/nonfinite handling or add hypothetical bad-value tests. Config is
mutable and the standalone functions are public; this review does not promise
runtime enforcement after arbitrary config mutation. Aggregate max/finite
semantics include supplied observations regardless of their individual weight;
the trainer excludes dummy observations before aggregation.

Validation: 112 existing mismatch, GRPO, token, Flow-DPPO/GRPO-Guard and precision
drift-guard tests passed on CPU. Coverage includes gradient effects of correction,
weighted diagnostics, padding and offsetting sequence ratios. Ruff check and
format check passed. No new checks or tests. Coverage: 194 reviewed, 305 pending.
