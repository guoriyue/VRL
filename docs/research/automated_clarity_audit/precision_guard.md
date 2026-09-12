# Precision guard collective and record ownership

Reviewed the complete online/precision_guard.py module, trainer enforcement
caller, public config/builder/preset thresholds, the performance probe caller,
and guard/config/update-path tests. The online trainer remains pending full
review; this change covers only its initial precision-drift method.
Previous audit commit: ff21ae58a.

## Change

Remove four redundant scalar reductions from OnlineTrainer after
measure_precision_drift. Measurement already all-gathers whole records and
selects the same record, including the aggregate violation flag, on every rank.
Reducing identical maxima and booleans again adds collectives without adding
information. Trainer now enforces and writes the selected record directly.

For normal measured records, thresholds and warn/fail behavior are unchanged.
For an empty timestep selection, worst_stats remains None rather than being
rewritten by the trainer into synthetic zero/finite fields. This matches the
measurement API and makes absence of measurements explicit.

## Retain and why

- Measurement and enforcement are separate because distributed selection must
  happen before any rank raises. Only the primary writer logs warnings/evidence;
  fail mode raises on every rank. The performance probe uses the combined public
  facade run_precision_drift_guard. A wrapper class adds no owned state here.
- _select_distributed_worst_record owns the collective boundary. Selecting one
  complete record preserves timestep/rank/metadata provenance; field-wise maxima
  could describe no actual measured sample. Do not replace this with independent
  reductions or remove it with the redundant trainer reductions.
- _drift_record_key is shared by local timestep and distributed rank selection.
  Its nested relative calculation combines two differently scaled limits,
  including zero thresholds and nonfinite measurements. Keep this actual shared
  ordering mechanism instead of creating an unrelated comparison utility class.
- select_guard_timesteps owns bounded, deterministic first/last/middle selection
  and checks its public index input. normalize_role_precision_label is shared
  with runtime precision consumers, including legacy no-to-fp32 normalization;
  quantization suffixes remain significant. Neither is an incidental forwarding
  function to inline at every caller.
- Public exports and record event/field names are interface/schema boundaries.
  There is no ALL_CAPS business vocabulary table to relocate in this module.
- Trainer's scalar collective helpers remain used by exact replay parity and
  work-admission agreement. Only the redundant drift call sites are deleted.

Non-goals: changing drift math, reordering collective participation across rank
branches, merging exact replay parity with precision-corrected drift, creating a
new validator hierarchy, or eliminating uniform public helper shapes for LOC.

## Open configuration finding

The threshold issue from optimizer_state.md remains actionable: temporary casts
in PrecisionDriftGuardConfig do not normalize stored values, and NaN passes the
existing negative-only tests. Positive infinity has historically been accepted;
the ordering helper handles it as an unbounded positive limit. A focused config
fix should preserve that existing infinity behavior while rejecting NaN and
storing the converted threshold, rather than introduce a new finite-only policy
or add per-timestep validation. The max_timestep_checks constructor similarly
checks int(value) but retains the original value; review that count contract at
the same config boundary. Presets inspected use ordinary integer counts and
finite thresholds. These config fixes are not claimed by this collective edit.

## Validation

The new trainer regression failed before the edit on a redundant scalar
reduction after a supplied already-selected record. It now verifies the trainer
returns and persists that exact record without another scalar collective.
Existing mocked two-rank selection tests verify intact remote provenance and
the update-path tests verify enforcement before optimizer updates.

57 precision guard, streaming/full update and config drift tests passed on CPU.
Ruff passed for both changed Python files. This suite includes mocked distributed
selection, not a live multi-rank NCCL performance measurement; removing four
reductions is established from call sites and the targeted regression, not a
training throughput claim.
