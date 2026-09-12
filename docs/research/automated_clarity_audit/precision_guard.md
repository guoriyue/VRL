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

## Configuration finding closed in follow-up

Follow-up to 03a849541 fixes the issue from optimizer_state.md at the config
owner. PrecisionDriftGuardConfig now stores converted float thresholds instead
of validating temporary casts and retaining incompatible strings. Its
non-negative predicate rejects NaN while retaining positive infinity as an
unbounded limit. The existing fail_on_nonfinite measurement behavior remains.
max_timestep_checks uses the shared require_int boundary instead of testing a
temporary int conversion and later passing the original fractional/string value
to range. Zero still selects no timesteps. No per-timestep guards were added.

Compatibility: negative thresholds remain rejected. Numeric threshold strings
already accepted by the constructor now work during measurement. NaN thresholds
and direct non-integer check counts now fail at construction. Public Pydantic
conversion still precedes dataclass construction; this does not change that
framework's own accepted input coercions. Existing integer-count presets and
positive-infinite thresholds retain their behavior. Mutation of a config after
construction is not revalidated by this change.

The new focused regressions first produced four failures: both NaN threshold
fields were accepted, a converted string threshold failed during measurement,
and the fractional count was accepted. After the fix, 103 guard, online config,
drift config and all-experiment loading tests passed. A positive-infinity case
also confirms that unbounded finite drift remains allowed. Ruff passed for the
two changed Python files. No generic validator class or new numeric policy table
was introduced; existing named config fields own these constraints.

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
