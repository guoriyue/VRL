# Trajectory review

Reviewed baseline: `b1e219dd4e182dcc4465026641abd02651d673e0`.
All eight `vrl/trajectory` modules were read, together with the trajectory tests,
token evaluator consumers, and the gather call sites for all six public builders.
This records source review; open findings below are not resolved by that label.

## Changes

- Multisegment evaluator: select categorical segments and the enabled primary
  directly in `evaluate`, removing two single-use selection helpers and checks
  on their internally constructed return values. Preserve the error for missing
  categorical segments and the first-enabled primary when the recorded primary
  is outside the requested selection. This is an explicit subset policy, not
  checkpoint-style guessing. Keep `_segment_tensor`: trajectory values are `Any`,
  whereas this evaluator requires torch operations.
- `validation.py`: remove the repeated negative-length check on frozen
  `TrajectoryAxis`; its constructor already checks this invariant. Structural
  axis-name, rank and sample-count comparisons stay at the batch boundary.
- Populate the validator's reference cache in `_ensure_tensor_refs`, retiring
  `_collect_tensor_refs`, which only built a temporary dictionary for that owner.
- Remove runtime-object classification branches that all returned `True` before
  the unconditional `True`. Keep the scalar/container/tensor-like admission
  rules and recursive rejection. Unsupported objects remain rejected without
  inspecting their model methods merely to reject them anyway.

## Retained boundaries and non-goals

| Module | Decision and evidence |
| --- | --- |
| `__init__.py` | Keep lazy public exports and `_PUBLIC_EXPORTS`: config imports storage policy while builders import torch. This table is the package API boundary, not algorithm policy. |
| `types.py` | Keep records and their schema enums; `validate_string_tuple` is shared with `RewardView`. Keep `role_tensor` for unique-role lookup. Selection/rebuild belongs to `TrajectoryBatch`; declared sample dimensions avoid confusing equal-sized unrelated axes. |
| `views.py` | Keep the lightweight reward view, with its explicit pixel range. Moving this into torch-dependent media code would weaken the declaration boundary. |
| `reader.py` | Keep named lookup and slicing, and current validation on reader construction: tensors/segments are mutable after construction. `tests/trajectory/test_reader.py` exercises modified values, nested sequences, axis failures and original backend exceptions. No new reader wrapper class. |
| `device.py` | Keep the free tree walker and device adapter. They serve storage, batch movement, weight snapshots and generation teardown. Caller-specific leaf rules are real: device movement duck-types `to`, storage casts only floating torch tensors, CPU snapshots also detach and copy. No blanket conversion merger. |
| `storage.py` | Keep `TrajectoryStoragePolicy` and its configuration parsing; config schema and collector construction pass this object onward. `_VALID_DEVICES`/`_VALID_DTYPES` derive from schema literals. Keep byte estimation as a shared traversal rather than putting arbitrary payloads on one batch class. It deduplicates object identities, not allocator storage. |
| `validation.py` | Keep ordered `REQUIRED_TRAINABLE_ROLES` and derived singleton membership: they define the trajectory schema. Keep `tensor_ref` as a canonical public format and `validate_shape_prefix` as a shared result/builder boundary. `_fail` centralizes the domain exception within the existing validator. |
| `builders.py` | Keep six public family-shape adapters and their explicit tensor declarations. These project distinct input contracts, not a generic constructor disguised by six names. Keep private shape/context helpers where they express those projections; do not replace all constructors with a configuration table. |

No global removal of validation: mutable records and tensor-like payloads do not
currently provide an immutable, previously-validated guarantee to every reader.
No dtype, sampling temperature, mask, probability or training-selection changes.

## Open findings

1. `build_ar_multisegment_trajectory` accepts `old_log_probs` as an alternate to
   `token_log_probs` and otherwise creates zeros. The only production caller,
   `JanusProR1GenerationBatchGatherer`, always emits `token_log_probs`, sometimes
   explicitly `None`. Its gatherer verifies consistent presence across workers.
   No producer for the alternate spelling was found. Do not remove the zero
   behavior blindly: inspect R1 segment generation and non-trainable text before
   deciding which absence is legitimate. Changing the exported builder's accepted
   mapping also requires an explicit compatibility decision. This remains open.
2. Builder filtering of context and non-sample-aligned diffusion replay extras is
   existing behavior. Tests explicitly preserve omission of scalar replay extras.
   Replacing this with an explicit typed projection requires following each family
   export, rather than rejecting previously accepted inputs in this local cleanup.
3. The tensor walker reconstructs dataclasses through their constructor. Current
   inspected payloads/tests use ordinary init fields. Supporting arbitrary
   `init=False` or specialized dataclasses is not established by these tests; do
   not advertise the walker as supporting every possible Python container.

## Validation

157 tests passed: `tests/trajectory`, `tests/rollouts/replay`,
`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`, and
`tests/models/families/janus_pro/test_r1_model.py`.
Ruff checks apply only to the two modified production files. This establishes
the exercised CPU contracts, not real checkpoint or GPU training parity.

Previous implementation group: `3e9c1250c` (reference model context).
