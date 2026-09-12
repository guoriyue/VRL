# Sample batch contracts

Reviewed complete `vrl/generation/execution/sample_batches.py`. Read its actual
base-executor consumer, representative gatherers, Cosmos3's ragged token export,
and existing sample-batch/gatherer tests. No production change is warranted.

## Retain shared operations

- GenerationSampleBatch owns identity, planning and split arithmetic. Splits
  preserve prompt index and ordered sample ranges. Batch keys are a protocol
  identifier used in retry/result joins. Keep these methods on the batch owner.
- `ordered_covering_batches` is a shared gather boundary: valid individual ranges
  alone do not establish exact coverage or order. Its protocol/type parameter
  preserves the concrete payload type for family-specific checks afterward.
- `require_sample_rows` supports actual tensor and Python-sequence payloads.
  `concatenate_sample_values` preserves a consistent type and tensor dtype.
  Joining results can otherwise promote tensor types and alter integer values;
  the tests demonstrate this at the wire reassembly boundary. This is not a
  hypothetical token-value checker inside the model.
- SampleAlignedValues explicitly marks ragged sample rows. Cosmos3 uses it for
  prompt-specific token IDs. Plain lists/tuples remain static schedule/shape
  data; do not infer sample alignment from their length. The wrapper removes
  ambiguity rather than adding an owner with no state.
- `gather_replay_tensors` distinguishes all-None, inconsistent presence,
  sample-aligned tensors, explicit ragged rows and shared static data. Each case
  has a different merge rule. `gather_batch_context` checks shared equality.
  Their `_values_match` helper handles nested tensors without ambiguous tensor
  truth conversion. These shared numerical/structural operations do not belong
  to one family result class or a generic checker namespace.
- `run_sample_batches_with_oom_retry` is a callback adapter around local batch
  execution. It clears failed-forward frame locals before emptying CUDA cache,
  then prepends ordered halves. Non-OOM and single-sample terminal failures keep
  their original traceback. Existing weak-reference and traceback tests verify
  those distinct lifetimes. Do not catch all errors or retry an unsplittable row.
- Export names and batch-key syntax are API/protocol vocabulary. There are no
  ALL_CAPS business tables to move. Tensor imports remain call-time dependencies.

Non-goals: consolidate every helper into a class, replace explicit ragged-row
declarations with shape guesses, change dtype policy or weaken error diagnostics.

## Evidence

The immediately preceding 133-test execution-contract run included
`test_sample_batches.py` and `test_batch_gatherer.py`. Git diff confirmed those
tests and this module are unchanged since `f6e57fe26`; no redundant rerun was
needed for a retained-module review. Those tests cover ordered OOM splitting,
frame-local release, terminal diagnostic preservation, ragged/static merge,
per-batch row counts and dtype preservation. No physical GPU OOM claim.

This review adds source-level coverage beyond the previous caller excerpts.
