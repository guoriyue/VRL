# Training launch and artifact trace

Reviewed full trainers/trace.py, rank-primary capture/seal callers, completion
verification tests and runtime metadata collection. Previous commit: abb9f03b5.
The evaluation archive owner and full supervisor remain separate pending
reviews; trace's delegation to those interfaces is not their full coverage.

## Change

Read each installed distribution's metadata once, then obtain its Name and
Version from that parsed record. The old comprehension read metadata for the
filter, again for the key, and again through Distribution.version. Inspected
the installed standard-library property's implementation: it returns
self.metadata['Version']. Record structure and package sorting are unchanged.
No helper class, namespace wrapper or new test-only abstraction was introduced.

## Retain and why

- TrainingRunTrace owns the launch path and artifact receipt lifecycle. Capture,
  load, seal and verification are distinct operations on this record, not
  independent functions waiting to be grouped elsewhere.
- Launch observation, content consistency and process outcome are different
  evidence. A startup record is not successful training; artifact hashes alone
  do not establish process exit or learning. Preserve the separate records and
  explicit verification methods rather than a single ambiguous success flag.
- _read_launch checks published identity/digest on disk. Verification deliberately
  rereads records instead of trusting an object's stale cache. Exclusive JSON
  publication prevents accidental launch/receipt replacement.
- _git_snapshot's nested git command centralizes executable arguments, stderr
  behavior and timeout for three commands. Its unavailable status is recorded,
  not silently converted into a clean checkout. _runtime_snapshot collects the
  environment observed by the trainer process, not an inferred fleet inventory.
- _configured_data_files hashes the same open descriptor before/after fstat.
  Replacing that with a path-only sha256 helper would lose this observation
  boundary. It hashes manifests/reports, not every referenced media file.
- RUN_EVIDENCE_SCHEMA and RUN_ARTIFACTS_SCHEMA are persisted protocol names.
  _RUNTIME_ENVIRONMENT_KEYS is an explicit environment boundary that avoids
  dumping unrelated secrets. These constants are necessary, not per-algorithm
  vocabularies to move into runtime workflow.
- LocalCheckpointContent supplies the shared file/tree digest representation;
  float32_precision_state supplies the shared backend precision observation.
  Keep these actual shared mechanisms instead of another trace base class for
  unrelated state lifetimes.

Non-goals: reintroducing an attempt-ID environment variable, redefining verdicts,
merging training and precision records solely for a common name, or interpreting
successful consistency checks as benchmark/model-quality evidence.

## Limits and validation

There is intentionally no proof that a supplied verdict belongs to this exact
launch. Self-contained hashes establish consistency, not authenticity. Git
status/diff observations and multiple filesystem reads are not an atomic system
snapshot; untracked file contents are not captured. Sealing precedes runtime
cleanup, so a later cleanup failure remains the verdict's concern.

28 trace tests passed. Coverage includes exclusive launch IDs, resume records,
artifact drift, relocatable receipts, aggregate rank outcomes, environment
filtering and a fresh-process precision-backend observation. The runtime snapshot
test enumerates real installed packages on CPU. Ruff passed on the changed
module. No new test was added for this local metadata-read simplification; no
GPU fleet or evaluation-quality claim is made.
