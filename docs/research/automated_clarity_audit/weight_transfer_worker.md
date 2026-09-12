# Weight transfer and worker version ownership

Completed source review of `generation/weight_transfer.py`,
`generation/ray/weight_sync.py`, `generation/ray/worker.py` and
`generation/ray/pipeline_protocol.py`. Finished the execution worker's transfer
and version-slot caller follow-up from `placement_rank_probe.md`, following the
runtime publication barrier, session teardown and existing protocol tests.

## Change: discard installed-version state with a released model

Worker release dropped the executor but retained `_policy_version` and
`_uses_versioned_slots`. A later cold load constructed the launch checkpoint,
then treated it as the old installed version or tried to activate slots that
belonged to the discarded model. Plain mode could accept an old installed-version
request against fresh bootstrap weights; slot mode rejected even the bootstrap
version as an evicted slot.

After successful model release, reset both fields to their initial launch
semantics. The next weight installation advances them normally. Keep sleep/wake
unchanged: parking retains the model and its installed state. Failed CuMem
release still raises before this reset and retains the existing quarantine.

Two regressions exercise release, real worker cold-load flow with an injected
tiny executor, bootstrap execution, rejection of the discarded installed
version, then reinstallation and execution. Both failed before the fix. This
does not load a production model checkpoint; it verifies worker state ownership.
Normal Ray session close releases actors before killing them, so this issue
primarily concerns the core's supported release/rebuild path, not ordinary
runtime parking. Do not claim that every training run encountered it.

## Retain and why

- `weight_manifest` is a source-state schema boundary. `iter_weight_chunks`
  bounds independent tensor storage and clones slices so transport cannot hold
  a full backing allocation through a small view. `iter_weight_buckets` packs
  multiple small chunks under the same byte ceiling. These are distinct shared
  serialization operations; receiver buffers belong to StagedWeightTransfer.
  Moving all three into a stateless class would not establish a new owner.
- StagedWeightTransfer's shape/dtype/offset/ID checks validate wire content
  before copying into receiver-owned CPU storage. `finish` rejects gaps without
  modifying the model. Empty tensors need no chunks but still appear in the
  finished mapping. Keep explicit CPU allocation under alternative default
  devices. Buffer size limits wire tensor bytes, not full receiver RAM.
- Worker begin/receive/commit/abort are protocol endpoints. An incomplete finish
  leaves staging available for continuation or explicit abort; after a complete
  finish, an installation attempt clears staging in finally. Sender failure
  invokes bounded abort on all engines, even if dispatcher admission has failed.
  Do not move finish into finally merely to make every failure discard state.
- The sender's local `broadcast` captures one transfer's version and serializes
  each payload once for all ranks. It centralizes three real RPC phases. ACK
  validation checks untyped remote output; equality alone would accept True for
  version 1. `uniform_rank_result` checks rank agreement, while the sender checks
  agreement with its requested version. They are different boundaries.
- Transfer completion is not a fleet-atomic parameter transaction. One rank can
  install before a peer fails; the runtime terminalizes failed updates before
  publishing success. Abort clears staging and does not roll back installed
  parameters. Preserve that explicit limitation.
- RayGenerationWorker's thin methods are the framework facade over a Ray-free
  core. HEALTH_CONCURRENCY_GROUP is a protocol name shared by method annotations
  and actor construction, not a removable workflow constant. It keeps progress
  and liveness queries separate from serialized GPU work.
- The progress callback registers produce fences under a lock; the health
  thread publishes only the contiguous completed prefix. Failed event queries
  leave that prefix unchanged. PipelinedRequestProgress is the wire schema;
  PipelinedProgressError carries terminal protocol failure. Keep this small
  protocol module rather than importing actor implementation into consumers.

Non-goals: introduce sender/receiver checker classes, weaken remote validation,
change bucket contents, merge callback lifetimes with global worker state,
promise distributed rollback, alter model slot retention, or reinterpret progress
as CPU-copy completion. Probe policy ownership remains the precisely scoped
follow-up recorded in `placement_rank_probe.md`.

## Validation

- Before the release fix, transfer/slots/weight-sync coverage: 67 passed,
  including real CPU Ray actors, all-rank readback, incomplete bucket abort and
  cancellation/publication cases. One upstream Ray environment warning.
- Worker identity/debug and pipeline-progress coverage: 25 passed. These cover
  the reviewed adapters; they do not establish GPU allocator behavior.
- After the release fix: 125 passed, 2 CUDA-dependent cases skipped across
  transfer, version slots, worker parking, pipelined execution and Ray sync.
  The same upstream Ray warning remains. Ruff check and format check passed
  for both changed Python files.

Previous isolated audit commit: `424c22fe0`.
