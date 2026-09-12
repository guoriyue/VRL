# Batch pipeline lifetime

Reviewed complete `vrl/generation/execution/pipeline.py`, its CPU/fake-stream
tests, CUDA test coverage and the worker's pipelined OOM recovery branch.
No production cleanup is justified without changing lifetime guarantees.

## Retain

- `_enqueue_cpu_copies` is an asynchronous transfer boundary. It allocates pinned
  host buffers, submits nonblocking copies and records the source storage on the
  copy stream. A completion event protects readers of host buffers; record_stream
  separately protects the allocator from recycling a still-read source. Keep both.
- The local leaf callback configures the shared tensor-tree walker; it is not a
  duplicate traversal implementation. Non-CUDA tensors stay unchanged and
  slots dataclass results retain their structure. A plain `.to('cpu')` helper
  does not express these ordering/lifetime requirements.
- Produce fences are recorded before publication and queried later without a
  device-wide synchronize. They describe produce completion, not CPU-copy
  completion. Keep them distinct from pending teardown events.
- On success, join submitted copy events before returning host payloads. On
  failure, also synchronize the stream to cover a partially submitted copy whose
  event was never appended. The worker then clears failed frames and releases
  memory before returning its typed OOM retry response. Do not reorder cleanup
  or collapse these responsibilities into a generic callback.
- `forward_batches_pipelined` is a real shared execution API. It owns local
  stream/event state for one call; the executor owns model production. No
  additional pipeline class is necessary. Export names are API declarations;
  no ALL_CAPS business tables exist here.

Non-goals: promise acceleration without measurement, change CUDA stream policy,
broaden accepted device placements, rewrite the exception model, or use one
event for both production and transfer. Multi-device payload behavior and CUDA
faults during synchronization are not proven by the bounded tests below.

## Validation

16 pipeline and worker request-execution tests passed on CPU; 4 real-CUDA tests
skipped with CUDA disabled. Tests exercise result order, one produce per batch,
record-before-publish, joining a prior copy after a later produce error, slots
dataclass transfer and source-stream registration. Fake events establish call
ordering, not physical overlap or GPU allocator safety under load.
Previous isolated audit commit: `e65514233`.
