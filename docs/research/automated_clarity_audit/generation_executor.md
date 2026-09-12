# Generation executor result classification

Reviewed the complete Ray generation executor, worker stale-slot result producer,
continuous producer exception routing, and OOM/progress/engine test coverage.

## Change

Move stale-slot classification from the initial execution path into the existing
OOM processing loop, before each round's results are accepted or retried. A
policy slot can disappear between the original OOM and child execution. Previously
that child result became RuntimeError instead of StaleSlotDiscard. The latter
has a dedicated continuous-producer path that fails the fixed-version prompt
batch; ordinary exceptions instead enter collection failure/retry handling.
This is classification consistency, not a new promise to resume an evicted slot.

The regression submits a four-sample batch that OOMs, then receives stale-slot
results for both two-sample children. Before the fix it failed with RuntimeError;
afterward it raises StaleSlotDiscard without gathering partial output. Initial
stale-slot behavior and all ordinary OOM splitting remain unchanged. On retry,
the diagnostic batch denominator describes the current retry round.

## Retain and why

- Rank result combiners are engine RPC adapters. They choose failure precedence
  and check rank agreement before the driver receives one engine result. Keep
  them distinct from request identity validation against submitted envelopes.
- The identity helper is shared by initial results and retry children. It owns
  a real request/result relation; a result's own constructor cannot establish it.
- OOM splitting belongs to this generation-aware executor, while actor dispatch
  owns engine slots. Binding children to their parent engine prevents concurrent
  retries on the same already-overfull device. Do not move generation concepts
  into the generic actor pool or introduce a new retry owner solely for line count.
- The pipelined waiter tracks completed work rather than treating heartbeats as
  progress. Its local callback adapts typed stale exceptions to dispatcher slot
  release. Keep the callback/framework boundary and cancellation cleanup.
- _PIPELINED_PROGRESS_POLL_INTERVAL_S is a transport polling cadence, not a
  business taxonomy. Keep it and the public __all__ facade. The per-request lock
  protects the worker's single active progress record; queued requests do not
  acquire an execution deadline before admission.

Non-goals: changing batch sizing, sample ordering, policy retention, timeout
semantics, rank failure precedence, or wrapping these methods in more classes.
No GPU throughput or end-to-end training correctness claim.

## Validation

New stale-after-OOM regression failed before the production change. Then 68
OOM split, pipelined progress and engine tests passed, including existing CPU
Ray multi-rank failure tests; one GPU allocator test was deselected. Ruff check
and format check passed on the two changed Python files.
Previous isolated audit commit: a917bd777.
