# Shared execution contracts

Reviewed complete `executor_base.py`, `planner.py` and `types.py` in
`vrl/generation/execution`. Inspected injected-gatherer tests, planner/OOM tests,
memory-probe and pipelined-progress tests, plus worker memory normalization and
parking snapshot validation call sites. Larger worker, parking, pipeline and
sample-batch modules remain pending complete review.

## Retain; no production changes

- BatchExecutorBase shares request-level execution across binding families.
  Its lambda binds a request to the batch retry callback, while gather_batches
  delegates to an injected protocol object. These are actual common execution
  and composition boundaries. Keep the missing-gatherer diagnostic rather than
  silently looking up a family registry from this neutral layer.
- EnginePlan is the single batch-width resolution owner: explicit override,
  request width, then whole prompt group. Those are deliberate precedence rules,
  not guessing. Unresolved `auto` fails rather than pretending to be a numeric
  batch size. GenerationSampleBatch.plan owns width/count validation; there is
  no need for another integer check in EnginePlan.
- GenerationBatchEnvelope and GenerationBatchResult carry request/batch identity
  across worker dispatch. StaleSlotDiscard and PipelinedRequestOutOfMemory have
  distinct recovery meanings; do not merge them into error strings or a generic
  successful result with missing output.
- BatchProduceFence remains local and queries completion without synchronizing.
  QueryableCompletion is the device-event interface, and BatchCompletionCallback
  preserves its public alias form. Do not serialize CUDA events with wire data.
- WorkerMemoryParkingSnapshot validates process-attributed residual memory before
  handoff. Its baseline accounts for unavoidable process footprint; physical
  residency is not equivalent to live torch tensor bytes. The shared residual
  validator and backend Literal vocabulary are real boundaries.
- BatchMemoryReading.from_metrics is the worker normalization boundary. The
  worker clears the original binding mapping afterward to avoid serializing the
  same measurements twice. Missing readings represent unavailable measurements,
  not inferred zeros. Its peak, non-torch and budget properties derive distinct
  quantities used by sizing; keep them on the reading owner.
- Probe trial/result dataclasses separate OOM outcomes from measured success.
  Successful measurements and positive chosen widths have actual consumers.
  Keep the wire constraints without introducing a universal checker class.
  No additional speculative float/type guards or tests in this review.
- Literal placement/backend vocabularies, field names and __all__ are wire/schema
  and public API declarations. No ALL_CAPS business vocabulary needs relocation.

Non-goals: eliminate protocol methods for line-count reduction, alter admission
math, change seed streams, weaken parking, or merge local events with wire types.
This disposition records why the small modules are useful instead of manufacturing
a refactor for every reviewed file.

## Validation

133 existing gatherer, sample-batch, memory-shadow and pipelined-progress tests
passed on CPU. Memory/progress tests use controlled readings and fake events/Ray
handles; they are not evidence of physical GPU parking or multi-node throughput.
Previous isolated audit commit: `5390f63f3`.
