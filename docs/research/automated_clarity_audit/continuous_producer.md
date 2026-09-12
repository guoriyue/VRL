# Continuous producer admission and failure ownership

Completed the producer module review after the capacity, consumer and owner
reviews. Read installation, admission, generation/scoring, harvesting, finite
batch draining, stop and observability paths, with their owner/consumer callers
and existing contract tests. Previous audit commit: c6f1d6119.

## Change

Read the active batch once in current_batch_id, then inspect its failure and
identity directly. Describe appended work as a prefetched prompt batch in the
two ownership docstrings. No scheduling, retry or training behavior changes.

## Retain and why

- _ActivePromptBatch owns immutable input selections alongside mutable pending
  slots, retries and deferred failure. The prompt tuple is a shallow snapshot,
  not a deep freeze of arbitrary prompt objects. _new_prompt_batch constructs
  and validates before replacing installed state; this helper has a real
  transactional purpose shared by installation and prefetch.
- _admit, _submit, _harvest_done and _enqueue_result describe distinct transitions
  in one producer owner. Combining them into one long coroutine would obscure
  pending, executing and ready lifetimes. No new controller class is needed.
- Split scoring retries the same generated receipt under the reward lock. A
  post-generation failure cannot re-enter generation through the ordinary slot
  retry path: _GroupProductionError preserves that distinction. A terminal
  runtime error invalidates the whole fleet rather than only one prompt batch.
- A failed prefetched batch stops its own pending work and siblings while the
  current batch can finish. Its error surfaces when that batch becomes the
  demanded head. Clearing completed traceback frames releases retained payloads
  while preserving traceback locations and explicit causes. Existing tests
  check both delayed batch failure and immediate fleet failure.
- Generated capacity is reserved before submission and retained through reward
  retry. The generation concurrency slot ends earlier. The failed-batch harvest
  path also releases reservations for tasks cancelled before their first turn,
  when the coroutine's finally block never ran.
- Finite-batch draining admits pending slots even while normal admission is
  paused. Waiting for only currently running tasks would permit a weight update
  before the rest of the installed batch generated at its fixed version.
- _CPU is the queue storage device; cadence and retry constants are explicit
  timing budgets, not algorithm vocabulary. Metric keys and __all__ are schema
  and public export boundaries. Keep these declarations in their owner.

Non-goals: changing retry policy, flattening lifecycle methods, replacing fixed
batch identity with completion order, renaming existing metrics, or claiming
that bounded cancellation guarantees GPU reclamation. The stop and owner-thread
limitations recorded in continuous_capacity.md and continuous_owner.md remain.

## Validation

All 74 existing producer/consumer contract tests passed on CPU; Ruff passed for
the changed production module. No new test was added for this local-state read
and documentation cleanup. This does not establish real GPU teardown behavior.
