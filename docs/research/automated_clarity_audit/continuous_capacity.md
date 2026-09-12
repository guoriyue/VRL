# Continuous capacity, receipts and consumer selection

Reviewed complete generated_capacity.py, queue.py, types.py, staleness.py and
consumer.py, producer admission/scoring/release/stop call sites, and mechanism
and producer/consumer contract tests. Subsequent full-module reviews are recorded
in continuous_producer.md and continuous_owner.md. Previous audit commit: 1788dd443.

## Change

Correct GeneratedRolloutCapacity.close documentation: it closes admission and
clears accounting, but does not prove payloads were released or producer tasks
stopped. Producer.stop has a bounded wait and may abandon cancellation-resistant
tasks before closing this ledger. Late release calls deliberately become no-ops.
Also use prefetched rather than preview in the queue occupancy description and
correct a test comment's misspelled waiting state. Runtime behavior is unchanged.

## Retain and why

- Generated capacity spans reservation, generated receipt, reward waiting and
  scoring/retries. A generation concurrency slot ends earlier. The ready queue
  instead stores completed scored items. Combining these counters would erase
  different lifetimes. Neither mechanism independently schedules or accelerates
  work; the producer controls backpressure and owns payloads.
- GeneratedRolloutCapacity has one production owner, but its methods centralize
  real state transitions and byte ceilings, not mere function names. Preserve
  this cohesive accounting object for now; no new wrapper is needed. The cached
  byte total keeps admission constant-time. Waiting and scoring sets carry
  stage information not derivable from the reservation byte map alone.
- Queue insertion rejects over-capacity before mutation rather than evicting a
  receipt required by an installed prompt batch. Identity removal avoids tensor
  equality. Frozen receipt fields stabilize bytes and work identity, while the
  referenced batch/stats remain mutable. An estimate is not allocator residency
  or a guarantee against future payload growth.
- ContinuousRolloutSettings carries configuration through multiple owners with
  no second set of defaults. ProducerState separates completion/error counters,
  cadence diagnostics and terminal control-loop error. Terminal error is not an
  interchangeable retry count. Keep these explicit carriers rather than one
  untyped dictionary whose fields need rediscovery at each layer.
- StalenessPolicy is shared by producer and consumer. Its small methods express
  missing, future and too-old version outcomes. Mechanism tests allow zero lag,
  while production settings require positive lag and strict mode handles zero.
  Keep this distinction and provider/count boundaries; do not add a generic
  checker class or delete all existing numeric tests by pattern.
- Consumer selection uses prompt_batch_id and complete unique group slots, not
  whichever prefetched batch finishes first. It checks one policy version,
  removes only the selected receipts, then remaps group IDs for advantage
  normalization. These cross-receipt checks cannot be done by one receipt's
  constructor. Keep the health check, selection and iteration assembly methods
  as separate owned operations rather than a monolithic polling loop.
- Timeout polling uses the remaining budget. Fatal producer errors bypass the
  retry threshold; regular fail-fast compares fresh errors/completions since
  demand. The monotonic age gauge is process-local, not a distributed timestamp.
- __all__ and metric keys are public/schema declarations. These modules contain
  no ALL_CAPS algorithm vocabulary to relocate. The lack of payload ownership
  in the capacity class is explicit, not hidden behind a queue-like name.

Non-goals: redesigning admission, replacing polling with another distributed
queue, changing staleness or batch identity, merging capacity into producer just
to reduce files, or claiming cancellation preempts a GPU kernel.

## Validation and limits

140 capacity, queue, staleness, identity and producer/consumer contract tests
passed. Coverage includes retained capacity during reward retries, frozen receipt
identity, overflow before mutation, named-head selection amid ready prefetch,
duplicate/future/mixed versions, failure propagation and bounded stop. No tests
were added for documentation edits. These CPU tests do not prove GPU reclamation
after abandoned work. The terminal owner/runtime teardown must establish that
separately; a zero capacity statistic is not evidence of memory release.
