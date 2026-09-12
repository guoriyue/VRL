# Batch memory sampling and fit

Reviewed complete `vrl/generation/execution/batch_memory.py`, the loop's capture
point, worker probe/reading consumers and existing memory-shadow tests.

## Change

Remove per-capture reflection checking whether the function's four literal
dictionary keys exist on BatchMemoryReading. This compares two source schemas
on every CUDA capture, not untrusted runtime data. Return the same occupancy
mapping directly. It also removes the otherwise unused wire-type import from
the sampler module. BatchMemoryReading's canonical home remains execution.types;
it was not in batch_memory.__all__.

No measurement fallback or exception handling changes: off-CUDA still returns
None, and failed CUDA queries still propagate. Complete worker readings still
require all fields. A direct controlled capture plus from_metrics assembly
verified the emitted mapping produces the expected phase peak, budget and
non-torch footprint.

## Retain

- `cuda_occupancy_snapshot` is a genuine time-of-measurement boundary: occupancy
  must be sampled before denoise, while phase peaks are only available afterward.
  Keep it separate from completed wire-record construction; no new partial-record
  class or generic CUDA telemetry wrapper is needed.
- AffinePeakFit owns the two-point fit and its capacity prediction. Its guards
  prevent a zero denominator and an invalid request ceiling. Non-growing fits
  use the request ceiling, and the worker confirms actual capacity with a trial.
  Do not turn an estimate into a guarantee or remove confirmation based on the
  affine formula. Allocator behavior and other-process occupancy remain variable.
- The four mapping keys and export names are telemetry/API schema, not business
  vocabulary. There are no ALL_CAPS tables to move in this module.

Non-goals: change batch-size selection, memory budgets, CUDA query order, probe
confirmation, or introduce additional scalar/type checks.

## Validation

81 existing memory-shadow and preallocation tests passed; 3 CUDA cases skipped.
The direct capture/assembly exercise used mocked CUDA readings, not physical
memory measurements. Ruff check/format and diff whitespace checks passed.
No new tests mirroring the removed source-schema comparison.
Previous audit commit: `28d2fc9f7`.
