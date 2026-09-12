# Online replay metrics (scoped review)

Following d201fb048, inspected the online trainer's opening helpers and records:
reward reduction, work agreement, PhaseTimer, TrainingBatch, _ReplayMetrics,
_ReplaySampleBatch, scalar/parity/initial-replay reductions, construction and lazy
optimizer/EMA ownership. Followed _ReplayMetrics through _run_replay_pass and
_backward_sft_regularizer, plus both callers' work-agreement gates. The full online
trainer remains pending; the remainder of collect/train/state orchestration still
needs its complete review.

Change: _ReplayMetrics.add reads the weight directly instead of converting it to
a second local and checking positivity again. Its production caller passes a
_ReplaySampleBatch weight only after excluding dummy batches. Real weights are
1.0 for a whole nonempty group or slice_count / group_count for a nonempty slice.
The private accumulator is not an input parsing boundary.

Remove the explicit parallel-list length check inside weighted_mean. add appends
all metric values and their weight together; no production caller independently
mutates those lists. The existing strict zip remains the pairing operation. Do
not add a test that mutates private lists into a state no producer creates.

Correct the initial-replay reduction comment: a rank without measurements is
neutral there, but replay planning does not synthesize batches for an entirely
empty rank. Both training paths first call _all_ranks_have_work; the planner uses
a local real batch to pad unequal nonempty slot counts. The diagnostic reduction
can still handle empty snapshots, as its existing tests exercise.

Retain and why:

- _ReplayMetrics is an existing accumulator owner used by streaming and full
  batch replay. Keep it; do not replace it with a new checker or metrics schema.
- Uneven replay slices need weighted loss/clip means. The existing 8+2 split test
  verifies 80%/20% contribution, while maximum mismatch keeps max semantics.
  Empty initial snapshots and zero-weight diagnostic ranks have real callers.
- The nested weighted_mean shares the local denominator across four metric
  fields. It removes actual arithmetic duplication without another public class.
- _all_ranks_have_work expresses a training collective admission decision;
  _distributed_all_true expresses the shared reduction used by other verdicts.
  Retain their separate names. Combining them into a generic checker would hide
  why every rank must execute the same branch.
- _ReplaySampleBatch owns sample slicing, contribution weights and dummy slots.
  Padding keeps distributed forward/backward counts aligned. Keep these class
  methods rather than scattering that state through both replay paths.
- PhaseTimer owns accumulated timings and events. Its context manager is a real
  lifetime boundary, including optional synchronization, not a thin namespace.
- This reviewed section has no module-level business vocabulary constants.
  Collective op and dtype choices are framework semantics, not a config table.

Non-goals: change loss normalization, gradient accumulation, distributed skip
policy, precision checks, optimizer choice, or move every trainer helper into a
class. No new checks or tests were added.

Validation: 38 existing tests passed in advantage/metrics, diagnostics, SFT
regularization and distributed skip/backward agreement modules. This CPU run
includes Gloo process tests, not GPU/NCCL validation. Ruff check and format check
passed. Coverage stays at 190 reviewed modules while online/trainer.py is pending.
