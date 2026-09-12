# Placement, rank lifecycle and probe anchor failure

Reviewed complete execution `batch_placement.py`, `rank_group.py` and
`__init__.py`, their launcher/executor callers and existing tests. Also read the
complete worker source, but keep its ledger entry pending until staged weight
transfer and version-slot caller contracts have been followed in full.

## Change

The batch-size probe accepted a successful warmup as proof that the later
single-sample fit trial succeeded. If that second trial OOMed, a ceiling of one
returned a successful capacity anyway; a larger ceiling could reach the missing
peak assertion or use a failed lower bound for bisection. Existing warmup-OOM
coverage did not cover this distinct failure.

Run both single-sample anchors through one loop and reject either OOM before
fitting or returning capacity. Diagnostics include the failed phase. Two new
regression cases failed before the fix (false success and AssertionError) and
pass afterward. Successful trial order, memory fit, confirmation and throughput
knee math are unchanged. No new validator helper or probe wrapper was added.

## Retain

- DistributedExecutionPlanner separates fleet placement from neutral sample
  planning. Dynamic assignments deliberately leave engine_id unset; the actual
  dispatcher consumes estimated_cost as priority. DeviceAssignment.batch reads
  the authoritative envelope and prevents duplicated batch state. Keep this
  property and the frozen assignment/plan records.
- Engine-ID uniqueness prevents ambiguous engine lookup; nonempty membership
  prevents an unusable plan. Strategy validation covers direct planner
  construction in addition to config parsing. These are entry checks, not a
  second per-assignment validation loop.
- RankGroupSpec is the process-launch rendezvous boundary. Port/rank/world
  checks guard the concrete collective coordinates; init/destroy are lazy
  torch.distributed adapters. Reject double initialization so teardown cannot
  claim another owner's group. Worker ownership is set only after init succeeds.
  An additional manager class would duplicate the worker's lifetime state.
- nccl/gloo are the explicitly supported transport vocabulary here, not a
  model taxonomy. Export lists are API declarations. The execution package
  initializer intentionally has no eager re-exports; preserve direct imports.
- Worker model identity checks straddle actual model construction and thus
  check different observations; do not collapse them because expressions match.
  Per-batch errors and pipelined typed OOM retries have different wire/control
  contracts; do not merge their handlers on superficial similarity.

## Follow-up and non-goals

Worker's three `_PROBE_*` settings are fixed probe policy mixed into workflow
code. Their values are not architecture dimensions or external protocol keys.
A future probe-owner cleanup should move the policy with the complete trial,
fit and confirmation operation, after auditing runtime auto-width caching;
moving only constants to a new tiny file would add navigation without ownership.
Do not add user-facing tuning knobs or change numeric values during that move.

Keep worker coverage pending for staged transfer/version-slot ownership review.
Do not interpret the CPU tests as proving GPU memory capacity, NCCL placement,
cross-node rendezvous, or the validity of two-point extrapolation for every model.
The single-sample fix changes only an unsuccessful probe's outcome. Other probe
heuristics, including smaller-than-high candidate selection, remain unchanged.

## Validation

- Placement, rank-group and launcher tests: 81 passed, including real two-process
  CPU gloo rendezvous/RNG agreement and the embedded Ray launcher; one upstream
  Ray environment warning. No GPU training was run.
- After the probe fix: 102 passed, 2 CUDA-dependent cases skipped across batch
  memory/probe, worker sleep and pipelined request tests.
- Ruff check and format check passed for both changed Python files.

Previous isolated audit commit: `568549b9a`.
