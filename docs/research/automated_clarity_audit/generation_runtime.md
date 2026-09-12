# Generation runtime lifecycle

Reviewed complete `generation/ray/runtime.py` with the shared lifecycle state,
request sample-count constructor, launcher/session/monitor composition and
existing activation, parking, cancellation, publication and probe tests.

## Changes

- Auto-width probing now passes request.samples_per_prompt directly. Its
  GenerationRequest constructor already requires a positive integer. Repeating
  int conversion and clamping here obscured that contract. Valid requests behave
  identically; invalid post-construction mutation is no longer silently repaired
  by this probe path. The request dataclass is mutable, so this is not a claim
  that arbitrary later mutations are impossible.
- Preflight's health-RPC submission moves inside the same failure handler as
  its await. Previously a synchronous submission error propagated while leaving
  lifecycle RUNNING and failure unset, despite preflight's stated failure policy.
  Two controlled regressions (RuntimeError and early TimeoutError) failed before
  the fix and now require the original error plus closed admission. Outer
  teardown still owns resource release, as for existing preflight await failures.
  The deadline now starts before submission and diagnostics count owned ranks;
  this includes submission time in the budget but cannot preempt synchronous
  submission. A probe-less fake returns after deadline construction rather than
  bypassing timeout construction entirely.

## Retain and why

- Accepted current_policy_version, installed version and pending payload have
  different consumers. A parked runtime can accept a new target before actors
  install it. `_PendingPolicyInstall` keeps that payload/version pair together;
  its frozen dataclass does not deep-copy or freeze tensor contents.
- Publication guards hold the shared lifecycle lock only across local assignments.
  A health thread can close admission while a remote update is running; successful
  RPC completion alone must not publish over that failure. Keep these guards.
- Probe caching checks before and after the async lock. The second check closes
  the concurrent first-request race; it is not redundant validation. Capacity is
  cached once per runtime from the initial request, not proven for every later
  possible resolution/sequence shape. OOM fallback remains the executor's concern.
- Activation/offload/shutdown tasks are shared operations protected from caller
  cancellation by shield. Their three completion callbacks clear only their own
  current task and consume exceptions; a reflected attribute-name helper would
  hide concrete state ownership for little gain. Keep the uniform typed shape.
- `_finish_control_wait_failure` distinguishes cancellation of the current waiter
  from cancellation/failure of shared cleanup. `_publish_failure` keeps the first
  failure but upgrades teardown when a later terminal transport error requires
  force-close. Merging these with ordinary method error handling would change
  cause retention or make cleanup await itself.
- Candidate sessions are not published until pending weights and capability
  checks succeed. Failed candidate cleanup retains the session for retry.
  `_shutdown_once` waits on activation/offload without joining itself, and
  `_teardown_session` clears session state only after close succeeds. Preserve
  these resource-publication boundaries.
- Repeated state checks after awaits observe new lifecycle state. Schedules own
  pause/drain admission; this runtime owns terminal admission and resource state.
  Don't remove checks simply because the same condition appeared before an await.
- Public capability properties and _owned_ranks are real protocol/adapter views.
  _RaySessionFactory is a type alias, not a workflow data table; __all__ declares
  the public API. No ALL_CAPS algorithm/backend vocabulary is mixed into this file.

Non-goals: introduce another lifecycle manager, unify accepted and installed
versions, add user-facing probe knobs, change same-version update semantics,
or claim serialized method bodies alone make remote operations atomic.

## Validation

After removing probe coercion, runtime config/lease/lifecycle and batch-probe
tests: 195 passed, 1 GPU case skipped, upstream Ray/SWIG warnings. This includes
concurrent auto-probe sharing and actual request ceiling forwarding.
After the preflight fix, lifecycle and lease tests: 96 passed, one upstream Ray
warning. Both new preflight submission regressions pass. Ruff check and
format check passed for both changed Python files. No GPU training was run.
Previous isolated audit commit: `39c699d6d`.
