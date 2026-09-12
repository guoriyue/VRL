# Engine fan-out ownership

Reviewed the complete `generation/ray/engine.py`, engine tests, dispatcher
submission handling, runtime terminal-error classification and session parking
callers. This completes one production module; dispatcher/session/runtime source
excerpts do not count as full reviews of those modules.

## Change

Three list-comprehension submissions could lose already-submitted rank refs if
a later rank's `.remote()` raised synchronously. No EngineCallRef reached the
dispatcher, so the dispatcher could not cancel those children. Its first-job
submission failure path also could not distinguish this partial remote work
from a wholly unsubmitted local failure.

One engine-owned `_submit_rank_calls` now accumulates refs across submission.
If submission fails before any returned ref, preserve the original exception.
If earlier refs exist, request best-effort cancellation and raise the existing
TerminalRuntimeError with the original submission error as its cause. Runtime
generation handling recognizes that marker and closes admission/tears down the
session. Keep the count and method in the diagnostic; cancellation failures are
attached by the existing cancellation utility.

Reuse this operation in multi-rank remote dispatch, sleep and wake. This helper
is justified by shared ownership and cleanup, not by extracting a short list
comprehension for appearance. Single-rank remote dispatch still returns the
actor's raw submission method, preserving completion/cancellation behavior.

Existing test prose claiming a multi-GPU backend did not yet exist was stale;
replace it with the actual scope of the fake-rank tests.

## Retain and why

- EngineCallRef waits for every rank and cancels sibling refs on raised failures.
  A rank-zero-only wait can hide nonprimary failure. `child_refs` is the shared
  cancellation adapter surface; generic Ray code need not import engine code.
- `uniform_rank_result` is a combiner factory capturing the method for errors.
  It checks agreement, not model contents. Generation result combiners must
  additionally inspect error payloads, whereas weight-sync callers validate the
  agreed ACK against the requested version. Keep those policies separate.
- `primary` supplies the established rank-zero view without another stored
  field. `rank_handles` is the shared flattening API used by launcher/session
  lifecycle code; no new rank-collection class is needed.
- Sleep returns all rank snapshots, checking each worker identity and physical
  handoff evidence. Wake has no corresponding snapshot payload. Keep their
  distinct result handling even though submission is now shared.
- Export names are API declarations; no ALL_CAPS business data occurs here.

Non-goals: change rank-combination policy, promise atomic fleet execution,
implement rollback, alter async await failure semantics, or claim cancellation
interrupts a running synchronous GPU kernel. The existing runtime teardown owns
actor termination. A call accepted remotely but failing locally before returning
any ref remains unknowable here; the fix covers refs this layer actually obtained.

## Validation

The new matrix covers failure on rank zero and rank one for execute/sleep/wake.
Before the fix, the three partial-submission cases failed and the three wholly
unsubmitted cases passed. Afterward, engine and weight-sync tests passed: 61
tests including real CPU Ray weight transport; one upstream Ray warning.
Fake submission tests verify ref ownership, cause retention and no later rank
submission, not physical GPU cancellation. Ruff check/format check passed.
Lifecycle/parking regressions: 93 passed, one upstream Ray warning. These
exercise existing terminal-error classification and session teardown separately
from the controlled submission-failure matrix; they are not a real GPU
mid-submission fault-injection experiment.

Previous isolated audit commit: `eb293f236`.
