# Continuous owner commands and thread lifetime

Reviewed complete continuous/owner.py, schedule.py and __init__.py; coordinator
snapshot/cleanup interfaces; producer batch-install/stop interfaces; existing
owner-thread and schedule tests. Previous audit commit: a334c7563.

## Changes

Keep prefetched_prompt_batch local to next_iteration's operation. It is created
and promoted to _installed_prompt_batch within the same serialized command;
no other method reads it. On command failure the owner terminalizes rather than
resuming partial demand. A long-lived field and its reset assignment therefore
added state without adding ownership. Producer batch state remains authoritative
for the actual in-flight work; the installed owner receipt still spans commands.

Pass self._stop_pipeline directly to _run_command for reset, removing an inner
coroutine whose entire body was one awaited call. The command wrapper still
owns locking, failure handling and active-command accounting. No public API or
timing behavior changes, and no new class is introduced.

## Retain and why

- ContinuousRolloutSchedule performs trainer-thread export and hands prepared
  snapshots to the owner. Its otherwise thin methods are protocol/thread
  adapters, not redundant facades to inline into the trainer.
- ContinuousRolloutOwner owns thread, loop, cross-thread futures and startup/stop
  events. _ContinuousOwnerRuntime owns only loop-local pipeline state. Collapsing
  them would mix two synchronization domains; the lock and command lock are not
  duplicate locks protecting the same access.
- _run_command provides a real shared operation lifetime. next_iteration and
  commit_weights closures contain multi-step transactions; unlike reset's removed
  wrapper they cannot be replaced by a single existing bound method.
- _await_command shields owner work from trainer-waiter cancellation. A cancelled
  waiter does not mean the owner command stopped. _shutdown_runtime publishes
  completion outside the thread lock before stopping the loop, while failed
  cleanup leaves it available for retry. Keep the single shared shutdown future.
- Terminal cleanup has an independent task so concurrent command failure and
  explicit shutdown join one attempt. A done callback retrieves failures even
  without surviving waiters. Keep that concrete task-lifetime mechanism rather
  than introduce generic callback managers or duplicate cleanup calls.
- _InstalledPromptBatch compares presented prompts without treating non-scalar
  tensor equality as truth. It holds a shallow tuple, not deep immutable prompts;
  identity acceptance is intentional. Group-size agreement remains separate.
- The weight barrier never resumes admission in finally after a failed push.
  Draining and version-slot modes have different wait requirements but both
  validate ready versions before reopening admission. Keep these state checks.
- _attach_producer_metrics translates owner-local counters into cumulative
  gauges. The historical lookahead metric key is a persisted schema name, while
  code uses prefetch terminology. Do not rename saved columns for code style.
- _OWNER_START_TIMEOUT_S/_OWNER_STOP_TIMEOUT_S are thread-lifecycle budgets and
  _MB implements the configured binary byte conversion; __all__ and package
  exports are public facades. No business taxonomy is mixed into these constants.

Non-goals: moving collectives to the owner loop, changing cancellation semantics,
removing protocol facades, or introducing another runtime class solely for naming.

## Validation and limits

76 owner and schedule tests passed. They cover early prefetch across weight
versions, mismatch rejection, no-prefetch demand, checkpointed prompt replay,
cadence while the trainer loop blocks, failed commits, shared cleanup, cancelled
shutdown waiters and cleanup retry. Ruff check and format check passed for
owner.py. No tests were added just to inspect the removed field/wrapper.

Thread finalization still cancels and gathers remaining loop tasks without its
own timeout. The caller's stop wait is bounded, but a cancellation-resistant
task can keep the daemon thread alive. This is not proof of completed resource
release; the capacity caveat in continuous_capacity.md still applies. Finalizer
exceptions can also bypass _stopped.set because it follows loop cleanup rather
than an outer finally. Review unexpected thread-exit reporting and noncooperative
task teardown together before changing the stopped-event contract.
