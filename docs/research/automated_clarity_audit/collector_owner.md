# Collector orchestration and shutdown

Reviewed complete collector/core.py and the rollouts, collector and orchestration
package initializers, strict/continuous collection call sites and runtime shutdown
forwarding, with collector and prompt collection regression coverage. Previous
audit commit: 99058b5b8. Batch-builder internals remain a separate pending review.

## Change

Simplify shutdown to sequential awaits followed by success-only state updates.
Its error list never aggregated failures: generation failure immediately stopped
before reward shutdown, and reward failure immediately propagated. Catching each
BaseException merely to store and rethrow it added no cleanup. Direct propagation
preserves cancellation, safe generation-before-reward ordering and retry state.

## Retain and why

- RolloutCollector already owns request/generation/reward handoff and prompt
  grouping for both schedules. Its from_family constructor binds family request
  and trajectory semantics. Do not split these into more single-use owner classes.
- GeneratedPromptGroup and UnscoredRollout hold actual deferred-work receipts,
  prompt identities and per-call timings. Keeping those with the work prevents
  mutable last-result caches from crossing concurrent collection calls.
- Nested generation and scoring functions capture call-local timings, receipts
  and task ownership. Moving them onto the shared collector would require storing
  per-call mutable state on an object used concurrently. Their locality is useful.
- Capability properties combine topology with reward-runtime isolation and
  nonblocking execution. An acceptance mode can restrict a safe collector but
  cannot grant accelerator isolation. Enum values are an intentional execution
  protocol, not workflow vocabulary requiring an asset file.
- Streaming owns one scoring task and drains it before dispatching another;
  serial-per-group is a measurement control, and batched-serial preserves shared
  GPU phase handoffs. These modes have distinct observable scheduling behavior.
- offload_generation_runtime_memory genuinely attempts both required parking
  operations and aggregates failures. Its error list is necessary, unlike the
  removed shutdown list. PromptCollectionCleanupError likewise preserves both a
  generation root cause and a scoring cleanup failure.
- finish_scored_prompt_groups centralizes timing accumulation, prompt remapping
  and splitting for strict and continuous consumers. Profile keys and reward
  component metadata are schema fields; VRL_PROFILE is an environment boundary.
- Package initializers preserve public import paths. The root facade is doc-only;
  collector and orchestration export established types and entry points. No new
  wrapper or registry is warranted.

Non-goals: change overlap policy, admission, staleness, reward batching, output
placement, prompt interpretation or shutdown concurrency semantics. Repeated
sequential shutdown is supported; concurrent shutdown serialization is not added.

## Validation and limits

59 collector runtime and prompt collection tests passed on CPU. The existing
shutdown regression fails generation first, then reward, checks each retained
state and verifies later repeated shutdown calls do not repeat successful work.
Other tests cover overlap, serial controls, identity remapping and cancellation
settlement. Ruff check/format check pass for core.py. No new test mirrors the
removed exception-list implementation.

Open: cancellation followed by gather does not itself bound a scoring coroutine
that suppresses cancellation indefinitely. The code comment's intent to avoid
an indefinite wait is not a timeout guarantee. Preserving cleanup attachment is
necessary; solving an uncooperative runtime needs its lifecycle contract rather
than detached tasks or an arbitrary timeout in this collector.

Open: streaming retains generated receipts through final accounting even after
individual scores finish. The single scoring-task bound therefore does not imply
that only one generated artifact group remains resident. A memory cleanup must
separate timing/identity data from owned output lifetimes with downstream batch
aliasing understood; it cannot simply clear receipt payloads after scoring.
