# Token-step protocol and sequence composition

Reviewed complete generation/steps/token/protocol.py, composition/token_autoregressive/
token_loop.py and both package initializers. Checked runner consumption, executor/
family call sites, cache-row operations and complete protocol/loop tests.
Previous audit commit: 323ff4828. Disposition: retain implementation.

## Retain and why

- TokenLoopInit, TokenStepBatch and TokenStepOutput define a real model/composition
  boundary: opaque family state, scheduled row subsets and published lane updates.
  Basic index/count checks differ from runner upper-bound checks against current
  token buffers. Those upper bounds are unavailable to TokenStepBatch.
- TokenAutoregressiveEnvelope owns per-lane ARCacheRows. Its factory splits initial
  batched values; build_step_batch gathers selected rows; apply_step_output scatters
  updates. Keep stateful routing here rather than duplicating it in family models.
- Unknown output names are checked before any scatter. Existing tests cover both
  dictionary orders; merging preflight and mutation would allow partial updates
  before an unknown key fails.
- The loop owns initialization, position-major bounded row batches and finalization.
  None means all rows; an oversized batch still schedules one batch. This is
  synchronous token scheduling, not asynchronous cross-batch prefetch.
- Protocol exports form a public facade. The composition initializer preserves
  direct token_loop imports. __all__ lists API names, not business vocabulary.

Non-goals: another class per hook, automatic retries, changed token/RNG ordering,
or unchecked fast-path flags merely to avoid repeated row checks across lanes.

## Validation and limits

30 protocol/composition tests passed on CPU. Deterministic runner tests verify
position-major scheduling, uneven final batches, state identity, output routing
and unknown-name rejection before mutation. No Python changes were needed; no
new tests mirroring the implementation were added. This is not GPU throughput or
pretrained numerical-parity evidence.

Dataclass fields/lists remain mutable. ARCacheRows independently validates index
range/type because it is also used outside this composition. Consolidation would
need a shared validated-selection contract, not weakened public APIs.

Output application is not transactional across valid lane names: a later invalid
value shape can fail after an earlier scatter succeeds. Family state may already
have changed during step_token too. The loop propagates errors without retry;
unknown-schema preflight is not a general rollback guarantee. Backend failure/
cancellation cleanup remains a lifecycle question for executor/runner owners.
