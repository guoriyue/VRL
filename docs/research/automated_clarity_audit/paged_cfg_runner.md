# Paged CFG loop and shared row helpers

Reviewed complete token/paged_attention_helpers.py, Janus runner, scoped Emu3
sampling, NextStep advance and GLM cache callers, attention input/output contracts,
CFG score tests and scheduled Janus loop tests. Previous audit commit: 5865bfe8a.
Disposition: retain implementation; family runners and attention backends remain
pending full independent review.

## Retain and why

- PagedCFGTokenRunner already owns the shared stateful sequence: prefill both
  branches, sample, record policy log-prob, advance KV state and publish row lanes.
  Janus and Emu3 own head projection, structural constraints and token embeddings.
  Moving those differences into a family-name table would worsen ownership.
- prefill_ar_prompt constructs a typed backend request consistently for CFG and
  NextStep. append_attention_token preserves mask dtype/device; it is also used
  by GLM's ordinary cache path. These are shared adapters, not state owners.
- select/scatter helpers serve separate state dataclasses, including NextStep's
  optional unconditional branch and GLM kv_rows. They cannot simply become methods
  on PagedCFGARState. Their None check catches an absent initialized cache at the
  consumption boundary. scatter's length check precedes mutation; strict zip alone
  would fail only after writing the shared prefix of unequal sequences.
- normalize_paged_last_hidden accepts the two actual backend forms, [B,H] and
  [B,1,H], and rejects a non-singleton sequence axis. It does not catch a tensor
  operation failure and retry a different representation.
- The shared CFG path samples guided logits but scores the conditional policy,
  applying temperature consistently. Emu3's mask is applied to both distributions.
  These are objective semantics, not removable casts or redundant score functions.
- Backend state-count validation occurs before scattering the two branches.
  Row buffers preserve original scheduler indices; compact backend batch order
  must not become the global row index. The final sampled token skips another
  attention step because no later token consumes its hidden state.
- State dataclasses and abstract hooks preserve cross-family protocol shape.
  __all__ is the only module-level ALL_CAPS list and declares public names.

Non-goals: changing sampling math, joining no-CFG/continuous NextStep into the
discrete two-branch loop, or creating a cache manager solely to replace list
selection helpers. The paged-oriented helper names are broader in practice for
GLM; a future cache ownership consolidation should rename them together with
that API rather than introduce forwarding aliases here.

## Validation and limits

18 Janus/Emu3 CFG and NextStep runner tests passed on CPU. Four scheduled Janus
paged loop tests also passed, exercising prefill/advance ordering with a recording backend.
These validate scheduler and conditional-score behavior, not a real vLLM GPU KV
allocator. No source changes or new implementation-mirroring tests were needed.

Deferred boundary findings: output contracts do not themselves validate hidden
batch length. CFG normalizes hidden rank after scattering cache updates; a broken
backend can therefore leave updated states before hidden validation fails.
NextStep's no-CFG advance slices the returned states and can ignore excess states.
These require a common backend-result contract review, not another per-runner
validator. Backend failures are not currently a transactional rollback boundary;
no recovery guarantee is inferred from these checks.
