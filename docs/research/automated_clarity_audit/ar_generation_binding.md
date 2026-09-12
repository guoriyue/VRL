# Token-autoregressive binding

Following 10923eee4, reviewed token_autoregressive/executor.py, layout.py and
__init__.py. Traced layout calls in Janus, Emu3, LlamaGen and NextStep, LlamaGen's
scheduler bound override, and NextStep's separate continuous-token gatherer.

Change: rename ordered_ar_chunks to ordered_batches in the discrete gatherer and
its NextStep counterpart. These are prompt/sample transport batches, not temporal
chunks or denoise transitions. Both keep the same ordering, concatenation and
context assembly. NextStep runtime is only scoped to this gatherer here; its
whole-module disposition is not changed by a local-variable rename.

Retain and why:

- ARBatchExecutorBase supplies common request/runner wiring; the discrete subclass
  adds a shared prefill, token loop, decode and result pipeline. NextStep continuous
  tokens and Janus-R1 refinement have genuinely different control flow and remain
  on the simpler base. Do not force them through the discrete template.
- ARBatchInputs is the family hook's actual data contract: loop arguments, decode
  settings, prompt tensors and context. ARDiscreteBatchResult removes a formerly
  duplicated field set across discrete families. Keep both records and the gather
  protocol rather than another result-wrapper class.
- The layout owns family defaults and common parsing/padding/sample assembly.
  _sampling_int reads request dictionary values at that boundary; it does not guess
  a missing required field from unrelated state. Right-padding preserves token
  masks and never truncates; tokenizer truncation is a different responsibility.
- validate_batch and batch_seed_offset keep one shared prompt-major contract for
  several executors. cat_batch_fields is used by discrete, continuous and refined
  gatherers. These small methods provide real cross-family consistency.
- resolve_scheduler_batch_size's base signature includes row_count for family
  policy. LlamaGen overrides it to require the full native static-KV row count.
  Its base del row_count is therefore not evidence that the parameter is useless.
- _build_ar_runner distinguishes native cache owners from runners accepting a
  shared attention backend. Family backend identity can differ from public model
  family (Janus-R1 uses Janus architecture). Keep explicit declarations and reject
  ignored backend requests instead of silently guessing an implementation.
- _embed is a common language-model embedding adapter used by family hooks.
  chunk_token_mask remains an override seam for Emu3 structural positions.
- The gatherer's field tuple is a result schema shared between row validation
  and concatenation. __all__ lists are API facades. No workflow ALL_CAPS business
  vocabulary needs extraction or an additional configuration asset.

Non-goals: change sampler seeds, cache scheduling, family defaults, prompt padding,
attention backend selection or trajectory schema. No classes, guards or tests
were added. Do not collapse protocol overrides merely to remove del statements.

Limits: the layout property currently builds a lightweight object per access;
this is not evidence of a meaningful performance issue. The discrete executor
sets process-global Torch RNG before family preparation; this review does not
claim concurrent-call RNG isolation or sample invariance across different batch
partitions. Such a change needs the complete runner RNG contract, not a local
seed-name cleanup.

Validation: 132 existing scheduler-batching, batch-gather, NextStep parsing and
runtime-input tests passed on CPU, with three warnings. Ruff check and format
check passed for the two changed Python files. No GPU/paged-kernel execution is
claimed. Coverage: 209 reviewed, 290 pending baseline modules.
