# SPRINT: Function organization audit

> **Conclusion first.** A conservative 12-package audit of `vrl/` for "disorganized
> standalone functions" (free module-level functions scattered away from their owner
> class or sibling family) found the repo **already well-organized: 10/12 packages
> clean, with 2 medium candidates**. Revalidation found one genuine fix (done) and one
> false positive whose proposed move would have broken precision ownership. The worry
> that there are
> "too many single separate functions" does not hold up at the architecture level —
> the free functions are overwhelmingly legitimate (protocol/facade, lazy-import
> boundary, framework adapter, registry/dispatch, stateless shared utility, one-shot
> script, or a cohesive symmetric family). The durable value of this sprint is the
> **organizing principle** (so new code stays organized) and the **non-goals catalog**
> (so a future "cleanup" does not flatten the protected abstractions the owner has
> reverted before). Audit: 12 parallel package reviewers, 2026-06-29, findings
> verified against source (file:line cited).

> **Current-path revalidation (2026-07-30).** The dated findings below remain the
> historical record of what was reviewed, but subsequent cleanup moved or deleted
> several owners. The protected-family catalog has been revalidated against current
> HEAD and now names current paths. Historical line numbers are evidence for the
> original decision, not an executable task list.

## The organizing principle — what "organized" means here

A function lives in a coherent place when it is **one of**:

1. **A method** on the class that owns its state/behavior.
2. **A co-located sibling** in a cohesive single-concern module — a *symmetric family*
   (e.g. a split/concat pair, a `build_*` family, a set of `RolloutBatch` ops, a
   resolver's `_resolve_*`/`_validate_*` stages). The module itself is the namespace.
3. **A sanctioned module-level function**: a protocol/interface or public-API facade,
   a lazy-import boundary, a framework adapter (Ray/torch/pydantic/OmegaConf), a
   registry/dispatch entry, a stateless utility genuinely shared by **many** callers,
   or an active one-shot procedural artifact kept outside the production import graph.
   A probe/smoke/spike is retired after its answer is recorded; stable data/eval CLIs
   are long-term assets rather than one-shot exemptions.

The **smell** is only: a free function that floats apart from its owner class **or**
its sibling family; **or** a grab-bag file of unrelated free functions. Outside the two
candidates below, neither was found.

**Current hard rules:** a single caller is an audit signal, not an automatic exemption.
Merge a single-caller concept split unless the helper is a real lazy-import boundary,
public facade, framework adapter, cross-family implementation seam, or a concept-naming
extraction that materially shortens a long owner flow. Never flatten a registry,
convention, or cross-family abstraction merely to reduce LOC; consistency can be the
more valuable property.

## The 2 candidates (one fixed, one correctly rejected)

### Fix 1 — `[med]` precision-resolution cluster → **historical verdict, later superseded**

The 2026-06 audit proposed moving a nine-function trainer cluster into the then-current
precision helper. That exact cluster and module layout no longer exist. Current ownership is
clearer:

- `vrl/config/precision.py` is the torch-free source of truth for public precision schema,
  normalization, and resolved `PrecisionPolicy`;
- `vrl/models/precision.py` is the PyTorch execution boundary for autocast and FP32 backend
  policy;
- `vrl/trainers/online/precision_guard.py` owns rollout-vs-replay drift measurement;
- `vrl/trainers/online/trainer.py` retains only trainer-owned GradScaler decisions and
  diagnostic/model-introspection metadata near their consumers.

Moving the remaining trainer functions into any of those siblings would mix config
resolution, model execution, drift validation, and trainer diagnostics. A new
`precision_runtime.py` would only hide that ownership split behind another thin file.
**Decision remains: leave the current cohesive functions with their consumers.**

### Fix 2 — `[med]` one evaluator broke the sibling on-class convention → **DONE**

The historical implementation now located at
**`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py`** kept five
evaluator-only helpers (`_segments_from_batch`, `_trajectory_segment_payload`,
`_extract_logprobs`, `_segment_tensor`, `_primary_segment_name`) as module-level free
functions. They had zero external callers and operated entirely on the evaluator's
batch/segment data, while sibling token evaluators kept equivalent implementation
details on their classes.

- **Fix (done):** pulled the five helpers onto `MultiSegmentTokenLogProbEvaluator` as
  `@staticmethod`s, matching the sibling evaluators; call sites now `self._foo(...)` (the
  two staticmethod→staticmethod internal calls reference the class). Pure relocation, no
  behavior change. Verified the 5 had **zero external callers** (the 5 grep hits for
  `_primary_segment_name` are unrelated same-named *methods* on `trajectory.py` /
  `batch_builder.py` — which actually confirms on-class is the codebase-wide convention).
  The current regression test is
  `tests/rollouts/replay/test_multisegment_token_logprob.py`.
- **Constraint kept:** NOT moved to a shared module — they are not shared; on-class is the
  established convention in `vrl/rollouts/evaluators/token/`.

## Non-goals — DO NOT "clean up" these (legitimately standalone, by package)

The audit explicitly cleared these; they satisfy the principle above and several are
abstractions the owner has reverted cleanups on. Listed so a future sweep does not
mistake them for scatter.

| package | legitimately-standalone families (keep as-is) |
|---|---|
| **config** | `vrl/config/builders.py` public `build_*` projections and typed-schema validation stages; `vrl/config/precision.py` resolved `PrecisionPolicy` family (deliberately torch-free); bundled-config loader in `loading.py`; shared readers and production-reward validation chain in `validation.py`; unknown-key and pydantic boundaries in `unknown_keys.py` / `schema.py`; `lint.py` CLI |
| **ray** | lazy-Ray and actor-metadata adapters in `vrl/ray/dependencies.py`; lifecycle family `kill_actors` / `kill_and_retain` / `remove_placement_group` in `resource_cleanup.py`; monotonic operation protocol in `operation_deadline.py`; placement helpers in `placement.py`; state-owning `RayActorDispatcher` plus deprecated public facade in `actor_pool.py`; pure resource resolver pipeline in `resources.py` |
| **trainers** | unit-tested trainer-local collective, precision-metadata, and sample-chunk planning helpers in `vrl/trainers/online/trainer.py`; drift guard in `online/precision_guard.py`; `distributed.py`, `strategy.py`, `fsdp.py`, `weight_sync.py`, `metrics_io.py`, `data/*`, and `checkpointing.py` each retain one concrete owner. PyTorch precision execution remains in `vrl/models/precision.py`, separate from torch-free config resolution |
| **rollouts** | symmetric `RolloutBatch` operations in `vrl/rollouts/batch/ops.py`; dependency-light `RolloutBatch` data contract in `batch/core.py` (the deleted `stack` / `_stack_*` helpers must not return); `vrl/rollouts/orchestration/types.py:build_rollout_iteration` remains a typed factory while context mutation belongs to `RolloutIteration.annotate_batch_context()` |
| **families** | `vrl/families/registry.py` is the unique composition taxonomy for model builds, generation bindings, gatherers, and static runtime capabilities. `FAMILY_REGISTRY`, dotted import paths, and `_VAE_DECODE_MEMORY_SECTIONS` are registry/protocol boundaries; do not move the registry back under rollouts or recreate a generation-side capability table |
| **generation** | subject-less chunk domain in `vrl/generation/execution/chunks.py`; public structural protocols in `protocols.py`; Ray/in-process wire and completion schema in `execution/types.py`; cross-family execution shapes in `bindings/`. Current ownership/call chain is documented in `docs/sprints/info/SPRINT_ray_generation_engine_map.md`. The deleted `vrl/generation/capabilities.py` / `FamilyCapability` seam is **not** a keep target |
| **rewards** | uniform function/model split: `vrl/rewards/functions/*.py` pins a dotted model class/factory into the shared runtime, while `models/*.py` owns model-specific loading/parsing; `functions/registry.py` is the public dispatch boundary. Preserve this cross-family shape even when an individual facade is thin |
| **models** | trainable-weight namespace and version slots in `vrl/models/weight_utils.py`; PEFT lifecycle in `peft_adapter.py`; dependency loading/quantization in `loader.py`; cross-family denoise helpers in `vrl/models/steps/denoise/common/{lora,tensors,timestep,cfg}.py`. Deleted `models/utils.py` and old `models/diffusion/common/*` paths are **not** keep targets |
| **trajectory** | `vrl/trajectory/views.py:{role_tensor,named_tensor}` is shared and publicly re-exported; builders, resolver, validation, storage, and ops are lifecycle-organized single-concern modules. The thin lazy facade in `vrl/trajectory/__init__.py` prevents torch-free schema imports from loading tensor builders |
| **utils** | module-as-namespace single-concern families (`config.py`, `cuda_memory.py`, `media.py`, `profiling.py`, …) are stateless and shared by many callers; `vrl/utils/` is not the default destination for a helper with no owner |
| **nn** | symmetric split/concat family in `vrl/nn/layers/attention/cache_rows.py`; layer-organized kernels/layers/modules/quantization |
| **algorithms** | `vrl/algorithms/advantages.py:group_relative_advantages` as a GRPO-contract utility; `diffusion_dpo_loss` / `diffusion_sft_loss` as the symmetric offline loss family in `vrl/algorithms/dpo.py` |
| **scripts** | `vrl/scripts/train.py` CLI/import facade; `vrl/scripts/common/__init__.py` public recipe facade; `common/factory.py` construction boundary; state-owning `_RayClusterSession`, `_OnlineRecipeLifecycle`, and `OnlineRecipeRun` in `common/online.py`. Run resolution belongs to `vrl/run.py`; do not restore `scripts/common/resolved_run.py`, deleted `common/types.py`, or deleted inline `common/fixed_eval.py`. Stable data/eval/perf CLIs are long-term assets, while explicit probes/smokes/spikes are retired after their answer is recorded |

### Current hygiene verdict

**Should change:** only stale documentation paths and claims were corrected in this
revalidation. No production helper needs relocation merely because it is module-level.

**Should stay:** the Ray cleanup/deadline functions are framework/protocol adapters; model
weight and PEFT helpers own distinct key/lifecycle contracts; generation bindings preserve a
uniform cross-family shape; registry and script factories are dispatch/composition
boundaries. Their thinness is the point, not evidence that they should be merged.

**Resolved-run verdict:** the old `vrl/scripts/common/resolved_run.py` split should stay
deleted. Current `vrl/run.py` is useful because it is the single config-to-runtime
composition seam shared by online and offline entrypoints: `resolve_run` /
`resolve_online_run` construct `ResolvedRun` / `ResolvedOnlineRun`, and every field is read
by non-logging runtime control flow in `vrl/scripts/common/online.py` or
`vrl/scripts/families/wan_2_1/train_dpo.py`. Keeping resolution beside model
`resolve_model` / `materialize` avoids a script-only helper bag without turning the pure
resolution chain into a stateful manager.

**ALL_CAPS review:** retained constants in these families name real boundaries: environment
variables, checkpoint/source identities, registry import-path protocols, architecture
dimensions, or deliberately isolated config taxonomies. Do not move workflow vocabulary
into new ALL_CAPS tables, and derive validation key sets from the typed schema that owns
them.

**Non-goals:** do not resurrect deleted capability/helper modules to make this historical
table true; do not flatten family facades, protocol files, lazy imports, or uniform
cross-family adapters for LOC reduction; do not turn pure resolver pipelines into mutable
classes without actual shared state.

## Second pass — file-level organization lens (2026-06-30)

The first pass asked "is each free function legitimate?". A second pass asked the harder
question the gut actually reacts to: **"is this FILE organized optimally — even if each
function is defensible, would collecting them into a class / splitting the file be better?"**
16 reviewers, one per the biggest free-function files (12–61 free functions each), file-level
lens. **Result: all 16 = LEAVE, 0 reorganize.** The free-function *count* is not the smell —
the reviewers looked specifically for the two things that WOULD make a big flat file a problem
and found neither anywhere:

1. **Shared STATE threaded as repeated params → collect into a class.** Every "repeated
   parameter" turned out to be a read-only input computed once and passed *down* a call tree,
   a *pipeline-intermediate* value produced sequentially, or state that is *already owned* by
   an existing class. None was stable shared state (a client/handle/accumulator) a class
   should own. In `resources.py` specifically, wrapping the `_resolve_*` chain's threaded
   `trainer_devices/rollout_devices/...` in mutable `self.*` would **downgrade an auditable
   pure dependency chain into ordering-dependent state in a correctness-critical resolver** —
   a regression, not a cleanup.
2. **Mixed concerns → split into cohesive modules.** Every candidate "second concern"
   (anatomy vs safety, bridge vs targets, generate vs score, RTMW vs HaMeR, conversion vs IO)
   was two *flavors of one concern sharing a spine* (shared helpers/types/detection). Splitting
   would sever the spine or spawn a third `_common` module — churn without cohesion gain.

Several files already have their state correctly in a class (`trainer.py`, `wan_2_1/model.py`
own everything as fields; the remaining free functions are stateless helpers). Several are
one-shot procedural scripts where a flat function sequence is the right shape (`danbooru.py`,
`video_world.py`, the eval CLIs).

### The two closest calls — one done, one deliberately left (2026-06-30)
These were the only two files where a split was even plausible. On acting:

- **`vrl/scripts/common/online.py` → DONE at the time, later retired.** The ~280-line
  fixed-eval cluster
  (`_FixedEvalResult`/`_FixedEvalLocalStats` + `_iter_fixed_eval_shard`/`_fixed_eval_group_seed`/
  `_fixed_eval_collect_kwargs`/`_run_local_fixed_eval`/`_merge_fixed_eval_stats`/`_run_distributed_fixed_eval`)
  moved to a new `vrl/scripts/common/fixed_eval.py`. On inspection it was NOT actually
  "tightly wired" as the reviewer feared — every function is param-driven (takes
  `collector`/`reward_fn`/`training_context` as args) and calls no `online.py` function, so the
  extraction is a clean pure move with **no circular import**. `online.py` 1360→1079 lines;
  new module 312 lines; removed a now-unused `import gc`; 2 test imports repointed. Tests green
  (114). This was worth it at the time — a genuinely large file shed a self-contained
  concern. On 2026-07-11,
  `docs/sprints/done/SPRINT_remove_inline_fixed_eval.md` removed inline fixed evaluation
  entirely and deleted `vrl/scripts/common/fixed_eval.py`; standalone checkpoint evaluators
  under `vrl/scripts/eval/` are the surviving boundary. Do not recreate the deleted helper
  module from this historical paragraph.
- **`vrl/config/validation.py` → LEFT (deliberately).** The ~226-line Kling production-reward
  gate is a distinct concern, but the file is only 357 lines (not a large-file problem), and a
  split would need a second new module **plus a lazy back-import** to dodge a cycle (the new
  module needs `require`/`path_exists` from `validation.py`, whose `validate_training_config`
  calls the gate). Complexity-for-marginal-gain on a small file — the cure is worse than the
  itch. Left as-is.

### Verdict (both passes)
The "too many single separate functions" feeling is **volume, not disorganization**. Two
independent audits (function-legitimacy + file-organization) across all 12 packages and the 16
biggest files found **one real fix (done) and zero systemic problems**. The free functions are
cohesive modules, pure pipelines, procedural scripts, and shared utilities — and forcing them
into classes would *degrade* several (pure chains → mutable state). **Non-goal: a
"class-ify the free functions" pass — the evidence says it would make the code worse.**

## How to apply going forward (the decision rule for a new function)

```
new function → does it own/touch a class's state?         → make it a method
            → does a single-concern sibling module exist?  → co-locate there (symmetric family)
            → is it protocol/facade/adapter/registry/
              shared utility/active one-shot artifact?      → module-level is fine for its lifecycle
            → none of the above + floats alone / grab-bag?  → THAT is the smell to fix
one caller   → keep only for a real concept/lazy-import/protocol boundary; otherwise merge
never: flatten a registry/convention/cross-family abstraction merely to reduce LOC
```

## Scope / status (2026-06-30)

- **Fix 2 — DONE + verified.** On-class relocation of the 5 evaluator helpers; tests green
  (2 + 148). Pure relocation, no behavior change.
- **Fix 1 — SKIPPED on purpose, ownership later clarified.** Torch-free config resolution now
  lives in `vrl/config/precision.py`, PyTorch execution in `vrl/models/precision.py`, drift
  checking in `vrl/trainers/online/precision_guard.py`, and trainer-only diagnostics remain
  with the trainer. No new precision helper file is needed.
- **Documentation (this file):** the principle + non-goals catalog — the lasting asset.
- **Non-goal:** any broader "flatten free functions" pass — the audit shows there is no
  systemic problem (10/12 packages clean, and Fix 1 shows even a flagged cluster can be
  correct where it is), and a blanket pass would hit the protected families above.
