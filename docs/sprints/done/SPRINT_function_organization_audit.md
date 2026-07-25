# SPRINT: Function organization audit

> **Conclusion first.** A conservative 12-package audit of `vrl/` for "disorganized
> standalone functions" (free module-level functions scattered away from their owner
> class or sibling family) found the repo **already well-organized: 10/12 packages
> clean, only 2 genuine fixable smells** (both `med`, both "functions split from an
> existing owner", both fixable with zero new abstractions). The worry that there are
> "too many single separate functions" does not hold up at the architecture level —
> the free functions are overwhelmingly legitimate (protocol/facade, lazy-import
> boundary, framework adapter, registry/dispatch, stateless shared utility, one-shot
> script, or a cohesive symmetric family). The durable value of this sprint is the
> **organizing principle** (so new code stays organized) and the **non-goals catalog**
> (so a future "cleanup" does not flatten the protected abstractions the owner has
> reverted before). Audit: 12 parallel package reviewers, 2026-06-29, findings
> verified against source (file:line cited).

## The organizing principle — what "organized" means here

A function lives in a coherent place when it is **one of**:

1. **A method** on the class that owns its state/behavior.
2. **A co-located sibling** in a cohesive single-concern module — a *symmetric family*
   (e.g. a split/concat pair, a `build_*` family, a set of `RolloutBatch` ops, a
   resolver's `_resolve_*`/`_validate_*` stages). The module itself is the namespace.
3. **A sanctioned module-level function**: a protocol/interface or public-API facade,
   a lazy-import boundary, a framework adapter (Ray/torch/pydantic/OmegaConf), a
   registry/dispatch entry, a stateless utility genuinely shared by **many** callers,
   or a one-shot procedural script (data prep / eval / smoke / probe).

The **smell** is only: a free function that floats apart from its owner class **or**
its sibling family; **or** a grab-bag file of unrelated free functions. Outside the two
fixes below, neither was found.

**Hard rules carried from AGENTS.md (the audit obeyed these, future work must too):**
never extract a helper that has one caller; never flatten a registry / convention /
cross-family abstraction; consistency beats LOC reduction.

## The 2 fixes (actionable, verified)

### Fix 1 — `[med]` precision-resolution cluster in trainer.py → **SKIPPED (not safe as proposed)**
The audit flagged the 9-function precision cluster at `vrl/trainers/online/trainer.py:204-309`
(`_resolve_mixed_precision`, `_get_autocast`, `_trainer_autocast_dtype`, `_needs_grad_scaler`,
`_precision_label`, `_dtype_label`, `_model_transformer_dtype`, `_trainer_precision_metadata`,
`_merge_rollout_precision_context`) and proposed moving it into the sibling
`vrl/trainers/precision.py`. **On inspection that move is wrong and was not done:**

- `vrl/trainers/precision.py` is **deliberately torch-free** (it injects `torch` as a
  parameter: `torch_dtype_for_mixed_precision(mixed_precision, *, torch)`) and is imported
  by **10 modules** (6 family `train.py` scripts, `scripts/common/online.py`,
  `offline/dpo.py`, `online/trainer.py`, a test). Its torch-free-ness is a load-bearing
  boundary.
- The 9 functions are **torch-heavy** (`torch.amp.autocast`, `torch.backends.cuda/cudnn`,
  `torch.dtype`, model-param dtype inspection). Moving them in would force `precision.py`
  to `import torch` (breaking the boundary for all 10 importers) or inject `torch` into 9
  signatures + every call site (churn for no gain).
- The other precision file, `online/precision_guard.py`, is a **different concern**
  (rollout-vs-replay logprob parity drift), not dtype/autocast resolution — so it is not a
  home either. A new `precision_runtime.py` would violate the no-new-lean-files rule and
  make a third precision module.
- The 9 functions are already a **co-located cohesive cluster** at the top of `trainer.py`
  (their torch-side owner file). The smell ("inside a 1607-line file") is mild; every cure
  degrades a real boundary. **Decision: leave them in trainer.py.** This is the audit's own
  hard rule in action — consistency/boundaries beat cosmetic relocation.

### Fix 2 — `[med]` one evaluator breaks the sibling on-class convention → **DONE**
**`vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:136-205`** keeps five
helpers (`_segments_from_batch`, `_trajectory_segment_payload`, `_extract_logprobs`,
`_segment_tensor`, `_primary_segment_name`) as **module-level free functions**, even
though they operate entirely on this evaluator's batch/segment data and have **zero
external callers** — and the same file already keeps `_enabled_segment_names` and
`_compute_segment_logprobs` as methods. Both sibling evaluators
(`token_logprob.py:85`, `continuous_token_logprob.py`) keep their per-evaluator helper
`_compute_logprobs` **as a `@staticmethod` on the class**. This one file alone splits
its helpers off, breaking the family's own convention.

- **Fix (done):** pulled the five helpers onto `MultiSegmentTokenLogProbEvaluator` as
  `@staticmethod`s, matching the sibling evaluators; call sites now `self._foo(...)` (the
  two staticmethod→staticmethod internal calls reference the class). Pure relocation, no
  behavior change. Verified the 5 had **zero external callers** (the 5 grep hits for
  `_primary_segment_name` are unrelated same-named *methods* on `trajectory.py` /
  `batch_builder.py` — which actually confirms on-class is the codebase-wide convention).
  Tests green: `test_multisegment_token_logprob.py` (2) + full `tests/rollouts/` (148).
- **Constraint kept:** NOT moved to a shared module — they are not shared; on-class is the
  established convention in `evaluators/ar/`.

## Non-goals — DO NOT "clean up" these (legitimately standalone, by package)

The audit explicitly cleared these; they satisfy the principle above and several are
abstractions the owner has reverted cleanups on. Listed so a future sweep does not
mistake them for scatter.

| package | legitimately-standalone families (keep as-is) |
|---|---|
| **config** | `build_*` family + its single-caller `_*` slicers (builders.py); `PrecisionPolicy` resolution family (precision.py, deliberately torch-free); OmegaConf overlay-loader stages (loading.py); `require`/`optional_none`/`path_exists` shared readers + the production-reward validator chain (validation.py); the `ConfigBlock` unknown-key walker (unknown_keys.py); pydantic boundary + walker glue (schema.py); `lint.py` CLI |
| **ray** | lazy-Ray + actor-metadata utilities (dependencies.py); symmetric teardown pair `kill_actors`/`remove_placement_group` (lifecycle.py); shared placement utilities (placement.py); `RayActorJob`+`run_actor_jobs` (actor_pool.py); the `resolve_distributed_resources` resolver pipeline + public derivation facades (resources.py) |
| **trainers** | distributed-collective primitives, unit-tested (`_global_reward_stats`, `_training_sample_chunks`, … trainer.py:46-435); precision.py / precision_guard.py / fsdp.py / weight_sync.py / data/* / checkpointing.py single-concern modules |
| **rollouts** | `families/registry.py` (registry/convention — reverted before); `batch/ops.py` symmetric `RolloutBatch` family; `batch/core.py` stack+`_stack_*`; the `orchestration/types.py::build_rollout_iteration` kwargs factory |
| **generation** | sample-chunk family (execution/chunks.py); `FamilyCapability` normalizers (capabilities.py); the 5-layer engine boundaries documented in `SPRINT_ray_generation_engine_map.md` |
| **rewards** | per-model uniform shape — one `XxxRewardModel` + factory `xxx_reward_model` (string-referenced registry/dispatch boundary from `functions/*.py`); repo-protected cross-family convention |
| **models** | shared-helper modules with `__all__` + de-dup rationale (utils.py, loader.py, diffusion/common/lora.py, tensors.py, timestep.py, cfg.py) |
| **trajectory** | `role_tensor` shared by 4+ modules + re-exported; lifecycle-organized modules |
| **utils** | module-as-namespace single-concern families (config.py, cuda_memory.py, media.py, profiling.py, …) — stateless, shared by many |
| **nn** | symmetric split/concat families (cache_rows.py); layer-organized kernels/layers/modules/quantization |
| **algorithms** | `group_relative_advantages` GRPO-contract utility; `diffusion_dpo_loss`/`diffusion_sft_loss` symmetric loss family (offline, intentionally bypasses online Algorithm) |
| **scripts** | hook-default framework adapters (common/types.py, online.py); data/eval/perf are one-shot procedural entrypoints |

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

- **`vrl/scripts/common/online.py` → DONE.** The ~280-line fixed-eval cluster
  (`_FixedEvalResult`/`_FixedEvalLocalStats` + `_iter_fixed_eval_shard`/`_fixed_eval_group_seed`/
  `_fixed_eval_collect_kwargs`/`_run_local_fixed_eval`/`_merge_fixed_eval_stats`/`_run_distributed_fixed_eval`)
  moved to a new `vrl/scripts/common/fixed_eval.py`. On inspection it was NOT actually
  "tightly wired" as the reviewer feared — every function is param-driven (takes
  `collector`/`reward_fn`/`training_context` as args) and calls no `online.py` function, so the
  extraction is a clean pure move with **no circular import**. `online.py` 1360→1079 lines;
  new module 312 lines; removed a now-unused `import gc`; 2 test imports repointed. Tests green
  (114). This was worth it — a genuinely large file shed a self-contained concern.
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
              shared-by-many-utility/one-shot-script?       → module-level is fine, leave it
            → none of the above + floats alone / grab-bag?  → THAT is the smell to fix
never:  extract a one-caller helper · flatten a registry/convention/cross-family abstraction
```

## Scope / status (2026-06-30)

- **Fix 2 — DONE + verified.** On-class relocation of the 5 evaluator helpers; tests green
  (2 + 148). Pure relocation, no behavior change.
- **Fix 1 — SKIPPED on purpose.** Moving the torch-heavy precision cluster into the
  torch-free, 10-importer `precision.py` would break a load-bearing boundary; no other home
  exists without a new file; the cluster is already co-located in its torch-side owner file.
  The cure is worse than the mild smell — left as-is. (A good example of the audit's own
  hard rule: boundaries/consistency beat cosmetic relocation.)
- **Documentation (this file):** the principle + non-goals catalog — the lasting asset.
- **Non-goal:** any broader "flatten free functions" pass — the audit shows there is no
  systemic problem (10/12 packages clean, and Fix 1 shows even a flagged cluster can be
  correct where it is), and a blanket pass would hit the protected families above.
