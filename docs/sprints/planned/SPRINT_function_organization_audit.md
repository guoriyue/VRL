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

### Fix 1 — `[med]` precision-resolution family split across two files
**`vrl/trainers/online/trainer.py:204-309`** holds a 9-function precision family
(`_resolve_mixed_precision`, `_get_autocast`, `_trainer_autocast_dtype`,
`_needs_grad_scaler`, `_precision_label`, `_dtype_label`, `_model_transformer_dtype`,
`_trainer_precision_metadata`, `_merge_rollout_precision_context`) inline in a 1607-line
trainer — while a **dedicated sibling module already owns this concern**:
`vrl/trainers/precision.py` (`normalize_mixed_precision`,
`torch_dtype_for_mixed_precision`, `torch_dtype_for_trainer_precision`). The inline
cluster is pure `(config, device, model)` — it touches **no** `OnlineTrainer` internals,
and the only precision symbol it imports is `normalize_mixed_precision`, which lives in
`precision.py`.

- **Fix:** move the 9 functions into `vrl/trainers/precision.py` as module-level
  functions; import them back into `trainer.py`. Restores one-module-per-concern.
- **Constraint:** keep them **free functions, NOT `OnlineTrainer` methods** —
  `_needs_grad_scaler` is unit-tested standalone (`tests/trainers/online/test_grad_scaler.py`)
  and `_dtype_label` is reused conceptually by the rollout side. This is a *co-locate
  with the sibling family* move, not an *extract-into-class* move.

### Fix 2 — `[med]` one evaluator breaks the sibling on-class convention
**`vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:136-205`** keeps five
helpers (`_segments_from_batch`, `_trajectory_segment_payload`, `_extract_logprobs`,
`_segment_tensor`, `_primary_segment_name`) as **module-level free functions**, even
though they operate entirely on this evaluator's batch/segment data and have **zero
external callers** — and the same file already keeps `_enabled_segment_names` and
`_compute_segment_logprobs` as methods. Both sibling evaluators
(`token_logprob.py:85`, `continuous_token_logprob.py`) keep their per-evaluator helper
`_compute_logprobs` **as a `@staticmethod` on the class**. This one file alone splits
its helpers off, breaking the family's own convention.

- **Fix:** pull the five helpers onto `MultiSegmentTokenLogProbEvaluator` as
  `@staticmethod`s, matching the sibling evaluators. Restores cross-evaluator symmetry.
- **Constraint:** do **NOT** move them to a shared module — they are not shared;
  on-class is the established convention in `evaluators/ar/`.

## Non-goals — DO NOT "clean up" these (legitimately standalone, by package)

The audit explicitly cleared these; they satisfy the principle above and several are
abstractions the owner has reverted cleanups on. Listed so a future sweep does not
mistake them for scatter.

| package | legitimately-standalone families (keep as-is) |
|---|---|
| **config** | `build_*` family + its single-caller `_*` slicers (builders.py); `PrecisionPolicy` resolution family (precision.py, deliberately torch-free); OmegaConf overlay-loader stages (loading.py); `require`/`optional_none`/`path_exists` shared readers + the production-reward validator chain (validation.py); the `ConfigBlock` unknown-key walker (unknown_keys.py); pydantic boundary + walker glue (schema.py); `lint.py` CLI |
| **ray** | lazy-Ray + actor-metadata utilities (dependencies.py); symmetric teardown pair `kill_actors`/`remove_placement_group` (lifecycle.py); shared placement utilities (placement.py); `RayActorJob`+`run_actor_jobs` (actor_pool.py); the `resolve_distributed_resources` resolver pipeline + public derivation facades (resources.py) |
| **trainers** | distributed-collective primitives, unit-tested (`_global_reward_stats`, `_training_sample_chunks`, … trainer.py:46-435); precision.py / precision_guard.py / fsdp.py / weight_sync.py / data/* / checkpointing.py single-concern modules |
| **rollouts** | `families/registry.py` (registry/convention — reverted before); `batch/ops.py` symmetric `RolloutBatch` family; `batch/core.py` stack+`_stack_*`; `orchestration/types.py` factory |
| **generation** | sample-chunk family (execution/chunks.py); `FamilyCapability` normalizers (capabilities.py); the 5-layer engine boundaries documented in `SPRINT_ray_generation_engine_map.md` |
| **rewards** | per-model uniform shape — one `XxxRewardModel` + factory `xxx_reward_model` (string-referenced registry/dispatch boundary from `functions/*.py`); repo-protected cross-family convention |
| **models** | shared-helper modules with `__all__` + de-dup rationale (utils.py, loader.py, diffusion/common/lora.py, tensors.py, timestep.py, cfg.py) |
| **trajectory** | `role_tensor` shared by 4+ modules + re-exported; lifecycle-organized modules |
| **utils** | module-as-namespace single-concern families (config.py, cuda_memory.py, media.py, profiling.py, …) — stateless, shared by many |
| **nn** | symmetric split/concat families (cache_rows.py); layer-organized kernels/layers/modules/quantization |
| **algorithms** | `group_relative_advantages` GRPO-contract utility; `diffusion_dpo_loss`/`diffusion_sft_loss` symmetric loss family (offline, intentionally bypasses online Algorithm) |
| **scripts** | hook-default framework adapters (common/types.py, online.py); data/eval/perf are one-shot procedural entrypoints |

## How to apply going forward (the decision rule for a new function)

```
new function → does it own/touch a class's state?         → make it a method
            → does a single-concern sibling module exist?  → co-locate there (symmetric family)
            → is it protocol/facade/adapter/registry/
              shared-by-many-utility/one-shot-script?       → module-level is fine, leave it
            → none of the above + floats alone / grab-bag?  → THAT is the smell to fix
never:  extract a one-caller helper · flatten a registry/convention/cross-family abstraction
```

## Scope / status

- **Documentation (this file):** the principle + non-goals catalog — the lasting asset.
- **Fixes:** 2 moves, both pure relocation (no behavior change), both keep their tests;
  Fix 1 = co-locate with sibling module, Fix 2 = on-class to match siblings. Land them
  when convenient; neither is urgent (both are organization, not correctness).
- **Non-goal:** any broader "flatten free functions" pass — the audit shows there is no
  systemic problem, and a blanket pass would hit the protected families above.
