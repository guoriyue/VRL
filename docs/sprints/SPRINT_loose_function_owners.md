# Loose functions: give each piece of logic an owner

Status: pass 1 and pass 2 landed 2026-09-14. Scope is `vrl/` outside `vrl/scripts/**`
(scripts are entry points, not helpers).

## Rule applied

A free function is loose when nothing owns it: one importer, or a helper
module that exists only because two places once shared a few lines. The
fix is an owner, in this order: (1) one consumer → inline or a private
method of that class; (2) two or more classes with a shared base → a method
or hook on the base, and if they share only the helper but also the same
class-level contract, they share a base; (3) cross-package and domain-free →
`vrl/utils/<topic>.py`; (4) otherwise it stays beside its users in a module
named for a domain noun. Never a new module for fewer than three consumers,
never a compatibility alias, never a rename counted as a fix.

## Survey

Class-free modules under `vrl/` with ≤ 2 importers (15), and public
module-level functions referenced from exactly one other module (~120).
Everything with one importer was read at the source with its call sites
and tests; two importers needed a reason to stay.

## Moved to an owner

| was | now | commit |
|---|---|---|
| `algorithms/previous_adapter.py`: `flow_time`, `flipped_advantage_losses`, `sync_previous_policy_adapter` (3 free functions used by NFT and V-GRPO) | `PreviousAdapterObjective` base class: the replay-branch contract both declared identically, `flow_time`, the seeded invariant check (subclasses supply the residual), `after_optimizer_step` with a post-sync hook | `49015be3b` |
| `algorithms/diffusion_nft.normalized_mse` (imported by V-GRPO) | `PreviousAdapterObjective.normalized_mse` | this pass |
| `ray/dependencies.raise_if_kill_failures` + `note_kill_failures` | one `kill_failures_error` returning the error; owners raise it or `add_note` its text | `49015be3b` |
| `generation/execution/types.combine_rank_batch_results` | `generation/ray/engine.py`, which owns the rank fan-out (the executor already imports the engine); `types.py` is data again | `49015be3b` |
| `rollouts/batch/ops.py`: `select_batch`, `split_batch_by_group`, `remap_group_ids_`, `move_training_batch_to_device` | methods on `RolloutBatch` (`select`, `split_by_group`, `remap_group_ids_`, `to_device`), mirroring `TrajectoryBatch`; lazy imports keep the leaf torch-free | `ad89eceee` |
| `rollouts/batch/ops.nonzero_advantage_mask` | private function in the trainer, its only user | `ad89eceee` |
| `models/steps/denoise/common/vae_decode_memory.py` (2 functions, one caller) | `VaeDecodeMemoryPass.apply` owns the policy; tests drive the pass | `827f58cbd` |
| `generation/execution/pipeline.forward_batches_pipelined(executor, ...)` | `BatchExecutorBase.forward_batches_pipelined` — it took the executor as its first argument and called its `forward_batch`; the D2H copy helper moved with it | `a7174c182` |
| `generation/execution/batch_memory.cuda_occupancy_snapshot` | `BatchMemoryReading.cuda_occupancy_snapshot` — it produced half of that record | `4baad287e` |
| `utils/validation.require_exact_dataclass_fields` (dataclass-only) | `require_mapping_keys(value, allowed, *, what, complete)`; the dataclass form is a one-line wrapper; three more sites adopted it | `05ccce80c` |
| `models/loader.apply_rollout_quantization` + `validate_rollout_quantization_support` (one runtime caller: `QuantizationPass.apply`; five test files) | `QuantizationPass.quantize` / `QuantizationPass.validate_support`; `loader.py` is Diffusers loading only; the "no scheme → 0" early return is gone because `enabled` is the gate | pass 2 |
| `trainers/online/ema.EMA._snapshot` (verbatim copy of `fsdp._full_cpu_tensor`, docstring said "mirrors") | `fsdp.full_cpu_tensor`, called by the EMA | pass 2 |
| `fsdp.load_checkpoint_state_dict`'s DCP scatter (same call as `load_full_state_dict`) | calls `load_full_state_dict(module, compatible, strict=False)` after its owned-key checks | pass 2 |

## Stayed, with the reason

- **`trainers/fsdp.py`** (16 functions, one importer: `FSDPStrategy`). The
  strategy is the trainer-facing adapter; this module is the FSDP2
  collective layer, imported lazily so `strategy.py` stays free of
  `torch.distributed` at import, and four test files exercise the gather /
  load functions on tiny modules without a strategy. Both docstrings and
  `schema.py` name the split. Rule 4. Pass 2 removed the two duplicates
  inside it (see below); the module itself stays.
- **`trainers/diagnostics.py`** (4 functions, one importer: `OnlineTrainer`).
  The trainer's debug instrumentation; folding 200 lines into a 2,300-line
  class would cost more than the module. Rule 4.
- **`config/rules.py`** (one importer, `schema.py`). 128 lines of
  cross-section validation rules; `schema.py` is already 800 lines.
- **`math/denoise/ddim.py`**, **`math/token/flow_matching.py`**: the math
  layer's step kinds, named for the noun; the latter is NextStep-1's only
  because it is the only flow-head token family so far.
- **`rewards/assets/{hpsv3_prompts,kling_prompt_templates}.py`**: the
  `rewards/assets` convention (prompt templates beside asset files) is
  followed by `video_judge_prompts.py` too.
- **`rewards/service/wire.py`**: every codec pair has one producer (server)
  and one consumer (client); the module is the wire contract.
- **`models/families/*/runtime.py` builders, `*_config_from_build`**:
  referenced by path string from the registry; the convention every family
  follows.
- **`nn/quantization/fp4_kernels.py`**: Triton kernels kept apart from the
  module class.
- **`config/reward_inference.require_http_origin`**,
  **`runtime_errors.root_failure_cause`**, **`models/families/names.py`**:
  each sits with the type or contract it validates.
- **`config/lint.py`**: script support with a `main`; an entry point.

## Not a loose function, done alongside

`precision.diffusion_math` → `precision.denoise_math` and
`DiffusionMathPrecisionConfig` → `DenoiseMathPrecisionConfig`: the YAML key
that had kept the class out of the denoise rename. No preset set it;
`docs/reference/generation_naming.md` records the break.

## Verification

Per change: ruff on touched files and the touched packages' tests
(`tests/algorithms`, `tests/rollouts`, `tests/trainers/online`,
`tests/generation` incl. the real-CUDA pipelined lane on the idle 5090,
`tests/nn`, `tests/models/steps`). `test_lifecycle_fsm::test_shutdown_kills_only_owned_actor`
failed once inside a full `tests/generation` run and passed 3/3 in
isolation; it is an actor-shutdown timing test not touched here.

Pass 2: `tests/nn`, `tests/models/steps`, `tests/models/families/wan_2_1`,
`tests/architecture` (minus the pre-existing public-floor failure),
`tests/trainers` (646 passed), `tests/config` + the precision/factory/anima
script tests (476 passed).
