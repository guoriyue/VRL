# Cosmos CP runtime integration candidate

Status: OPEN. Production-candidate primitives, self-attention processor and
opt-in model-level sharding are implemented. Fixed-row Linear compute resolves
the retained small-model CUDA regression below. Training integration and
released-weight validation of this candidate remain absent. This does not close
the original context-parallel sprint or the four-L40S hardware goal.

## Isolation and commits

Candidate worktree: `/home/ubuntu/VRL-cosmos-cp`, branch
`feat/cosmos-cp-runtime`, based on frozen runtime `382d0825`.

- `b842821e`: differentiable token/head exchanges in the existing
  `vrl/trainers/distributed.py` module.
- `7d7d371d`: Cosmos self-attention processor in the existing shared family
  module, with actual Diffusers attention forward/backward tests.

The running integration worktree and shared dependencies were not changed.
The processor is opt-in; no existing model loader installs it automatically.

## Implemented contracts

The exchange primitives use functional autograd all-gather, not the deprecated
diagnostic collective API. They transform `[B,H,S/P,D]` to `[B,H/P,S,D]` and
back, with explicit process group, equal nonempty shards and divisibility
checks. This is a gather-based baseline with temporary full-QKV allocation,
not an optimized all-to-all implementation.

`CosmosContextParallelSelfAttnProcessor` preserves the actual Cosmos QKV
projections, QK normalization, local-position RoPE and output projection.
Attention runs over full tokens and local heads through the existing
Diffusers dispatcher. Masked and cross-attention inputs fail explicitly.
The caller must provide correct local RoPE and matching collective order.

Activation gradients are handled by autograd collectives. Replicated
parameter gradients still require CP SUM in the strategy. The processor does
not implement model token splitting, precision policy or gradient reduction.

## Verification

Primitive milestone:

- CPU primitive/distributed-training/strategy regression: 31 passed, 1 skipped.
- Two-L40S NCCL primitive tests: 1 passed, 3 deselected.
- CPU coverage includes noncontiguous subgroups `[0,2]` and `[1,3]`, exact
  roundtrip gradients and SDPA Q/K/V gradients.

Processor milestone:

- Two-L40S NCCL: 1 passed, 3 deselected, 10.63 seconds.
- Actual Cosmos block self-attention, FP32, two ranks; both with/without RoPE
  and with/without nonreentrant checkpoint recomputation.
- Compared outputs, local input gradients and CP-summed parameter gradients
  against the ordinary unsharded Diffusers processor.
- Output/input tolerances: atol 2e-6, rtol 2e-5; parameter gradients:
  atol 1e-5, rtol 2e-5. These are unit tolerances, not replay acceptance gates.
- Expanded CPU Cosmos-family plus exchange suite: 31 passed, 2 skipped,
  5 failed. All five failures are Cosmos3 fixtures unable to import
  `Cosmos3OmniPipeline` from shared Diffusers 0.38. The same five tests fail
  identically in untouched `/home/ubuntu/VRL-mgpu-integration` (1.71 seconds).
  Do not describe the complete family suite as green.
- Ruff check and diff whitespace check passed. GPU test exited 0; fresh
  compute-process inventory empty after completion.

Reproduce processor GPU test from the candidate worktree:

```bash
env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/ubuntu/VRL-cosmos-cp \
  /home/ubuntu/VRL/.venv/bin/python -m pytest \
  tests/models/families/cosmos/test_context_parallel_attention.py \
  -k nccl --distributed -q
```

## Remaining required integration

1. Model-level token/RoPE slicing, output gathering and cross-attention
   conditioning contract through real Cosmos forward and checkpoint paths.
2. The validated padded-64-row Linear and FP32 LoRA compute contract, applied
   consistently to rollout and replay, plus strict deterministic configuration.
3. CP strategy/process mesh and correct parameter-gradient reduction without
   accidental extra loss division or interaction with DP averaging.
4. Full-shape real-family replay parity and actual online GRPO update,
   checkpoint/resume, EMA and memory evidence under the original sprint gates.
5. End-to-end timing and compile acceptance only after correctness gates.

Prior full-shape numerical and DiT-only timing evidence remains in
`cosmos_cp_480p_capacity_20260912.md`; it used diagnostic installation and
does not automatically validate this new processor or a complete trainer.

## Model-level candidate: 558a3342

`vrl/models/families/cosmos/context_parallel.py` installs explicit, reversible
hooks. Tokens remain sharded throughout the block stack; the final output
projection is gathered once for the native unpatchify. Hooks bind the actual
forward signatures, including native checkpoint positional arguments. Each
block receives local RoPE, per-token timestep/conditioning and optional
positional/residual tensors. Text conditioning remains replicated and uses
the ordinary cross-attention. ControlNet projection blocks are rejected.

The context must remain installed through backward. It does not create groups
or automatically divide loss/reduce gradients. A replicated full-output loss
must be divided by CP size before backward; parameter gradients are then
SUM-reduced across CP. Tests also SUM the replicated input/text gradients to
compare them with the reference. This is not yet wired to an online strategy.

### Tests and a retained failure

Two-layer actual Cosmos models use three frames so a two-rank token boundary
crosses a frame. Four configurations cross learnable-position enabled/disabled
with text-projection enabled/disabled. Each exercises ordinary and native
checkpoint forward/backward, global and per-frame timesteps, full output,
input/text gradients, every used parameter gradient, duplicate-install
rejection and restoration to ordinary unsharded forward.

- CPU model/attention/exchange suite: **10 passed, 6 skipped**, 39.21 seconds.
- Two-L40S model NCCL matrix: **3 passed, 1 failed, 4 deselected**, 28.98 seconds.
- The target pinned Predict2.5 configuration has `extra_pos_embed_type=null`
  and `use_crossattn_projection=true`; the corresponding tiny FP32 model case
  passes. This is not released-weight/BF16/replay evidence.
- Failing case: no text projection, zero-initialized learnable position,
  checkpoint enabled, per-frame timestep. Parameter
  `learnable_pos_embed.pos_emb_h` has one mismatching element of 256:
  absolute difference 0.625, relative difference 6.545102223753929e-5,
  at index `(0,10)`, versus atol 2e-5 / rtol 5e-5. The same case failed before
  the configuration matrix was expanded. At this milestone it remained an
  ordinary failing test, not skipped, xfailed or given a relaxed tolerance.
  The subsequent fixed-row result is recorded below.
- Adding text projection changes initialization and passes even with learned
  positions; that does **not** explain or resolve the original failure.
- No strict deterministic/padded-Linear production precision policy has been
  installed yet. Do not attribute the mismatch solely to CP without controls.
- Ruff and whitespace checks pass. All test sessions exited; fresh GPU process
  inventory empty. Frozen integration runtime and dependencies unchanged.

Next: integrate the previously validated deterministic/Linear/LoRA compute
contract, investigate the retained positional-gradient failure, then validate
actual family replay and strategy gradient semantics. Original full online
update/resume/EMA/capacity gates remain open.

## Fixed-row compute: 5c89cf7f

The existing `vrl/models/precision.py` now provides an opt-in
`fixed_row_linear_compute` context. All Linear calls flatten their leading
dimensions, execute fixed 64-row GEMMs, pad the tail, discard padding inside
the Linear and restore the original output shape. The context also accepts
explicit FP32 Linear modules for LoRA A/B computation with autocast disabled;
it rejects non-FP32 parameters rather than silently converting state.

Parameter identities, state keys and preexisting custom forwards survive the
context. Empty tensors, noncontiguous input, tail sizes, duplicate installation,
exceptions and restoration are tested. Installation must follow model copying/
loading, remain active through backward, and apply equally to rollout and
replay. Copying an actively wrapped model and concurrent installation are
unsupported. No runtime loader or trainer automatically enables this yet.

### Same-failure control and full matrix

The original no-text-projection/learnable-position case was rerun with only
fixed-row Linear compute enabled on both reference and CP models: **1 passed,
8 deselected**, 11.02 seconds. No tolerance change or separate strict
determinism change was used in this control. This shows that the fixed-row
policy is sufficient for this particular retained FP32 regression; it does
not isolate every kernel or prove BF16/full-size equivalence.

Then the entire model matrix used the same policy:

- Two-L40S NCCL: **5 passed, 4 deselected**, 35.48 seconds. Four model
  configurations plus the named original regression test. The original
  parameter-gradient atol 2e-5 / rtol 5e-5 remain unchanged.
- CPU precision/helper/model matrix: **24 passed, 5 skipped**, 23.10 seconds.
- A real PEFT adapter on BF16 base weights, with nonzero B and FP32 A/B,
  passes CPU BF16-autocast and nonreentrant-checkpoint checks: FP32 branch
  outputs, finite nonzero A/B gradients, frozen BF16 base unchanged.
  This is a helper-level adapter check, not Cosmos BF16 replay acceptance.
- Ruff and whitespace checks pass. All owned jobs exited, fresh GPU inventory
  empty and GPUs released. Frozen runtime and shared dependencies unchanged.

The original failing configuration remains covered with the corrected compute
policy, not skipped or xfailed. The prior failure above remains historical
evidence of why ordinary untiled compute is insufficient for that guard.

Next required: real-family BF16/FP32-LoRA validation with this model-level
candidate, strict deterministic runtime configuration, CP strategy/loss/gradient
integration, and the original full-shape online update/resume/EMA gates.
