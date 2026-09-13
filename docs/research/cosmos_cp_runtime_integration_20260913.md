# Cosmos CP runtime integration candidate

Status: OPEN. Production-candidate primitives and self-attention processor are
implemented and tested, but model-level sharding and training integration are
not installed. This does not close the original context-parallel sprint or the
four-L40S hardware goal.

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
