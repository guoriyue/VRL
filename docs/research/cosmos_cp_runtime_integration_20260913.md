# Cosmos CP runtime integration candidate

Status: OPEN. Production-candidate primitives, self-attention processor and
opt-in model-level sharding are implemented. Fixed-row Linear compute resolves
the retained small-model CUDA regression below. Released-weight fixed-action
CPS validation with head-sharded cross-attention now has exact output/logprob
and gradient relative L2 2.2244e-6 (see below). An explicit strategy now reaches
the trainer optimizer boundary; online recipe/data integration remains absent.
This does not close
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

## Released-weight full-shape CPS candidate validation

Runtime remains candidate `5c89cf7f`, frozen during execution. The existing
NVMe diagnostic now accepts `--runtime-cp`: it calls the candidate model-level
context and fixed-row precision helper, bypassing the old per-block diagnostic
installation and forward monkeypatches. No shared dependency changed.

Evidence root:
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_candidate_fullshape_cps_l40s`
with `rank-0.json`, `rank-1.json`, `probe_source.py` and adjacent `.log`.
Probe snapshot SHA256:
`343d704ac8e3a1e2c4d7969494eda0a492f75311903e38ac29c02bfd1691390e`.

Pinned released Predict2.5 2B transformer/scheduler revision
`0d37c7498f54cee3c599d438d895a0a4a8608064`. Latent `[1,16,9,60,104]`,
14,040 tokens, corresponding to 480x832 / 33 frames; synthetic text
`[1,512,100352]`. Actual family `forward_step`, CFG5, UniPC20 index18,
sigma 0.0281333942. Fixed action from the unsharded reference, actual CPS
log-prob, BF16 base, native default/frozen-previous adapters, nonzero default
B std .001 seed73, FP32 LoRA, padded64 Linear, native GPU checkpoint.
Strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, efficient
SDPA context spanning both forward and backward. CP replicated loss divided
by2; parameter gradients SUM-reduced across the two ranks.

Both ranks report identical results:

- Output max absolute error: **0**.
- CPS log-prob max absolute error and ratio deviation: **0**.
- Global gradient relative L2: **0.00016143085017429523**.
- All **560** trainable gradient tensors finite and nonzero: 280 A + 280 B.
- Trainable dtype FP32; all 560 previous-adapter tensors frozen.
- Largest per-tensor gradient relative L2: **0.0200619791** at
  `transformer_blocks.26.attn2.to_k.lora_B.default.weight`; max absolute error
  **3.0536240203e-11**, error L2 **2.2238370867e-10**.
- The corresponding A relative L2 is .0200533625. Other largest relative
  errors also involve later cross-attention K projections.

The whole-gradient error is higher than the older strict-deterministic
per-block-gather prototype's ~2.2244e-6. This is not exact gradient equivalence,
and the global norm must not hide the ~2% small-tensor relative errors. The
diagnostic's `finite_network_diagnostic_complete` status asserts finiteness
and scalar log-prob <=1e-3, not a gradient acceptance threshold. No original
gate was relaxed. Investigate the local-query cross-attention K-gradient
accumulation path with a controlled head-sharded cross-attention comparison;
its causal role is not yet proven.

Launch used `--runtime-cp --method ulysses --family-forward
--production-adapters --guidance-scale 5 --timestep-index 18
--fp32-lora-compute --lora-b-std 0.001 --linear-token-tile 64
--pad-linear-tiles --latent-height 60 --latent-width 104 --latent-frames 9
--text-tokens 512 --checkpoint --deterministic --dtype bf16` with the pinned
model path above and candidate PYTHONPATH after the TF5.13 overlay.

This is one synthetic-condition, fixed-action full-weight test. It does not
exercise real text encoding, VAE, reward, a rollout trajectory, advantages,
optimizer update/resume, EMA, compile or end-to-end performance. It also is
not a peak-memory benchmark. Both processes exit0; fresh GPU compute inventory
empty and claim released.

### Online integration remains broader than a strategy switch

Current `vrl/scripts/common/online.py` gives `PromptBatchSampler` physical
world size/rank, and initializes rank-distinct RNG streams. CP requires
identical replay tensors, timesteps, conditions and stochastic decisions
within a CP group. Prompt sampling, rollout ownership/broadcast, filtering,
advantage statistics and denominators must distinguish DP from CP. Existing
DDP averages parameter gradients whereas this probe SUMs after dividing the
replicated loss; integrating both without accounting for that distinction
would scale gradients incorrectly. No configuration-only CP enablement is
claimed or installed.

## Head-sharded cross-attention: 60676789

The processor is now named `CosmosContextParallelAttnProcessor`. Optional
`cross_attention=True` exchanges local query tokens for local heads, while
replicated text K/V are sliced by head. The resulting full-token attention
output is exchanged back to local query tokens. Key-only broadcast masks are
supported; query-dependent masks, RoPE on cross-attention and image context
are rejected. Replicated text gradients and parameter gradients still need
CP SUM at the caller boundary.

The model context accepts `shard_cross_attention=True`; its default remains
False for the prior local-query control. The validated full-shape candidate
must explicitly enable this option. Processor restoration includes both
attention modules. This changes no existing runtime loader automatically.

### Unit validation

- CPU attention/model suite: 8 passed, 7 skipped, 71.08 seconds.
- Two-L40S NCCL suite: 7 passed, 8 deselected, 46.95 seconds.
- Actual Diffusers self/cross-attention output, local query gradients,
  CP-summed text gradients and all used parameter gradients; key-mask and
  checkpoint variants. Multi-block tests exercise cross-head sharding with
  the fixed-row compute contract and existing tolerances.
- Ruff and whitespace checks pass. Commit made before the full-weight run;
  runtime stayed unchanged while the job ran.

### Same-input full-shape result

Evidence root:
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_candidate_crossheads_fullshape_cps_l40s`
with both rank JSON reports, frozen `probe_source.py` and adjacent `.log`.
Snapshot SHA256:
`5cd541339ec691a7a341049d0b59dd5e770a0c51b7db9190488c127ee7c7418b`.

Same released revision, full latent/text geometry, BF16 base, native adapters,
nonzero B seed, FP32 LoRA, fixed64 Linear compute, deterministic environment,
checkpoint, CFG5/index18 and fixed-action CPS objective as the preceding run.
The additional `--runtime-cross-attention` flag enables the new path. No
gradient tolerance was loosened and no weights/inputs were changed.

Both ranks report:

| Measure | Local-query cross-attention | Head-sharded cross-attention |
| --- | ---: | ---: |
| Output max absolute error | 0 | 0 |
| CPS log-prob max absolute error | 0 | 0 |
| Global gradient relative L2 | 1.6143085017e-4 | 2.2243869835e-6 |
| Worst tensor gradient relative L2 | .0200619791 | 4.9922809922e-6 |
| Finite/nonzero LoRA gradient tensors | 560 | 560 |

The new worst tensor is block26 cross-attention Q LoRA A, max absolute error
2.3201926491e-16. The previous ~2% cross-K outliers are no longer present.
The new global gradient result matches the earlier deterministic per-block
diagnostic result numerically, but now uses model-level token sharding across
the block stack and the candidate runtime helpers.

This controlled result supports using head-sharded cross-attention instead
of local-query cross-attention for the validated compute contract. It is not
bitwise gradient equality, a full rollout/replay trajectory, an optimizer or
online update, nor a throughput/capacity benchmark. The original broader
training, EMA, checkpoint/resume and compile gates remain open.

Both processes exited0, fresh GPU compute inventory empty, GPUs0-1 released.
Frozen integration runtime and shared dependencies remain unchanged. Next
required work is strategy/data ownership integration and actual trainer
update/resume with the validated attention/precision contract.

## DP x CP accumulation primitives: e3f7f54b

The existing distributed module now provides `ContextParallelGroups`,
`create_context_parallel_groups` and `reduce_context_parallel_gradients`.
For four ranks with CP2, CP groups are `[0,1]` and `[2,3]`; DP groups are
`[0,2]` and `[1,3]`. All ranks create groups in the same order. The default
process group must already exist; its owner remains responsible for shutdown.
DP rank/size, not physical rank/size, define independent sampler identity.

The reducer is called **once after accumulation and before clipping/step**:
SUM across CP, then SUM/divide across DP. It is not combined with a DDP
reducer. A replicated full-output loss must be divided by CP size before
backward and by the accumulation count as appropriate. Globally unused
parameters retain `grad=None`, preserving AdamW skip/weight-decay semantics.
Rank-local missing gradients are replaced with zeros only when another world
rank used the parameter. Sparse gradients cause a collective-consistent error.
The caller must supply identical ordered parameter lists on one device; no
FSDP/DTensor integration is claimed.

### Four-L40S update comparison

`tests/trainers/test_context_parallel_updates.py` runs DP2 x CP2 with an actual
tiny Cosmos self-attention module and fixed-row compute. CP peers share
inputs; the two DP groups use distinct inputs. Two AdamW updates each have
two accumulated microbatches. A full-batch unsharded reference computes all
four contributions per update. Checks cover gradients, updated parameters,
AdamW state, a globally unused parameter, and a parameter used only by DP1.
Invalid CP sizes and rank0-only sparse gradients also exercise failure paths.

- Four-L40S NCCL final test: **1 passed, 1 deselected**, 9.97 seconds.
- CPU update/exchange/distributed regression: **11 passed, 2 skipped**,
  20.31 seconds.
- Gradients use atol1e-5/rtol3e-5; updated parameters atol1e-6/rtol1e-5;
  optimizer state atol2e-6/rtol5e-5. This is tolerance agreement, not bitwise
  equivalence. Existing full-weight CPS metrics remain separate evidence.
- Ruff and whitespace checks passed. All processes terminal; fresh GPU
  compute inventory empty, GPUs0-3 released. Frozen integration runtime and
  shared dependencies unchanged.

This is a FP32 attention/reduction/optimizer unit, not a full Cosmos model
online update, checkpoint/resume, reward or advantage test. The helpers are
not yet invoked by a selectable training strategy or online data path. Next
integration must connect group-aware replay ownership, precision contexts,
accumulation boundaries and checkpoint handling before opening a config gate.

## Explicit CP strategy: 60829dc1

`ContextParallelStrategy` is now implemented in the existing strategy module,
but intentionally absent from schema/config dispatch. It can be supplied
explicitly to the trainer interface only by an orchestrator that already owns
group-consistent replay inputs and DP-aware sampling.

Preparation validates one Cosmos transformer with FP32 trainable parameters,
initializes/validates the process group, creates DP x CP groups, broadcasts
initial parameters and buffers from rank0, and installs fixed-row Linear,
FP32 LoRA branches and self/cross-head sharding. CUDA requires strict
deterministic IEEE compute and a supported cuBLAS workspace setting; efficient
SDPA remains selected through backward/checkpoint recomputation. Rollout must
separately use the same precision contract without replay CP hooks.

`backward` divides replicated full-output loss by CP size. At the existing
trainer `clip_grad_norm` boundary, accumulated gradients are CP-SUM/DP-mean
reduced once before clipping. Even when clipping is disabled, the trainer
calls this boundary with infinity. FP16 GradScaler and shared-GPU state
parking fail explicitly. FP32-master parameter substitution is not supported:
the clip list must match the prepared trainable parameters. Shutdown removes
execution hooks/contexts and closes process groups.

Checkpoint/optimizer state uses the existing unsharded strategy implementation;
rollout export uses the existing trainable-state snapshot helper. The strategy
is included in all structural-interface and shared-method ownership tests.

### Actual trainer boundary, not a complete online loop

New `test_context_parallel_strategy.py` builds actual tiny Cosmos transformers
on four ranks (DP2 x CP2), deliberately perturbs initial candidate weights
by rank and verifies broadcast restores equality. Two AdamW updates each
accumulate two microbatches. It invokes the real
`OnlineTrainer._clip_and_step` on a minimal state carrier, comparing clip
norms, updated parameters and every optimizer-state tensor with a full-batch
unsharded reference. Export/load optimizer state is checked by exact tensor
tree equality. Duplicate preparation, unsupported scaler/parking and hook
cleanup are covered.

- Four-L40S NCCL: **1 passed, 1 deselected**, 10.72 seconds.
- Initial combined CPU strategy/DDP/MRO regression: **34 passed, 1 skipped**,
  9.02 seconds.
- Expanded MRO/interface matrix including CP: **28 passed**, .23 seconds.
- Ruff and whitespace checks passed. All jobs terminal, fresh GPU process
  inventory empty, GPUs0-3 released. Runtime frozen during GPU tests; shared
  integration runtime and dependencies unchanged.

This test does not construct the full OnlineTrainer collector/algorithm loop,
save a distributed training checkpoint, or exercise reward, advantage/filtered
batch semantics, EMA, rollout synchronization or full-weight updates. The
public online recipe still uses physical-rank data ownership and must not be
enabled for CP by merely adding a strategy string. Those integration and
original full online acceptance gates remain required next work.

## Leader-owned replay sharing: 37791c05

The existing rollout schedule module now exposes an explicit
`collect_context_parallel_iteration` function. Every world rank participates;
only CP rank0 invokes the asynchronous collection callback. Leaders serialize
one typed `RolloutIteration` into a private temporary file on a shared
filesystem. CP peers receive only the path/error metadata through their
collective, then all load the same file onto CPU. Large trajectory tensors
are not serialized into GPU object-broadcast buffers. The directory is
validated/created before collection starts and removed after all readers
have reported success/failure.

Ordinary collection and read failures are gathered across the complete world,
so one failed CP group does not let another DP group continue into training
collectives. Job/process death or broken collective transport still requires
external recovery. Only locally created private spool files are deserialized
with the same trusted-object approach as the existing streaming spool; this
is not an external-file import API. All ranks need access to the shared
spool path. Use NVMe for real runs, not the nearly-full root filesystem.

### Four-rank data ownership test

DP2 x CP2 uses the real `PromptBatchSampler` with DP rank/size: CP peers
receive identical prompt indices and the two DP groups receive disjoint
indices. Synthetic typed rollout/trajectory records test rewards, group IDs,
policy version, nested tensors, BF16 values, CPU placement after loading,
and one collection callback per CP group. The trajectory fixture is a
transport record, not a replay-valid generated video.

Leader collection failure, rank1-only read failure, and nonexistent spool
directory all fail coherently across the world. A missing directory fails
before invoking the collector. Temporary files are absent after the readers'
completion barrier, including failure cases.

- Four-L40S NCCL final test: **1 passed, 1 deselected**, 4.58 seconds.
- CPU sharing/iteration/topology regression: **8 passed, 1 skipped**,
  3.43 seconds.
- Ruff and whitespace checks passed. All processes terminal, fresh GPU
  compute inventory empty; GPUs0-3 released. Shared runtime unchanged.

### Integration prerequisites still open

This interface is not automatically installed by the schedule factory. Its
leader callback must be rank-local: it cannot call trainer/world collectives
while other CP ranks wait for collection. The current strict schedule's
lifecycle uses strategy methods, so wrapping it blindly with the CP training
strategy would risk deadlock. Rollout owner lifecycle and weight-sync ownership
must be wired deliberately.

Identical replay files also do not synchronize random decisions. The actual
trainer selects replay timesteps using CPU `torch.randperm`/`torch.randint`;
CP peers must agree on RNG state or selected indices. Existing global RNG
checkpoint helpers capture all CUDA devices, so a rank-local synchronization
path needs care not to create contexts on another owner's GPU. No RNG
synchronization, live generation, actual online loop, or public CP config
enablement is claimed by this milestone.
