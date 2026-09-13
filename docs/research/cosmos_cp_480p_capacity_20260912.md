# Cosmos 480p/33f DiT capacity preflight

Status: full-shape synthetic DiT forward/backward completes on two L40S and
one L40S. Not full-family replay, GRPO, optimizer/resume or completed CP P1.

## Scope

Pinned Cosmos Predict2.5 2B revision
`0d37c7498f54cee3c599d438d895a0a4a8608064`, frozen runtime `382d0825`.
Pinned VAE temporal/spatial compression is 4/8. The original 480x832/33-frame
target maps to latent `[1,16,9,60,104]`, 14,040 transformer tokens. Synthetic
text features are `[1,512,100352]`; no text encoder/VAE/reward is loaded.

BF16 base, actual family LoRA construction, FP32 LoRA compute, seeded nonzero
B std 1e-3 and padded 64-row Linear tiles on both baselines. Two ranks use
head sharding; one rank does not. GPU-only nonreentrant checkpointing, no CPU
offload. Objective is squared transformer output, not production CPS/GRPO/NFT.
All 560 trainable gradient tensors are finite and nonzero in both cases.

## Results

Peaks are PyTorch bytes, not total device usage. Backward includes gradient SUM.

| Case | Forward seconds | Backward seconds | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| CP rank 0 | 9.462 | 19.244 | 8,998,452,736 | 9,376,366,592 |
| CP rank 1 | 9.323 | 19.239 | 8,998,452,736 | 9,376,366,592 |
| Single rank | 17.096 | 36.260 | 11,220,165,120 | 11,679,039,488 |

Rank-0 summed phase timing is 28.71s versus 53.36s, about 1.86x for this one
cold step, excluding loading and pipeline work. Allocated peak per device is
about 19.8% lower. Both cases fit below 32 GB: the historical single-card OOM
is NOT reproduced under this checkpointed synthetic setup. This is neither
sustained training throughput nor proof of a multi-GPU-only capacity unlock.

## Failure and evidence

The first attempt completed forward but failed checkpoint metadata validation:
the efficient-SDPA context ended before backward and recomputation selected
a different backend. Keeping backward in the same context fixes the probe;
the consistency check was not disabled. All artifacts remain under
`/mnt/nvme/outputs/wan22_i2v_cache/`:

- `cosmos_cp_memory_probe.py`: driver, records imported diagnostic source hash.
- `cosmos_cp_480p_capacity_l40s`: retained failed attempt, exit 1.
- `cosmos_cp_480p_capacity_backendfix_l40s`: two-rank result, exit 0.
- `cosmos_single_480p_capacity_l40s`: single-rank result, exit 0.

Each run has rank JSON and adjacent log. Full-shape family logprob parity,
real update/resume, rollout precision consistency and production mesh/lifecycle
integration remain open. The prototype still gathers between blocks and uses
gather-based head exchanges. All three jobs terminal; fresh compute inventory
empty, GPUs 0-1 released. No runtime/dependency edits or long queue restart.

## Full-shape family/CPS follow-up (September 13)

The numerical probe now accepts rectangular latent dimensions, temporal
length, text length and GPU checkpointing. Trace hooks assuming one forward
per block are rejected when checkpoint recomputation is enabled. Both
forward and backward remain inside the same selected SDPA backend context.

At `[1,16,9,60,104]` with 512 synthetic text tokens, actual family adapters
(nonzero B std 1e-3), local padded 64-row linears, FP32 LoRA and head-sharded
attention, the real family forward_step and CPS fixed-action loss complete
at scheduler index 18, sigma 0.0281334, CFG5. Both precisions have exactly
matching output and scalar logprob. All 280 A and 280 B gradients are nonzero.

| Precision | Rank 0 aggregate gradient relative L2 | Rank 1 |
| --- | ---: | ---: |
| FP32 | 4.41248e-6 | 4.34723e-6 |
| BF16 | 0.0343482 | 0.0350728 |

The full-size gradient result does not match the earlier small-input result.
Both rank reports have identical forward/logprob metrics but differ in 550
parameter-gradient error records. Comparison gradients are summed across
ranks; each rank has its own independently executed reference backward.
Do not assume all discrepancy is CP-specific without a matching full-shape,
checkpointed unsharded control. The earlier exact small unsharded baseline
does not cover this shape/backend/checkpoint workload.

Evidence: `cosmos_fullshape_family_parity_l40s/rank-{0,1}.json` and adjacent
log under the same NVMe root. This crosses the full-shape scalar diagnostic
gate, not full trajectory, reward, update/resume or gradient-equivalence
acceptance. Both rank jobs terminate and torchrun exits 0; fresh GPU inventory
is empty, GPUs 0-1 released. Runtime and thresholds unchanged.

## Matching unsharded family baseline (September 13)

`cosmos_fullshape_replicated_baseline_l40s` changes only the preceding
full-shape numerical probe's method from Ulysses to replicated. Same latent
and text shapes, nonzero adapters, padded linears, FP32 LoRA, CFG5, CPS
action, GPU checkpoint and efficient SDPA. Comparison loss is still divided
by two and comparison parameter gradients summed across ranks. This isolates
the effect of removing sequence/head sharding, not a throughput comparison.

| Precision | Rank 0 aggregate gradient relative L2 | Rank 1 |
| --- | ---: | ---: |
| FP32 | 3.48174e-6 | 3.46183e-6 |
| BF16 | 0.0295807 | 0.0285717 |

Both ranks have exact output/logprob agreement and all 280 A plus 280 B
gradients nonzero. Unsharded backward comparison therefore exhibits sizeable
variation at this shape. The preceding 0.03435-0.03507 CP discrepancy cannot
be attributed wholly to CP. Relative L2 norms cannot be subtracted to infer
an isolated CP contribution; neither measurement proves training parity.
The appropriate next control is deterministic backward execution on both
paths, with unsupported deterministic operations failing explicitly.

Frozen script snapshot for both full-shape family runs:
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_fullshape_probe_38cc13de.py`, SHA256
`38cc13de1b36977b43259c099484c6826bc8d7654dc78c90f5e671db35f3e6af`.
The snapshot matches the source used by these jobs byte-for-byte. Evidence
is rank JSON and adjacent log under the baseline directory in the same root.
Torchrun exits 0, fresh compute inventory is empty, GPUs 0-1 released.
Runtime worktree remains clean; no production settings or thresholds changed.

## Strict deterministic full-shape comparison (September 13)

Enable `torch.use_deterministic_algorithms(True)` (not warn-only), disable
cuDNN benchmarking, and launch with `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
CUDA initialization. The diagnostic rejects missing workspace configuration.
Preserve the full shape, nonzero adapters, local padded linears, FP32 LoRA,
head sharding, actual family CPS loss, CFG5 and GPU checkpoint. Neither
checkpoint validation nor unsupported-operator checks are bypassed.

Both ranks' complete case reports now match. FP32 aggregate parameter-gradient
relative L2 is 2.24143e-6 and BF16 is 2.22439e-6. Final family output and CPS
logprob max errors are exactly zero in both precisions. All 280 A and 280 B
gradient tensors remain nonzero. This eliminates the previously observed
percent-level drift in this controlled deterministic configuration; it does
not prove which individual nondeterministic kernel contributed each error.

Evidence: `cosmos_fullshape_deterministic_cp_l40s/rank-{0,1}.json` and adjacent
log. Immutable `probe_source.py` in that run directory has SHA256
`19457906ab7ecc1685308959d01300084331f317810ea731938e5644950860b1`.
Reports explicitly record deterministic mode and workspace configuration.
Stage logs distinguish load, reference/comparison forward/backward and case
completion, avoiding opaque waits in subsequent diagnostics.

This is full-shape one-step numerical evidence, not a generated trajectory,
real reward/GRPO update, checkpoint/resume or production mesh integration.
The earlier 1.86x capacity timing did NOT use strict deterministic mode and
must not be presented as throughput for this final numerical configuration.
Re-measure performance and memory after production integration. Torchrun
exits 0, fresh compute inventory empty; GPUs 0-1 released. Runtime unchanged.

## Native optimizer/checkpoint continuation (September 13)

Full-shape BF16 deterministic family/CPS gradients now drive the repository's
`build_optimizer` factory (fused AdamW, OptimConfig lr 1e-4 and its defaults),
with gradient clipping at 1.0. First gradient norm is 9.69262e-6 and all 560
trainable tensors change. Single-reference versus two-rank updated parameter
relative L2 is 1.26901e-9. This compares parameter values, not relative update
deltas, and does not demonstrate reward improvement.

The probe uses native `save_training_checkpoint`, `TrainingCheckpoint.load`,
`restore_training_checkpoint`, owned-state export and RNG capture/restore.
Its minimal state carrier stores optimizer/progress; it is not OnlineTrainer
and does not include EMA, the online optimizer manifest or rollout state.
Each rank publishes to its own diagnostic checkpoint directory, not the
production distributed writer topology.

After step 1, the saved optimizer has 560 entries and 1,120 nonzero Adam moment
tensors on each rank. The probe then computes a fresh second CPS gradient and
update, snapshots the result, creates a new optimizer, strictly restores step
1 model/optimizer/RNG, and recomputes the same second update. Native exact-tree
checks pass for owned model state and optimizer/progress; second-step loss is
also tensor-exact. Both ranks report success. This is real optimizer-state
continuation, not reusing saved gradients or merely loading a file.

Evidence: `cosmos_fullshape_optimizer_resume_l40s/rank-{0,1}.json`, adjacent
log and `bfloat16/checkpoint-rank-{0,1}/checkpoint.pt` under the NVMe root.
Immutable `probe_source.py` in the run directory has SHA256
`34ea6064e271e546e16f63de67bc6819c1831102644b2f86bb93fed64cfcd16d`.
Torchrun exits 0, fresh compute inventory empty; GPUs 0-1 released.

The fixed-action CPS objective still has synthetic conditioning, no reward,
advantage computation, online loop or generated trajectory. Production CP
mesh/lifecycle integration, complete GRPO/EMA acceptance and final-config
performance remain open. Do not label this as completion of the full sprint.

## Deterministic warmed DiT timing (September 13)

Remeasure the original squared-output DiT workload with strict deterministic
algorithms, cuBLAS workspace :4096:8, GPU checkpoint, padded 64-row linears
and FP32 LoRA. Same pinned weights/nonzero B, input dimensions and synthetic
text features. Each arm runs one excluded warmup and two measured iterations;
clear gradients and synchronize before timing. Backward includes gradient SUM.
No optimizer step, CFG/family wrapper, reward or I/O is included in timing.

| Case | Measured forward seconds | Measured backward seconds | Total seconds | Peak allocated GB |
| --- | --- | --- | --- | ---: |
| CP rank 0 | 12.913, 12.989 | 41.180, 41.028 | 54.093, 54.017 | 9.048 |
| CP rank 1 | 12.910, 12.985 | 41.183, 41.032 | 54.093, 54.017 | 9.048 |
| Single rank | 24.841, 24.714 | 66.799, 66.114 | 91.640, 90.829 | 11.270 |

Take the slower rank's total per CP iteration, then average: 54.0552 seconds
versus 91.2344 seconds single rank, 1.6878x and 40.75% less elapsed time.
Allocated peak per device falls about 19.71%. Reserved peaks are 9.469 GB
per CP rank and 11.713 GB single rank. GB here is decimal. All iterations
have 560 nonzero finite gradients. Two measured samples do not establish
long-run stability or full online training speedup.

This replaces the earlier nondeterministic 1.86x as the DiT-only reference
for the current deterministic candidate. It does not benchmark the CFG5 CPS
acceptance workload or claim that a production training integration exists.

Evidence directories in the same NVMe root:
`cosmos_deterministic_cp_timing_l40s` and
`cosmos_deterministic_single_timing_l40s`, rank JSON/progress JSON and logs.
Frozen scripts in the CP timing directory preserve their sibling import path:
memory driver SHA256 `b00cb36b6c69657ed1ddc4fbf02082d8d43f9841d55411d74bdc1e4a5f5a83a8`;
network definitions SHA256 `34ea6064e271e546e16f63de67bc6819c1831102644b2f86bb93fed64cfcd16d`.
Both jobs exit 0, fresh GPU inventory empty; GPUs 0-1 released. Runtime clean.
