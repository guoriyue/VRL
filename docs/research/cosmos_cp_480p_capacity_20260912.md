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
