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
