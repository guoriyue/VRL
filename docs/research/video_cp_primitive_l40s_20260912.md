# Native context-parallel primitive on two L40S GPUs

Status: synthetic primitive logprob diagnostic passed; full Cosmos P0 and P1
remain open. No model/replay adapter, training mesh integration or production
default changed. Both processes exited 0 and GPU inventory is empty.

## Setup

Use installed Torch 2.12.0+cu130 native experimental `context_parallel`, with
all-gather rotation, load balancing disabled, noncausal SDPA and the efficient
attention backend. Two NCCL ranks use identical seeded Q/K/V tensors of shape
`[2,4,128,64]`. Compare local-Q/distributed-KV forward and backward against a
full single-device attention reference on each rank, then unshard output and
input gradients. Local loss is divided by CP world size to preserve the global
mean loss. FP32 is tested before BF16.

The [official PyTorch CP tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
describes the native API and its experimental status. Runtime signatures were
also inspected in the installed package. This is a dependency probe, not a
new hand-written attention or communication implementation.

The reconstructed full outputs feed VRL's actual `sde_step_with_logprob`
function with a FlowMatch Euler scheduler, a fixed interior timestep, identical
synthetic observations/actions, FP32 logprob math and noise level 0.7. The
diagnostic's predeclared absolute logprob limit is 1e-3. This artificial SDE
input is not a Cosmos EDM trajectory or the family precision guard itself.

## Observed results

Both rank reports agree:

| Metric | FP32 | BF16 |
| --- | ---: | ---: |
| Attention max absolute error | 1.93715e-7 | 0.00390625 |
| Attention relative L2 error | 1.86949e-7 | 0.00290208 |
| Largest Q/K/V gradient relative L2 error | 3.22839e-7 | 0.00344780 |
| SDE logprob max absolute error | 0 | 1.90735e-6 |

All outputs and input gradients are finite. The script asserts finiteness and
the synthetic logprob limit; gradient error magnitudes are reported, not
asserted against a production gradient-equivalence criterion. A single
attention call's mean logprob can hide elementwise differences. No full-model
numerical gate is closed, and there is no memory-benefit or throughput claim.

Script: `/mnt/nvme/outputs/wan22_i2v_cache/context_parallel_primitive_probe.py`.
Rank JSON reports: `/mnt/nvme/outputs/wan22_i2v_cache/cp_primitive_l40s/`.
Log: `/mnt/nvme/outputs/wan22_i2v_cache/cp_primitive_l40s.log`.
Runtime checkout `382d0825` stayed clean; shared dependencies unchanged.
The scalar metric conversion emits a requires-grad warning; it does not alter
the already-computed backward. No Ray or released model weights were used.

## Next required gate

Integrate the exact model's video-token, positional embedding and conditioning
sharding semantics before repeating full-network output, parameter-gradient
and real replay/logprob comparisons. Global CP wrapping must not inadvertently
shard or duplicate cross-attention text keys. Preserve the original family
thresholds and fail on numerical regression; only after P0 succeeds measure
the original 480p/33-frame per-rank memory target and real update/resume.
