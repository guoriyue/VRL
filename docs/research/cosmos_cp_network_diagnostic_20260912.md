# Cosmos network CP path diagnosis on two L40S GPUs

Status: small-network diagnostics completed; released-model P0/P1 and training
integration remain open. The results select an experimental next path, not a
production CP implementation. No shared dependencies/runtime files changed.

## Scope and implementation

Two-block real `CosmosTransformer3DModel`, random weights, two frames, partial
conditioning, distinct per-frame timesteps, text cross-attention, rotary and
learned positional embeddings. Compare identical state/inputs under FP32 then
BF16. This is not the pinned pretrained Predict2.5 network, CFG replay, an SDE
trajectory, checkpointed backward, LoRA/FSDP composition or a full-size workload.
All model parameters are cast to the tested dtype; native mixed-module precision
contracts still require a separate production-model test.

The temporary script splits video tokens, sequence-aligned timestep embeddings
and positions at each block boundary, leaves text replicated, and uses CP only
inside self-attention. Each block output is differentiably gathered. Replicated
loss is divided by two and parameter gradients are summed across ranks; this
explicit diagnostic scaling must not be copied blindly into a DDP/FSDP strategy
that applies its own reductions. Fifty parameter gradients are compared by name.

The installed Cosmos model has no `_cp_plan`; its attention processors do not
consume the generic Diffusers `_parallel_config`. Enabling generic CP hooks
alone is therefore not an established Cosmos training implementation. The
probe wraps modules locally and uses existing PyTorch attention/collectives;
it does not edit vendor code or install a global training feature.

## Results

First, a 32-token sequence (16 per rank, latent spatial size 8) fails inside
native PyTorch efficient-attention CP output/LSE merging: size 32 versus 16.
Both ranks exit 1 before numerical comparison. The failure is preserved;
the larger-sequence diagnostic does not repair or accept this short case.

At 128 tokens (64 per rank, spatial size 16), both ranks agree:

| Method / dtype | Output max abs | Output relative L2 | Worst parameter-gradient relative L2 |
| --- | ---: | ---: | ---: |
| Native CP / FP32 | 5.36442e-7 | 2.20105e-7 | 5.12968e-7 |
| Native CP / BF16 | 0.0078125 | 0.00383277 | 0.721760 |
| Gather full K/V / FP32 | 5.96046e-7 | 2.24268e-7 | 5.09042e-7 |
| Gather full K/V / BF16 | 0 | 0 | 0.00480953 |

Native CP's largest BF16 discrepancy is block-0 self-attention Q normalization;
Q/K projection gradients also have relative L2 errors of roughly 0.31-0.49.
The native primitive-only finite/logprob test therefore was insufficient to
justify a full-network backward claim. The gathered-K/V path calls ordinary
SDPA with local Q and differentiably gathered K/V, avoiding partial-attention
output merging. This controlled change sharply reduces observed gradient drift;
it suggests the native CP path needs more investigation, not a proven upstream
root cause or a universal BF16 defect.

The script asserts finite output/gradients and records numerical errors; its
`finite_network_diagnostic_complete` status is NOT an equivalence pass. No
production gradient tolerance was introduced or relaxed. Native CP and gather
K/V longer-sequence jobs both exit 0, and all three sessions are terminal.

## Evidence and next gate

Script: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_network_probe.py`.
Each output directory has rank-0/rank-1 JSON; matching `.log` files are adjacent:

- `cosmos_cp_network_l40s`: initial 8x8 short-sequence failure.
- `cosmos_cp_network_128tokens_l40s`: native CP larger-sequence results.
- `cosmos_cp_network_gatherkv_l40s`: `--method gather_kv` comparison.

All paths above are under `/mnt/nvme/outputs/wan22_i2v_cache/`.
The initial short failure can be reproduced with `--spatial-size 8`; the
current default is 16. GPU 0/1 were exclusively claimed; fresh final compute
inventory is empty and the claim is released. No Ray or old queue restart.

Prefer full-K/V collection as the next numerical baseline allowed by the
original sprint. Before production integration, validate the pinned model's
native precision, real CFG/replay/logprob and gradients, checkpoint recomputation
and CP/FSDP reduction semantics. Gather-per-block and full K/V have nontrivial
communication/storage cost: do not infer the original <32GB per-rank P1 target
or a throughput improvement from this diagnostic.
