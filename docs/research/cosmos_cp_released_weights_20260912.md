# Released Cosmos 2.5 weights: two-rank CP diagnostic

Status: released-network forward/backward diagnostic completed; production
P0/P1 and real rollout/replay/update remain open. Neither finite gradients
nor small FP32 errors establish the original BF16 training acceptance.

## Weights and data movement

Use preset revision `nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064`.
The original approximately 20 GiB root-disk cache remains unchanged. Copying
the entire cache was slow and was explicitly stopped; its partial destination
was moved out of the HF cache to
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_partial_cache_copy`.

Instead, selected transformer, scheduler and model-index files were fetched
directly to NVMe in about 25 seconds. Four files total 4,118,418,669 bytes.
All content-address hashes pass verification (SHA-256 for the large weight
blob, Git blob SHA-1 for metadata). The remote weight metadata resolves to the
same pinned commit and hash
`469752430ae6615f119c6fbe061bb7851e85f8a4a22918524cc8abd94d53d493`.
This does not establish completeness of the text encoder/VAE cache on NVMe.

Verification script and receipt:
`/mnt/nvme/outputs/wan22_i2v_cache/verify_cosmos_cache.py` and
`cosmos_cache_verification.json`. The original copy exited 143; selected
downloader and verifier exited 0. No required transfer remains running.

## Full-network diagnostic

The same temporary gather-K/V probe now loads the actual 28-layer 2B
transformer and applies rank-32/alpha-64 LoRA to the preset's six target
families. Each rank compares an unsharded reference with a two-rank context
parallel copy, same base weights, adapters, synthetic inputs and squared-output
loss. Original nontrainable weights stay frozen. Test FP32 first, then BF16.

Input latents `[1,16,2,16,16]`, six synthetic text tokens of width 100352,
partial conditioning mask and per-frame timestep values 0.1/0.5. These are
deliberately small numerical-probe inputs, not the full 512p/93-frame recipe,
real text encoding, CFG wrapper, native scheduler replay or a learning batch.
No previous-policy adapter, optimizer step, FSDP, checkpoint recomputation or
resume comparison is exercised here.

Both rank reports agree:

| Metric | FP32 | BF16 |
| --- | ---: | ---: |
| Output max absolute error | 3.09944e-6 | 0.03125 |
| Output relative L2 error | 5.63571e-7 | 0.00771812 |
| Worst LoRA gradient relative L2 | 6.37569e-6 | 0.0579797 |
| Compared gradient tensors | 560 | 560 |
| Nonzero gradient tensors | 280 | 280 |

All gradients and outputs are finite. Nonzero gradients are the LoRA B tensors;
A gradients are initially zero because B starts at zero. The largest relative
gradient discrepancy is block 14 self-attention K-projection LoRA B in both
precision modes. Its BF16 max absolute discrepancy is 4.81606e-5.

The longer real network amplifies BF16 discrepancy beyond the two-block
diagnostic. Do not claim equivalence from the tiny model's identical output,
and do not use FP32 results to close BF16 gates. The script asserts finite
values/nonzero gradient presence and reports differences; it does not contain
a production gradient tolerance or a full P0 pass verdict.

## Evidence and next requirement

Script: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_network_probe.py`, using
`--method gather_kv --model-path <pinned NVMe snapshot>`.
Reports: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_realweights_l40s/rank-{0,1}.json`.
Log: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_realweights_l40s.log`.
Runtime candidate `382d0825` and shared dependencies stayed unchanged.

Both CUDA ranks and torchrun are terminal, exit 0. Fresh compute inventory is
empty and GPUs 0-1 are released. No Ray or old long queue was started.

Next measure the actual family CFG/replay/logprob and training objective under
native precision with the original thresholds, then checkpoint/FSDP reduction
semantics. Only after numerical acceptance may the full P1 memory and
paper-shaped training gates be evaluated. The current per-block output gather
is a diagnostic convenience, not a measured production-memory solution.
