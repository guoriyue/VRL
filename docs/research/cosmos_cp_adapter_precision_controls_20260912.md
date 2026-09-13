# Cosmos CP production-adapter precision controls

Status: controlled diagnostics completed; adapter construction/storage dtype
does not explain away the observed BF16 CP gradient discrepancy. This is not
a production equivalence pass, real FSDP/CP mesh test or NFT update.

## Why this control was needed

Earlier probes directly called PEFT with one adapter and
`autocast_adapter_dtype=False`. Actual `CosmosPredict25Model.apply_lora`
uses the family's Gaussian initializer, PEFT's native adapter promotion and a
frozen `previous` mirror. Therefore the earlier BF16 single-adapter evidence
could not alone establish behavior under the family's actual adapter contract.

The probe now invokes the real family method with a typed `ModelBuild`, then
deep-copies that result for the CP branch. It asserts 560 frozen previous
adapter tensors and records trainable dtypes. A second control additionally
uses the actual `normalize_fsdp_parameter_dtype` helper. This changes storage
precision only; it does not install FSDP or exercise its gradient collectives.

Keep the pinned 28-layer model, two ranks, identical reference/CP inputs,
family forward, CPS logprob-loss on the same reference action, low-noise
index 18 and CFG=5. The experimental gather-K/V implementation and its
explicit CP loss/reduction convention stay unchanged. The native-storage
and normalized-storage controls use the same construction path and seed.

## Results

Both ranks' complete per-case result dictionaries match in each control:

| Adapter construction | BF16-forward trainable storage | BF16 logprob max error | BF16 aggregate gradient relative L2 |
| --- | --- | ---: | ---: |
| Actual family, native PEFT | FP32 | 1.35464e-6 | 0.412917 |
| Actual family, FSDP dtype normalization | BF16 | 1.35464e-6 | 0.411823 |

Both controls retain 560 frozen previous tensors. FP32 cases have aggregate
gradient relative L2 about 2.44e-5 and logprob absolute error 7.28e-12.
The fixed scalar 1e-3 diagnostic limit passes; gradient equivalence remains
unproven. No scalar or gradient threshold was relaxed. Similar discrepancy
under both adapter storage paths rules out that simplification as the sole
explanation; it does not identify a kernel-level root cause.

## Evidence

Under `/mnt/nvme/outputs/wan22_i2v_cache/`, rank JSON reports and adjacent logs:

- `cosmos_cp_native_adapters_cfg5_l40s`: actual construction/native storage.
- `cosmos_cp_fsdp_adapters_cfg5_l40s`: retained failed launch with a mistyped
  local model path, no model result. This was an operator error, not capacity.
- `cosmos_cp_fsdp_adapters_cfg5_pathfix_l40s`: corrected normalized-storage run.

The existing `cosmos_cp_network_probe.py` now accepts `--production-adapters`
and `--normalize-fsdp-dtype`, with a local config-file preflight check. Previous
scripts/results remain diagnostic and are not relabeled as native-adapter runs.
Runtime candidate `382d0825` stayed clean; no shared dependency changes.

All three torchrun sessions are terminal; both substantive controls exit 0.
Fresh compute inventory is empty and GPUs 0-1 are released. No Ray or long
experiment was started. Next diagnosis should isolate forward/gradient kernel
path differences under sequence sharding, not retune adapter dtype or accept
scalar logprob agreement as sufficient training semantics evidence.

## Shared-output-cotangent isolation

A further pinned-weight control preserves the actual adapter construction and
CFG5/low-sigma setup, but supplies the reference branch's retained output
gradient to the CP output backward. It does NOT use the CP branch's changed
logprob-loss derivative. Both ranks agree: aggregate gradient relative L2 is
1.06094e-5 in FP32 and 0.103606 in BF16, versus 0.412917 for BF16 when each
branch differentiates its own fixed-action loss.

This indicates that the loss-derivative change amplifies part of the observed
gradient discrepancy. It does not prove a collective backward bug: the two
branches' internal forward activations can still differ, even with identical
output cotangents, and their Jacobians are evaluated along those different
numerical paths. BF16 family output relative L2 remains 0.0439753 in this CFG
stress case. No gradient-equivalence pass or production tolerance is implied.

Evidence: `cosmos_cp_shared_cotangent_cfg5_l40s/rank-{0,1}.json` and adjacent
log under the same NVMe root. Script option `--shared-output-cotangent` marks
the altered diagnostic backward explicitly. Torchrun and both ranks exited 0;
fresh GPU inventory is empty and GPUs 0-1 are released. Runtime unchanged.
