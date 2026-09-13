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

## First-block localization

Per-block hooks on the same shared-cotangent CFG5 diagnostic show BF16
divergence already at block 0 (relative L2 0.00313467), increasing to
0.0126416 at block 27 in the first CFG branch. FP32 block 0 and block 27
relative errors are 4.72727e-7 and 1.22734e-6 respectively. Both rank reports
match. The 56 recorded events cover 28 blocks in each of two CFG branches,
not 56 independent layers or training updates.

A second trace of first-block submodules finds BF16 relative L2 0.000965296
at norm1's modulated hidden output and 0.000998577 at its gate output,
before the first self-attention call. Self-attention output relative L2 is
0.00201252. This rules out self-attention as the first observed divergence
in this test; it does not establish that attention contributes no further
error, or that a particular kernel is faulty. FP32 norm1 output relative L2
is 2.36213e-7. Both ranks' complete case reports match.

Evidence under the same NVMe root:
`cosmos_cp_block_trace_l40s/rank-{0,1}.json` and
`cosmos_cp_firstblock_modules_l40s/rank-{0,1}.json`, with adjacent logs.
These are synthetic, pinned-weight diagnostics, not a production CP pass.

### AdaLN internals

Follow-up `cosmos_cp_adaln_trace_l40s/rank-{0,1}.json` adds hooks inside
norm1 without changing runtime operators. Both ranks match, and torchrun
exits 0. In the first CFG branch:

| Output | FP32 relative L2 | BF16 relative L2 | BF16 max absolute |
| --- | ---: | ---: | ---: |
| norm1.activation | 0 | 0 | 0 |
| norm1.linear_1 | 0 | 0.00218015 | 0.0078125 |
| norm1.linear_2 | 2.49304e-7 | 0.000927816 | 0.03125 |
| norm1.norm | 0 | 0 | 0 |
| norm1 modulated hidden | 2.36213e-7 | 0.000965296 | 0.199526 |

The first observed BF16 discrepancy is therefore in the timestep-conditioned
AdaLN linear projection, with equal activation inputs and equal plain
LayerNorm outputs. The local versus full sequence changes projection shapes;
shape-dependent numerical execution is a hypothesis to isolate next, not a
verified kernel defect. Shared-cotangent aggregate BF16 gradient relative L2
remains 0.103061, so training equivalence is still open. A useful next control
is identical full-shape conditioning projections before selecting local
tokens, while preserving gradients and separating this from attention changes.

No production code, dependency or acceptance threshold changed. The frozen
runtime worktree remains clean. Fresh compute inventory is empty; the
per-block-trace GPU claim is released.

## Full-shape conditioning control

The diagnostic flag `--full-shape-conditioning` restores full sequence shapes
only for linear_1 and linear_2 in every block's norm1/norm2/norm3. Each
projection differentiably gathers its input, applies the unchanged projection,
then selects the rank-local token slice. Attention remains local-Q/full-KV;
this is deliberately redundant diagnostic work, not a production memory or
throughput optimization. Both ranks still sum replicated parameter gradients.

With the actual family adapters, CFG5, step 18 and shared output cotangent,
BF16 family output and scalar logprob now match exactly. All first-block
submodule traces also match exactly. Aggregate parameter-gradient relative
L2 falls from 0.103061 in the prior traced control to 0.0278607, but does not
vanish. FP32 aggregate gradient relative L2 is 1.01486e-5 and output relative
L2 is 2.96771e-6. Both ranks' reports match and the job exits 0.

This intervention supports conditioning projection shape as a source of the
observed forward discrepancy in this particular synthetic case. Exact final
output is not proof that every later internal activation matches, and the
remaining backward discrepancy still prevents declaring training equivalence.
No tolerance changed. Evidence:
`cosmos_cp_fullshape_conditioning_l40s/rank-{0,1}.json` and adjacent log under
the same NVMe root. Runtime and shared dependencies remain unchanged.

A follow-up removes `--shared-output-cotangent` and differentiates each
branch's actual fixed-action CPS logprob loss. BF16 output and logprob still
match exactly; aggregate gradient relative L2 is 0.0264940, compared with
0.412917 in the native-adapter control without full-shape conditioning.
FP32 aggregate gradient relative L2 is 2.39874e-5. Both ranks match, the job
exits 0, and its evidence is
`cosmos_cp_fullshape_conditioning_ownloss_l40s/rank-{0,1}.json` plus log.
Different backward reductions can still round differently; this result does
not identify the remaining error's source. Next work should isolate backward
paths with matched forward activations before production CP integration.

Both jobs are terminal and fresh compute inventory is empty. GPUs 0-1 are
released. This is a two-rank small-input diagnosis, not full-resolution P1,
a training update, or a four-GPU throughput acceptance.
