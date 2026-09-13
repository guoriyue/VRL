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

## Matched-forward backward trace

`--trace-block-gradients` attaches tensor hooks to each block's full output.
For CP, a detached gradient copy is summed across the two ranks before
comparison; the hook does not return or modify the actual backward gradient.
This accounts for rank-local downstream work and the replicated loss divided
by two. Both CFG calls are tracked separately. The full-shape-conditioning
control uses each branch's actual CPS loss, not a shared output cotangent.

All 56 BF16 forward block outputs match exactly. Block 27 output gradients
also match exactly in both CFG branches. Relative L2 for selected earlier
block output gradients is:

| Block | CFG call 0 | CFG call 1 |
| --- | ---: | ---: |
| 27 | 0 | 0 |
| 26 | 0.0000259116 | 0 |
| 25 | 0.00185231 | 0.00175426 |
| 21 | 0.00779608 | 0.00765408 |
| 14 | 0.0107842 | 0.0109955 |
| 7 | 0.0133349 | 0.0130731 |
| 0 | 0.00755762 | 0.00738407 |

Aggregate BF16 parameter-gradient relative L2 is 0.0265821; the largest
individual relative error is 0.0609939 for block 3 self-attention K LoRA B.
FP32 aggregate error is 2.40103e-5. Both ranks' full case reports match.

This establishes matching block-boundary forward states, then divergence
during backward propagation in this case. It does not establish equality of
every internal activation, a collective bug, or an acceptable training error.
Next isolation should examine backward inside the final blocks, separating
local projection/reduction arithmetic from attention backward. Production
integration and full-resolution acceptance remain open.

Evidence: `cosmos_cp_backward_blocks_l40s/rank-{0,1}.json` and adjacent log.
Torchrun exits 0, fresh compute inventory is empty, and GPUs 0-1 are released.
The runtime candidate remains clean; no production or dependency changes.

## Attention backend control

Selecting PyTorch math SDPA for both branches while retaining full-shape
conditioning and actual CPS loss does not remove the backward discrepancy.
BF16 aggregate parameter-gradient relative L2 is 0.0275351, compared with
0.0265821 for the traced efficient-SDPA control. All 56 BF16 block outputs,
the final output, and logprob match within each reference/CP comparison.
Final block output gradients match, but block 26 relative errors are
0.000657616 and 0.000741438 in the two CFG calls. FP32 aggregate parameter
gradient error is 2.81032e-5.

This is evidence against attributing the discrepancy solely to efficient
attention. It is not an efficient-versus-math output equivalence test:
changing the backend also changes the reference numerical trajectory.
Evidence: `cosmos_cp_math_sdpa_l40s/rank-{0,1}.json` and adjacent log.
Both ranks match and torchrun exits 0. No production backend or threshold
changed; the diagnostic rejects math selection for its separate native-CP
mode, whose implementation was only exercised with efficient SDPA.

## Gather communication precision control

`--fp32-gather` promotes differentiable gather inputs to FP32 and casts the
concatenated result back to its original dtype. It covers block outputs,
self-attention K/V and the full-shape conditioning projections. This promotes
the gather backward's cross-rank accumulation without changing the forward
BF16 values. It does not promote local operator arithmetic or claim a memory
optimization. This control returns to efficient SDPA.

BF16 aggregate parameter-gradient relative L2 is 0.0271955, with all block
outputs, final output and scalar logprob still exactly matching. FP32 aggregate
error is 2.39783e-5. Both ranks' full case reports match and torchrun exits 0.
Evidence: `cosmos_cp_fp32_gather_l40s/rank-{0,1}.json` and adjacent log.
The absence of a material reduction rules out low-precision gather accumulation
as the sole explanation in this case, not every possible distributed numeric
effect. Local backward projections/reductions remain to be isolated.

Both backend/communication controls are terminal, fresh compute inventory is
empty, and GPUs 0-1 are released. No production runtime, dependencies or
thresholds changed. These controls do not establish training acceptance.

## Block-26 internal backward trace

The diagnostic option `--trace-module-gradients 26` records output values and
tensor gradients of norm1/attn1/norm2/attn2/norm3 and FF submodules. CP tensors
at these points are local token slices; detached copies are all-gathered in
token order, not summed or rescaled. Actual backward gradients are unchanged.
This control uses efficient SDPA, full-shape conditioning, ordinary gather
precision and each branch's CPS loss.

All recorded BF16 forward values match exactly in both CFG calls. Selected
output-gradient relative L2 errors, ordered in backward direction:

| Module output | CFG call 0 | CFG call 1 |
| --- | ---: | ---: |
| ff / ff.net.2 | 0.0000906487 | 0.0000394666 |
| ff.net.0 | 0.000236147 | 0.000126368 |
| norm3 (FF input) | 0.000578365 | 0.000335665 |
| attn2 | 0.000400255 | 0.000270394 |
| norm2 (cross-attention input) | 0.00538186 | 0.00191891 |
| attn1 | 0.000482450 | 0.000290001 |
| norm1 (self-attention input) | 0.00365589 | 0.00359196 |

Attention backward increases the relative discrepancy at these boundaries;
this is not yet a localization to SDPA versus Q/K/V/output projections,
nor proof of a faulty operator. Residual paths mean the table is not a single
unbranched gradient chain. Aggregate BF16 parameter-gradient relative L2 is
0.0277477, FP32 2.38874e-5. Small variations from earlier runs are preserved,
not hidden by substituting previous measurements.

Evidence: `cosmos_cp_block26_internal_l40s/rank-{0,1}.json` and adjacent log.
Both rank case reports match; torchrun exits 0, fresh compute inventory is
empty and GPUs 0-1 are released. No production changes or acceptance pass.
Next isolation should capture attention-internal gradients, distinguishing
replicated cross-attention K/V from token-sharded self-attention tensors.

## Attention-internal trace

The block-26 trace now includes Q/K/V projections, Q/K normalization and
output projection. Self-attention tensors and cross-attention Q/output tensors
are token-sharded and gathered for comparison. Cross-attention K/V and norm_k
are replicated: forward values are compared directly, while detached gradient
copies are summed across ranks in FP32. Actual gradients are not changed.

All traced BF16 forward outputs match. CFG call 0 output-gradient relative L2:

| Module | Self-attention | Cross-attention |
| --- | ---: | ---: |
| to_out.0 | 0.000453839 | 0.000373945 |
| norm_q | 0.000994751 | 0.00392359 |
| to_q | 0.00111407 | 0.00247088 |
| norm_k | 0.00322613 | 0.00304969 |
| to_k | 0.00486029 | 0.0212226 |
| to_v | 0.00288178 | 0.00300477 |

The cross-attention K normalization backward is a specific amplification
boundary: about 0.305% error at its output cotangent versus 2.12% at its input
cotangent. This does not prove the entire discrepancy originates there or
constitute an operator bug. Replicated nonlinear operators receive partial
cotangents on each rank; finite-precision local backward followed by summation
need not numerically match backward on a summed cotangent. A targeted next
control should test that ordering at cross-attention norm_k without changing
forward values or confusing replicated and sharded gradient conventions.

Aggregate BF16 parameter-gradient relative L2 is 0.0271407, FP32 2.40468e-5.
Both rank reports match. Evidence:
`cosmos_cp_attention_internal_l40s/rank-{0,1}.json` and adjacent log.
Torchrun exits 0; fresh compute inventory empty and GPUs 0-1 released.
Production runtime, dependencies and acceptance thresholds remain unchanged.

## Cross-key cotangent ordering control

`--sync-cross-key-cotangent` averages each replicated cross-attention norm_k
output cotangent across ranks in FP32, casts back to the original gradient
dtype, and then runs local normalization backward. Final parameter gradients
are still summed. For identical replicated states, averaging before the
local Jacobian and summing the resulting parameter gradients preserves the
real-arithmetic total; this diagnostic does not establish bitwise equivalence
or support higher-order differentiation.

The first attempt failed in the newly added hook because cloning preserved
a noncontiguous layout rejected by NCCL. Evidence remains at
`cosmos_cp_crosskey_cotangent_l40s` with its adjacent log; it is not a model
capacity failure. Explicit contiguous-format cloning fixes that probe error.

The corrected run, `cosmos_cp_crosskey_cotangent_contiguous_l40s`, completes
both dtypes with matching rank reports. BF16 block-26 CFG-call-0 norm_k output
gradient relative L2 is 0.00356277 and its input gradient error is 0.0148262,
versus 0.00304969 and 0.0212226 in the preceding control. However aggregate
parameter-gradient relative L2 is 0.0274953, versus 0.0271407 before, so there
is no demonstrated global improvement. FP32 aggregate error is 2.41074e-5.
Traced BF16 forward values still match exactly.

This ordering change reduces part of one local discrepancy, but is not a
sufficient remedy and is not promoted to production. Existing cotangent
differences and other local backward paths remain. No acceptance threshold
changed. Both jobs are terminal (failed attempt exit 1, corrected exit 0),
fresh compute inventory is empty, and GPUs 0-1 are released.

## Q/K normalization input precision control

Inspection of the installed Diffusers RMSNorm shows an FP32 variance branch
while the original input also participates directly in normalization. The
diagnostic `--fp32-qk-norm-input` passes FP32 inputs to the unchanged vendor
forward in all self/cross Q/K norms, then casts outputs back. Both reference
and CP receive this change. Parameter storage is unchanged; this is not a
claim that the original implementation is mathematically incorrect.

With full-shape conditioning and actual CPS loss, BF16 aggregate parameter
gradient relative L2 is 0.0261260. Block-26 CFG-call-0 cross-key norm output
gradient error is 0.00325701 and input error is 0.0132187. The local input
error is lower than the preceding unmodified trace's 0.0212226, but the global
discrepancy remains substantial. FP32 aggregate error is 2.41754e-5. Traced
BF16 forward outputs and final reference/CP outputs match exactly. This does
not compare the modified reference tensor directly against the old reference.

Evidence: `cosmos_cp_fp32_norm_input_l40s/rank-{0,1}.json` and adjacent log.
Both rank reports match; job exits 0 and fresh compute inventory is empty.
GPUs 0-1 are released. No production fix or acceptance pass is claimed.
Precision changes to normalization alone have not resolved the remaining
backward mismatch, so further work must retain the full gradient comparison.
