# SD3.5 execution-mode precision isolation

A follow-up to the [batch/dtype experiment](sd3_5_batch_precision_20260909.md)
preserves the model's loaded parameter dtypes: 908 BF16 parameter tensors and
486 FP32 trainable adapter tensors. It executes backward as well as forward and
compares eager and production `torch_compile_transformer("default")` paths.
The experiment completed successfully on the shared RTX 5090 on September 9,
2026 Pacific, from clean revision `3a2b66a59`.

[Raw results](sd3_5_execution_precision_20260909.json) and the
[exact diagnostic source](sd3_5_execution_precision_20260909_probe.txt) are archived.
The script is a one-off diagnostic, not a supported CLI. It ran with
`CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17 PYTHONPATH=.` and the repository
virtualenv; local log: `/tmp/vrl_sd3_execution_precision_probe.log`.

## Comparison

The source recreates the prior seed-17, prompt-index-1971, batch-16 eager trajectory
and retains its step-8 transition. It never calls a whole-model dtype conversion.
Each arm compares a no-grad batch-16 forward with sixteen batch-1 replay-state
forwards on the same observations, actions, timesteps and conditioning. The
backward arms differentiate the sum of production SDE log probabilities for each
single-row forward and clear gradients between rows. No optimizer step occurs.
The inherited model forward wrapper supplies the resolved BF16 outer autocast.
Compilation happens only after the eager arms; all arms reuse eager actions.

| Transformer | Single-row replay | Max batch logprob difference | With FP32 CFG recombination |
|---|---|---:|---:|
| Eager | No-grad | 0.018531210720539093 | 0.018472250550985336 |
| Eager | Grad and backward | 0.018531210720539093 | 0.018472250550985336 |
| Compiled | No-grad | 0.0043965913355350494 | 0.004400096833705902 |
| Compiled | Grad and backward | 0.0037156082689762115 | 0.0036974921822547913 |

Both backward arms observed finite existing parameter gradients. This is not an
assertion of gradient equality or of nonzero gradients for every trainable tensor.
The FP32 CFG column recombines the same returned conditional/unconditional branch
outputs in FP32 before log-density evaluation; it does not rerun the trajectory
with a different CFG policy. Its small effect does not remove eager batch error.
Conditional and unconditional branch differences already exist before CFG:
maximum eager differences are 0.3125 and 0.3212890625 respectively.

Keeping FP32 adapters and enabling backward did not change this eager result.
That narrows the investigation beyond the earlier whole-transformer cast caveat.
Compiled outputs differ from eager outputs, and compiled grad/no-grad execution
also produces different measured differences. None of these fixed-transition
results establishes parity for the actual compiled rollout trajectory: relative
to original eager rollout logprobs, compiled single-row replay still differs by
0.01868427 (no-grad) or 0.01800328 (grad). The production run's failure remains
unresolved. One sample group and one transition do not establish complete curves,
repeatability, post-update behavior or recipe-wide numerical acceptance.

## Boundaries

No production code, numerical gate, batch size or recipe changed. The existing
model-owned autocast wrapper and compile method remain necessary shared boundaries.
No extra contract dataclass or global family table is introduced. Further diagnosis
must identify where branch outputs begin diverging and validate any selective
precision intervention against actual compiled generation and replay.

## Localizing the batch difference

A second eager diagnostic compares sample 0's unconditional/conditional module
outputs in batch 16 with that same sample alone, and compares final logprobs for
all 16 rows. It preserves initial adapter dtypes and the same step-8 transition.
The [measurements](sd3_5_sensitive_modules_20260909.json) and
[exact source](sd3_5_sensitive_modules_20260909_probe.txt) are archived. This ran
after the first experiment, with documentation-only untracked files present; no
production source changed. The local log is
`/tmp/vrl_sd3_sensitive_modules_group_probe.log`.

Sample 0's positional embedding output is exactly equal across batches. Its
`time_text_embed` output differs by up to 0.25 and `context_embedder` by up to 4.0
before the first transformer block. These are absolute differences, not relative
error measures. Hooks inspect only explicitly selected modules and only sample 0's
two CFG branches; this is not an exhaustive operator trace.

The experiment then disables
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`, separately
promotes `time_text_embed` parameters and floating inputs to FP32 with autocast
disabled inside that module, and finally combines both interventions. The module's
output is cast back to BF16 before the rest of the transformer. The process keeps
its existing TF32 policy; FP32 here describes storage and the autocast boundary,
not a claim of IEEE-only kernel multiplication. No checkpoint reloading recovers
precision already absent from the BF16 checkpoint.

| Eager intervention | Max logprob batch difference, all 16 rows | Difference from original rollout logprobs |
|---|---:|---:|
| Original | 0.018531210720539093 | 0.018531210720539093 |
| Disable BF16 reduced-precision reduction | 0.002146463841199875 | 0.015066094696521759 |
| FP32 timestep/text embedding boundary | 0.005175154656171799 | 0.017813026905059814 |
| Both | 0.0 | 0.013645537197589874 |

With both interventions, the sampled embedding outputs, selected block outputs
and final projection also match exactly across batches. Internal FP32 timestep
linear outputs still have tiny differences that disappear when the embedding
output is cast back to BF16. The all-row claim covers final logprobs, not every
intermediate tensor for all samples.

This identifies a concrete selective-precision candidate instead of requiring a
whole-model FP32 conversion. It does not prove the candidate works for compiled
execution, backward, other prompts, other steps or updated adapters. The original
rollout comparison remains above 0.01: a replay-only intervention is insufficient.
The next acceptance must regenerate trajectories with the candidate applied to
both generation and replay, retain actual trainable dtypes, run the original
parity gate and measure memory/performance. No production policy is changed on the
strength of this one-transition diagnostic. The observed reduction flag also needs
to be recorded in environment-bound numerical evidence.

## Fresh compiled trajectory rejects the two-part candidate

The candidate was applied before generating a fresh compiled trajectory, with
the same configured prompt, seed, shape and flow-GRPO schedule. All trainable
adapters remain FP32; eight frozen conditioning parameter tensors are promoted
from BF16 to FP32. BF16 reduced-precision reduction is disabled. Generation and
replay both use the same candidate and the production default compile method.
[Results](sd3_5_candidate_compiled_20260909.json) and
[executed source](sd3_5_candidate_compiled_20260909_probe.txt) are preserved.
The run started from clean production revision `135c4390a`; the subsequent
`7f4c1a559` change only adds evidence capture and does not alter this probe's model
or numerical execution. Local log: `/tmp/vrl_sd3_candidate_compiled_probe.log`.

At step 8, the maximum original rollout versus batch-1 replay logprob difference
is 0.004651270806789398 without gradients, but 0.012948013842105865 with backward.
Observed gradients are finite. FP32 CFG recombination still gives a batch
difference of 0.012934364378452301 in the backward arm. No optimizer update or
production trainer acceptance is claimed. The backward mismatch exceeds the
unchanged 0.01 limit, so the candidate is not ready for production.

This supersedes any inference that the eager fixed-transition zero implies
compiled training success. Before broader prompt/step acceptance, the next
isolation checks Inductor's treatment of low-precision intermediate rounding.
The installed PyTorch source (`torch/_inductor/config.py`,
`emulate_precision_casts`) explicitly describes removal of intermediate
downcast/upcast pairs during fusion and provides an opt-in way to retain them.
That source observation motivates an experiment; it does not prove causality.
