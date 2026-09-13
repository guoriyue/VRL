# Wan 2.2 dual-expert L40S launch preflight

Status (2026-09-13): the earlier real update failed strict replay acceptance.
A subsequent fixed-trajectory matrix now gives exact replay on all 20 real
transitions with native frozen FP32 preservation and matching batch size two.
Batch size one still differs. Gradient/update equivalence and controlled resume
remain open; historical diagnostics and the latest matrix follow below.

Latest gradient gate: matched-batch real one/two-rank gradients differ by
0.403% relative L2 and are not accepted under the fixed 1e-4 gate. Independent
contribution capture now reproduces both results exactly using their respective
BF16 accumulation orders. A subsequent real FP32-LoRA one/two-rank experiment
passes gradient and first-update equivalence; public configuration, updated
rollout delivery and controlled resume remain unverified for that path.

## Reproducible artifacts

Runtime: clean `382d08254848f19e398243348a3e3f35a4f774f5` in
`/home/ubuntu/VRL-mgpu-integration`, shared Torch/Diffusers plus the isolated
Transformers 5.13 overlay. CUDA was hidden throughout this preflight.

- Script: `/mnt/nvme/outputs/wan22_i2v_cache/preflight_dual_expert.py`.
- Report, resolved two-rank config and exact overrides:
  `/mnt/nvme/outputs/wan22_i2v_cache/dual_expert_preflight/`.
- Existing pinned T2V snapshot:
  `Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df9619b54f4e2667b2755bc6a756675b5cd7`.

The script reads safetensor headers, not full weight tensors. It checks unique
tensor names and index-to-shard coverage, resolves/parses the existing proof
recipe for two ranks, and invokes the runtime's actual expert-routing helper
on CPU scheduler timesteps. This supplements earlier whole-cache digest
verification; it does not repeat the full digest audit or demonstrate GPU fit.

## Host-memory budget

Each expert has 14,288,491,584 stored FP32 elements across 12 shards. At the
recipe's native BF16 runtime dtype each needs 28,576,983,168 weight bytes.
The text encoder contributes 11,361,820,672 BF16 bytes and the FP32 VAE
507,570,124 bytes. LoRA and runtime state are additional.

For N symmetric ranks, estimate replicated rollout pipelines plus globally
sharded replay experts plus per-rank replay conditioners. Nominal weight
storage alone is:

| Ranks | Nominal steady weight bytes | Approx. GiB |
| --- | ---: | ---: |
| 2 | 218,939,462,192 | 203.9 |
| 3 | 299,832,210,120 | 279.2 |
| 4 | 380,724,958,048 | 354.6 |

This is not measured RSS: shared mappings may reduce physical usage, whereas
conversion copies, activations, trajectories, optimizer slots, Ray and allocator
caches increase it. The report also records a hypothetical fully unsharded
trainer-plus-rollout inventory; those components need not all coexist during
the actual staged loader lifecycle. Do not treat that estimate as a measured
loading peak or an OOM verdict. The host reports about 372.7 GiB total RAM.

Use the already planned two-rank proof first. Four symmetric workers would
leave little nominal headroom even before activations; four available GPUs
do not imply that four host-replicated pipelines are the appropriate topology.
Actual per-stage host/GPU peaks still need to be measured during loading,
expert switching, training, export and resume.

## Expert coverage and scope

The existing lifecycle proof remains 320x320, 17 frames, ten denoise steps,
two samples per local prompt, CPU OCR objective, both experts trainable,
FSDP CPU offload, sequential rollout offload and compile disabled. It is an
explicitly separate lifecycle proof, not a reduced replacement for the full
480x832/81-frame physics experiment.

Native scheduler timesteps are 999, 964, 923, 875, 818, 750, 667, 563, 429, 251.
The real boundary is 875; equality routes to the high-noise expert. The first
four rollout steps use `transformer`, the remaining six `transformer_2`.
The recipe's first nine replay indices also include both experts. These CPU
routes establish schedule coverage, not successful expert GPU execution.

The current CPU distributed test
`tests/trainers/test_wan_fsdp_distributed.py::test_wan_dual_expert_fsdp_stage_isolation_sync_and_resume`
passes: **1 passed in 17.40 s**. Two gloo ranks use tiny real Wan modules and
check nonzero gradients in each expert, stage-isolated updates, both rollout
sync prefixes and tensor-exact serialized restoration of weights and optimizer
slots. It does not test released weights, CUDA offload or a continued-update
comparison against an uninterrupted baseline.

The GPU gate must still reject an OCR-zero/zero-gradient run even if its step
counter advances. Require both experts' real finite nonzero updates, lifecycle
memory evidence, rollout/replay agreement and exact controlled resume. Do not
claim success from this preflight or restart the disabled SD3 long queue.

## Bounded CUDA prerequisite verified

The same clean candidate now passes the existing tiny-real CUDA/NCCL test on
this L40S host at world sizes one and two: **2 passed in 20.39 s**, exit 0.
Test: `test_wan_dual_expert_fsdp_cuda_cpu_offload` in the same distributed
test module, invoked with `--distributed`, `CUDA_VISIBLE_DEVICES=0,1` and
`OMP_NUM_THREADS=4`. No Ray process or released-weight pipeline was involved.

Assertions cover nonzero CUDA gradients and changed weights for both experts,
CPU-resident local parameter shards after execution, exportable optimizer
state, CPU rollout-state export and exact buffer-device restoration across
training-state parking. Six lifecycle log events cover the one-rank high/low
pair and both ranks' high/low pairs in the two-rank case. Every event reports
zero inactive CUDA parameter bytes. These counters are captured after forward
resharding, not evidence that an active expert used no GPU during its forward.
The trace's largest forward peak counter is 17,136,128 bytes; tiny-model
counters do not estimate released 14B activation peaks or production throughput.

Log: `/mnt/nvme/outputs/wan22_i2v_cache/dual_expert_cuda_l40s_distributed.log`.
The first invocation omitted the repository's required distributed opt-in and
was skipped, not passed. Its separate `dual_expert_cuda_l40s.log` is retained.
Both pytest sessions are terminal and fresh GPU compute inventory is empty;
GPUs 0-1 are released. Full released-weight GPU update and controlled resume
remain open; neither CPU nor tiny CUDA prerequisites close those gates.

## Released-weight diagnostic, 2026-09-13

Runtime was candidate `7bf2b57907b3e34f850d1b29181d6a1774e31571` in
`/home/ubuntu/VRL-cosmos-cp`. Two ranks used both pinned full 14B T2V experts,
320x320, 17 frames, ten denoise steps, four global samples per update and
rank-32 LoRA. Real Kling reward replaced OCR; two VideoPhy prompts had fixed
request seeds. FSDP CPU offload, sequential rollout offload, IEEE precision,
full_cpu checkpointing and EMA every update were enabled; compile was disabled.
This is not the full-size I2V physics recipe or an OCR comparison.

All artifacts are under `/mnt/nvme/outputs/wan22_i2v_cache/`:

- `wan22_native_dual_local_hash_attempt/`: intentionally stopped local-path
  launch; identity hashing consumed about 192.5 seconds before model building.
- `wan22_native_dual_kling_pinned*`: pinned Hub identity avoided full-tree
  hashing, but offline shard discovery failed without `local_files_only`.
- `wan22_native_dual_kling_offline*`: explicit `model.local_files_only=true`
  reached the real update. Preflight verified the pinned revision and flag in
  both rollout and replay model-build arguments.
- `wan22_native_dual_kling_offline/first_update_diagnostic.json`: authoritative
  completed-update audit, not a two-update success verdict.

The first native update took about 406.08 seconds, including cold reward work.
Four clips decoded to 17 frames and received nonconstant real rewards. Both
experts updated: each has 640 trainable tensors, 320 nonzero Adam first moments
and 320 nonzero LoRA B tensors. All 1,280 optimizer states and FP32 masters were
finite; every master-to-BF16 projection exactly matched the checkpoint model.
EMA count was one. This is checkpoint integrity, not controlled resume.

However, pre-update maximum log-probability difference was
`0.0009684562683105469`, with clip fraction `0.4722222222222222` and active clip
fraction `0.19444444444444445`. Passing the existing 0.01 guard does not establish
strict equivalence when the algorithm's clip ratio is 0.0001. The second update
was deliberately prevented; second-generation artifacts exist, but there is
only one completed update and no successful final checkpoint.

Sampled NVML usage reached 10,223,616,000 bytes per participating GPU; minimum
sampled available host RAM was 143,991,476,224 bytes. Monitoring began after
early model loading, so these are not whole-lifecycle peak guarantees. Expert
traces reported zero inactive CUDA parameter bytes after forward resharding.
Controllers and monitor terminated; fresh GPU and Ray process inventories were
empty after stopping. No performance or quality improvement is claimed.

## Frozen precision admission fix

Actor precision normalized 125 FP32 frozen tensors per expert to BF16. This is
a plausible replay mismatch contributor, not a proven sole cause; generation
batch two versus replay batch one remains another controlled-test variable.

Candidate commit `4c527cb1` preserves frozen floating dtypes under native FSDP
`precision_policy=none`, derives validation dtype from trainable parameters,
and retains rejection of mixed trainable or nonfloating parameter dtypes.
Existing actor-policy casting is unchanged. No default policy was flipped.

Verification: 207 CPU tests passed, 13 skipped; four CUDA/NCCL cases passed
in 34.17 seconds across one and two ranks, including mixed BF16 trainables and
frozen FP32 dual experts. Tiny-model native forward outputs exactly matched
unsharded references; both experts updated and frozen FP32 values stayed exact.
An initial test-only CPU DTensor gather failure was corrected by staging the
gather copy to CUDA; its original log is retained.

Full-model cached-trajectory replay and gradient comparisons are still required
before resuming training. Native policy also changes gradient reduction dtype,
so tiny forward equality alone cannot close single-card training semantics.
Independent all-step replay, controlled resume, full-size I2V and quality gates
remain open. The stopped SD3 queue must not be restarted.

## Replay fixture capture exposed a separate reward-device defect

On 2026-09-13, a standalone native collector was prepared to save one two-sample
group and initial dual-expert LoRA for controlled replay. CPU preflight passed
with trainer device 0, dedicated rollout device 1 and reward sharing device 1.
The first attempt, `wan22_replay_fixture_real`, exited before generation because
the script omitted the schedule-owned `runtime.activate()` call. Its source
and log are retained. This was a harness admission error, not an OOM.

The corrected `wan22_replay_fixture_active` attempt generated both real videos
in 69.035 seconds and entered real Kling scoring, but reward parking failed.
Both attempts ran candidate `4c527cb1`, exited 1 and cleaned up their owned Ray
sessions. Neither produced `rollout_batches.pt`; initial weights alone are not
a usable replay fixture. `wan22_replay_fixture_audit.py` is prepared but has
not run successfully because its required trajectory artifact is absent.

A two-GPU small-tensor regression reproduced the problem: with current CUDA
device 0 and configured reward device 1, CuMem captured no model allocations.
Binding only construction captured the allocations but sleep then failed with
a CUDA invalid-argument error. Pool construction, sleep, wake and terminal
release all require the configured device context.

Candidate `96bcac9c` adds that scoped context to the in-process reward lifecycle,
restoring the caller's current device after each operation. No parking threshold
or dependency was changed. The real two-GPU regression now passes (4.76 seconds),
including eager and lazy model allocations, physical parking validation under
the existing standard CUDA residual allowance, exact values after wake and
terminal cleanup. CPU reward regression: 413 passed, 8 skipped in 8.56 seconds.
Ruff and diff checks passed. The CPU residual-boundary mock now also mocks the
CUDA device context; production CUDA admission was not weakened for that test.

All GPU and Ray process inventories were empty after testing. Full Kling
parking must still be rechecked on the corrected candidate while completing
the fixture capture; the Wan replay/gradient matrix remains pending. This
reward lifecycle defect is separate from the previously observed replay drift.

## Real capture and controlled replay matrix

The unchanged capture harness completed on candidate `96bcac9c` in
`wan22_replay_fixture_devicefix`. Both real clips were scored; CuMem backed up
and released 4.69 GiB on reward device 1 while the driver's current device was
0. Collection wall was 107.186 seconds: generation 69.588, reward 37.597,
overlap zero. Rewards were -1.412971019744873 and -0.5611866116523743. These are
fixture values, not an improvement over different sampled videos. The process
exited zero including owned Ray shutdown, and GPU/Ray inventories were empty.

`fixture_audit.json` verifies 20 finite transitions, FP32 observations/actions
of shape `[2,10,16,5,40,40]`, BF16 conditioning, initial LoRA with 640 tensors
per expert, and artifact hashes:

- Initial weights (419,904,183 bytes):
  `794d1df1b9c639e26effebefb84d8d83c668fe3a16ad4bffc4e46cb822890938`.
- Trajectory (28,876,490 bytes):
  `d245e36c3b1bcb5262193be3127bc733dc58c00aaaff4eb1960db78f5200e555`.

The first replay attempt (`wan22_fixed_replay_actor`) failed before forward:
the replay-only constructor had not initialized the inherited pipeline-offload
state used by `load_trainable_state`. Candidate `52a7cf44` initializes this
state to `None`; T2V/I2V single/dual-expert state-load regressions were added.
Wan-family and inheritance tests: 86 passed in 4.20 seconds. No fake pipeline
or offload hook is introduced into the replay-only model.

Both successful matrix arms ran clean `52a7cf44`, one-rank native FSDP with
CPU offload, IEEE math, the same initial LoRA, observations, actions, scheduler
and conditioning. Native `DiffusionSDELogProbEvaluator` replayed all ten steps
for both samples at batch two and then batch one. No gradients or optimizer
updates were performed. Generation batch size was two.

| FSDP precision | Replay batch | Maximum absolute log-prob difference | Clipped transitions |
| --- | ---: | ---: | ---: |
| actor | 2 | 0.0008442997932434082 | 11/20 |
| actor | 1 | 0.0006373822689056396 | 10/20 |
| none | 2 | 0 | 0/20 |
| none | 1 | 0.0003195483877789229 | 9/20 |

Clipping uses the unchanged recipe ratio 0.0001. Actor precision removes the
250 frozen FP32 tensors (58,368,000 elements) across the two experts; native
precision preserves their inventory. Native batch-two log probabilities are
tensor-exact to the saved generation values at every step, across both experts.
Changing only replay batch shape still causes meaningful drift under this
narrow clip ratio. Therefore preserving frozen precision alone is insufficient
for the original generation-two/replay-one configuration. The matched-batch
result is a bounded real replay pass, not proof of general batch invariance.

Outputs: `wan22_fixed_replay_actor_loaded/` and `wan22_fixed_replay_none/`, each
with `result.json`, `transitions.json`, `logprobs.pt` and executed script copies.
`wan22_fixed_replay_compare.py` rehashes the fixture, checks exact coverage of
all 20 transitions per arm, compares saved old log probabilities to the actual
fixture and recomputes differences/clipping from tensors. Its audited summary
is `wan22_fixed_replay_comparison.json`. All scripts/artifacts are beneath the
NVMe evidence directory used above. Both runtime sessions and the comparison
audit exited zero; fresh GPU compute inventory was empty after completion.

Next gate: preserve frozen FP32 and match generation/replay batch shape while
comparing actual gradients, accumulation and updates across one and multiple
ranks. Native policy also changes gradient reduction dtype; do not infer its
gradient semantics from forward equality. No public default, clipping threshold
or full-size I2V recipe was changed; training remains stopped pending that gate.

## Four-sample gradient and accumulation control

On clean candidate `52a7cf44`, `wan22_four_sample_fixture` captured four distinct
real samples in one prompt group, using two generation batches of two. The
initial LoRA hash is unchanged from the earlier two-sample fixture. The saved
40-transition trajectory hash is
`f441099acd2af7f976710547eb2f9b61ea01ab575731996cfc991dd16f7a081c`.
The audit checks finite values and distinct initial latents, not just row count.
Collection wall was 180.918 seconds (generation 138.164, reward 42.754), with
successful real Kling parking and owned Ray shutdown, exit zero.

`wan22_fixed_gradient_probe.py` ran one-rank FSDP accumulation and two-rank
FSDP on this same fixture. Both used native frozen FP32 preservation, replay
batch two, full_cpu activation checkpointing, the same four-sample advantages
and all nine replay steps from the bounded recipe. The single rank averaged
two two-row chunks per step; two ranks each handled one of those same chunks,
with FSDP rank averaging. Each arm covered exactly 36 sample/step pairs and
every pre-update log probability was tensor-exact to generation. All 1,280
trainable gradients were saved before optimizer preparation or clipping.

The initial configuration attempt exited before loading because the harness
used `actor.microbatch_size=2`, which counts prompts, not replay samples.
The correct public knob is `actor.samples_per_replay_batch=2`. The successful
single output is `wan22_fixed_gradient_single_matched`; the two-rank output is
`wan22_fixed_gradient_two`. Both completed and exited zero.

An important harness limitation was discovered while auditing optimizer output:
the executed probe called `build_optimizer` directly instead of the online
trainer's `_ensure_optimizer`, omitting FP32 master weights. Its `update.pt`
files are therefore **plain low-precision Adam diagnostics, not native online
optimizer acceptance**. This does not affect the captured pre-clip gradients.
Executed source copies are preserved in each output. The adjacent probe has
since been corrected to call `_ensure_optimizer`, but that corrected GPU
optimizer path has not yet been rerun.

To inspect the consequences without repeating model backward,
`wan22_cached_master_update.py` rebuilt the initial LoRA on CPU, attached each
arm's saved full gradients, and called the native `_ensure_optimizer` and
`_clip_and_step`. It asserted the FP32 wrapper, finite gradients and positive
norm below the clipping limit. The resulting `master_update.pt` files include
FP32 masters, Adam slots, exact parameter order and updated BF16 projections.
This is CPU optimizer-only replay, not distributed optimizer lifecycle proof.

`wan22_fixed_gradient_compare.py` verifies sample/step coverage, zero replay
errors, identical advantages and optimizer hyperparameters, finite tensor
state, parameter ordering, first-step counters and exact master-to-BF16
projections. Its fixed thresholds were declared before comparison; it exits
two and writes `wan22_fixed_gradient_comparison.json` with `not_accepted`:

| Compared quantity | Relative L2 difference | Fixed limit |
| --- | ---: | ---: |
| Pre-clip gradients | 0.0040335740695426345 | 0.0001 |
| BF16 model update from CPU master replay | 0.007979809568899933 | 0.0001 |
| FP32 master update | 0.007895163367033808 | 0.0001 |
| Adam first moment | 0.004033572380764671 | 0.0001 |
| Adam second moment | 0.007646695396405864 | 0.0002 |

Maximum gradient difference is 1.52587890625e-05; maximum master difference is
9.521072206553072e-05. Thus matching forward precision and batch shape does not
establish single/multi-rank training semantics. Native BF16 reduction and
different accumulation order are candidates to isolate, not yet a proven sole
cause. Do not relax thresholds or infer learning quality from this diagnostic.

The measured gradient loops took about 247.041 seconds (one rank) and 144.280
seconds (two ranks), excluding setup; these are component timings, not a fair
end-to-end training speedup claim. The optimizer-wrapper omission also prevents
treating the complete probe times as native-update performance acceptance.
All GPU/runtime and CPU audit sessions are terminal, and fresh GPU/Ray process
inventories are empty. Next isolate per-microbatch gradient reduction versus
cross-microbatch accumulation, then rerun the correctly wrapped distributed
optimizer and controlled resume only after the gradient gate passes.

## Exact reconstruction of the BF16 discrepancy

On the same clean candidate `52a7cf44`, `wan22_gradient_contributions.py`
replayed the unchanged four-sample fixture with one-rank FSDP, frozen FP32
preservation, two-row chunks and full_cpu checkpointing. It cleared gradients
between each backward and saved all 18 independent contributions: nine steps
times two chunks, each with 640 finite BF16 trainable gradients for the active
expert. Loss divisor remained 18, exactly as in the previous single-rank arm.
Every pre-update log probability remained exact. No optimizer was constructed
or stepped. The capture finished in 254.645 seconds excluding setup and exited
zero, with an empty GPU compute inventory afterward.

`wan22_gradient_rounding_audit.py` checked all step/chunk coverage and then
reconstructed four sums on CPU from those actual captured tensors:

- BF16 sequential accumulation in the previous single-rank order.
- BF16 pairwise rank averaging followed by accumulation in the previous
  two-rank order. Contributions are doubled for the two-rank loss divisor nine,
  summed and halved to model rank averaging.
- The same sequential and pairwise orders with FP32 additions.

Results in `wan22_gradient_contributions/rounding_audit.json`:

| Comparison | Relative L2 | Maximum absolute | Exact tensors |
| --- | ---: | ---: | ---: |
| Reconstructed BF16 sequential vs actual single rank | 0 | 0 | 1280/1280 |
| Reconstructed BF16 paired vs actual two ranks | 0 | 0 | 1280/1280 |
| Actual single vs two ranks | 0.0040335740695426345 | 1.52587890625e-05 | 640/1280 |
| FP32 sequential vs paired simulation | 1.0627227236804594e-10 | 1.8189894035458565e-12 | 688/1280 |

Thus BF16 summation order is sufficient to reproduce the entire observed
cross-rank gradient discrepancy in this fixed experiment. This is stronger than
attributing it from dtype inspection or similar error magnitudes. It does not
establish a reward/quality regression or a different mathematical GRPO objective.

The installed Torch 2.12 FSDP implementation also shows the relevant boundary:
`_fully_shard/_fsdp_collectives.py` converts reduced gradients back to
`orig_dtype` before adding to existing sharded gradients (lines 687 and 722 in
this environment). Therefore changing reduction dtype alone does not guarantee
FP32 cross-microbatch accumulation for BF16 trainables. The FP32 master optimizer
currently receives source gradients after that accumulation, too late to undo
the earlier rounding.

The FP32 arithmetic simulation is **not a distributed implementation test**.
Next evaluate a supported higher-precision trainable/gradient path on both
single and multiple ranks, while retaining exact forward replay and verifying
the actual native optimizer wrapper. Such a change intentionally changes the
old single-rank BF16 numerical result as well; compare like-configured arms and
do not relabel it as bitwise preservation of the old BF16 baseline. Keep public
defaults unchanged until the runtime, updated rollout delivery and controlled
resume have passed their own gates.

Both capture and CPU audit sessions are terminal, executed scripts are copied
into the NVMe output directory, and GPU0 is released. No shared runtime code,
model weights, optimizer defaults or acceptance thresholds changed this turn.

## Real FP32-LoRA distributed gradient and update pass

The next controlled experiment ran on unchanged clean candidate `52a7cf44`,
using `wan22_fp32_lora_gradient_probe.py`. After loading and verifying the same
initial LoRA, the diagnostic explicitly cast only the 1,280 trainable tensors
to FP32. Frozen parameter dtypes and storage pointers were asserted unchanged.
The upcast initial values were readback-verified. This is a post-build diagnostic
override recorded in the report, **not a public model-build option** or an
implicit change to canonical model identity/defaults.

Both one and two ranks used this same new numerical baseline: FSDP `none`,
CPU parameter offload, IEEE math, replay batch two, full_cpu checkpointing,
four fixed real samples, identical advantages and all nine bounded-recipe
steps. One rank accumulated two chunks per step; two ranks each handled one
of the same chunks. Both arms covered all 36 sample/step pairs with exact
pre-update log probabilities. Native `OnlineTrainer._ensure_optimizer` and
`_clip_and_step` performed the actual GPU-run update. Because trainable source
parameters are already FP32, the correct native optimizer is ordinary AdamW
with FP32 moments; there is no separate low-precision/master binding to omit.

Both expert updates completed, with 640 changed trainable tensors across the
two roots. Outputs `wan22_fp32_lora_single/` and `wan22_fp32_lora_two/` contain
`raw_gradients.pt`, `update.pt`, per-rank transition receipts, result metadata
and executed scripts. Both torchrun sessions exited zero before CPU comparison.

`wan22_fp32_lora_compare.py` verifies identical advantages and optimizer
hyperparameters, exact global sample/step coverage, zero replay errors, all
1,280 FP32 trainable tensors, FQN-keyed Adam slots, finite state and step-one
counters. It recomputes differences from saved tensors and passes the same
unchanged thresholds used in the failed BF16 experiment:

| Quantity | Relative L2 difference | Maximum absolute difference |
| --- | ---: | ---: |
| Pre-clip gradients | 1.0627227236804594e-10 | 1.8189894035458565e-12 |
| Parameter update | 5.665924813117295e-10 | 6.730260793119669e-11 |
| Adam first moment | 1.1011508792136078e-10 | 2.2737367544323206e-13 |
| Adam second moment | 4.38071304300408e-12 | 1.0842021724855044e-19 |

Summary: `wan22_fp32_lora_comparison.json`, status `passed`, audit exit zero.
The gradient difference also equals the independently simulated FP32 addition
order difference from the previous contribution audit. This closes the bounded
real gradient/first-update comparison for this explicit FP32-LoRA experiment,
not bitwise equality or preservation of the old single-rank BF16 rounding.

Gradient norms were 0.033943140142922275 and 0.033943140142920436. Measured
gradient-loop times were 257.582 and 151.220 seconds; times through state export
were 264.171 and 159.267 seconds. These exclude model setup, generation and
reward and are not end-to-end throughput or quality acceptance.

All GPU compute inventory was empty after both runs, and GPUs0-1 are released.
Next integrate an explicit family-owned precision option with checkpoint
identity and strict dtype compatibility, then verify updated FP32 LoRA delivery
to actual rollout workers and controlled checkpoint continuation. Keep existing
defaults unchanged; the post-build diagnostic alone does not enable the native
online recipe to construct this dtype consistently across trainer and workers.

## 2026-09-13: public opt-in FP32 LoRA storage

Candidate `6ab984cf` adds `model.lora_parameter_dtype: float32` to the shared
Wan T2V/I2V family schema and LoRA construction path. The unset/null default
retains prior construction behavior and checkpoint identity. Explicit FP32
storage contributes to checkpoint identity; it requires `use_lora=true` and,
when FSDP is selected, explicit `distributed.training.fsdp.precision_policy=none`.
Only trainable adapter parameters are converted after fresh creation or warm
loading. Unsupported storage values fail before adapter mutation.

CPU verification: 437 tests passed in 30.20s across the Wan family, checkpoint
identity and configuration suites; Ruff passed for all six changed files.
New tests exercise actual PEFT fresh creation and saved-adapter reload in both
default BF16 and opt-in FP32 storage, unchanged frozen parameter values/dtype/
data pointers, invalid-value rejection, schema/FSDP policy rules and default
identity compatibility for both Wan families. These tests do not establish
actual distributed rollout delivery or checkpoint continuation.

No GPU run or long training queue was started for this configuration change.
Next use the public option, without diagnostic post-build casting, to verify
updated FP32 adapters reaching actual generation workers, matched replay and
controlled checkpoint continuation. Full-size I2V, quality and end-to-end
throughput remain open.

## 2026-09-13: updated public FP32 adapter delivery and replay

The public build option was exercised with the actual saved two-rank FP32
update, not another initial adapter. Source `wan22_fp32_lora_two/update.pt`
SHA256 is `baf7196bb758def031a46d64166e87b41afdf0965b54b59a8050f121918a849f`.
Both trainer-side replay construction and actual Ray generation construction
used the public FP32 option, with no post-build diagnostic parameter casting.

First attempt `wan22_updated_fp32_rollout` exited 1 before generation: exact
readback rejected sequential-offload meta parameters. The existing weight
load had returned, but that did not prove installed content. Candidate
`21ae2051` adds Wan readback through the existing hook-suspension transaction:
materialize on CPU, compare actual parameter bytes, restore offload hooks.
Mismatch remains fail-closed. All 49 Wan tests passed in 3.97s, including actual
Accelerate/PEFT hook restoration and wrong-payload rejection; Ruff passed.

Retry `wan22_updated_fp32_rollout_readback` exited 0. All 1280 FP32 adapter
tensors matched the sender on the actual worker before generation, version 1.
Two 320x320/17f clips completed all ten steps; Kling scores were
`[-1.1431514024734497, -0.6394544839859009]`. Collection took 111.773s,
generation 74.467s and reward 37.291s, with zero measured overlap. These are
phase observations, not throughput or learning-improvement acceptance.
Trajectory audit checked 20 finite transitions, two distinct initial latents
and version-1 metadata. The legacy `initial_trainable_state.pt` filename and
fixture auditor's initial-LoRA scope string mean the starting state for this
capture; its tensors are the updated policy, not the original initial policy.

Native one-rank FSDP replay (`precision_policy=none`) then completed in
`wan22_updated_fp32_replay`, exit 0. Frozen inventories remained 250 FP32 and
1940 BF16 tensors; all 1280 adapter tensors remained FP32. Both branches
covered every one of the 20 saved sample/step pairs:

| Replay batch | Maximum absolute log-prob error | Clipped at ratio 1e-4 |
| --- | ---: | ---: |
| 2, matching generation | **0 exactly** | **0/20** |
| 1, diagnostic shape change | 0.00039689987897872925 | 8/20 |

Standalone `wan22_updated_fp32_audit.py` exited 0. It hashes the source update,
checks every exported parameter against it, proves 320 changed tensors per
expert versus the original initial adapter, compares replay tensors directly
with the saved old log-probs, validates full coverage, identity, inventories
and policy version. Receipt: `wan22_updated_fp32_replay/acceptance_audit.json`.
Executed capture and replay scripts are retained beside their outputs.

This closes public updated-weight delivery to one real worker and matched-
batch updated replay, not arbitrary batch-size invariance, checkpoint resume,
full-size I2V, end-to-end scaling or quality. All compute processes exited;
fresh GPU inventory empty and claims released. Next controlled checkpoint
continuation should retain matching batch geometry and the public FP32 option.
