# Cosmos real conditioning and complete video sequence on L40S

Status: complete-sequence numerical and artifact verification passed; visual
quality and integration into multi-GPU online training remain open.

## Scope and code

Candidate `/home/ubuntu/VRL-cosmos-cp` commits:

- `fbf5af15`: optional artifact directory retaining real conditioning, complete
  decoded media and result JSON; explicit finite nonnegative replay tolerances;
  fail closed on nonfinite per-step latents/log-probs or replay errors.
- `435c8fa2`: optional existing model preset consumed through the production
  model-build resolver. CLI family/path remain authoritative and mismatched
  preset families fail before building.

The first attempt failed before weight loading because the generic probe
disabled LoRA while Cosmos Predict2.5 is LoRA-only. The existing
`model/cosmos/predict2_5_2b.yaml` supplies native rank32/alpha64 default/previous
adapters. No family requirement was bypassed and no arbitrary new adapter
configuration was substituted. The failed attempt's output directory was
retained separately; it is not a passing result.

## Real execution

Two independent production-registry rollout builds ran sequentially on physical
GPU2 (logical cuda:0 under `CUDA_VISIBLE_DEVICES=2`). Both use the pinned
`nvidia/Cosmos-Predict2.5-2B` revision
`0d37c7498f54cee3c599d438d895a0a4a8608064`, including real Qwen text encoder,
tokenizer, transformer and VAE. Encoder offloads to CPU after encoding;
native VAE tiling/slicing is enabled. Base precision is BF16 with IEEE FP32
and outer autocast; LoRA follows the native family setup. No optimizer update
occurs in this generation-only probe.

Geometry is the existing512p_93f preset's **512x512,93 frames**, with20 CPS
steps, no-CFG (guidance1), seed42 and16fps. This is one clip, not the paper's
32conditions x8samples x256updates training budget.

Prompt: "A wooden block slides across a smooth table, slows down, and comes
to a stop. The camera remains still."

Both executions completed with exit0:

| Result | Root-cache run | Complete NVMe-cache repeat |
| --- | --- | --- |
| Seconds including load and generation | 986.3690 | 88.1280 |
| Peak allocated bytes | 22,213,337,088 | 22,213,337,088 |
| Decoded tensor | `[1,3,93,512,512]`, finite | same |
| Decoded std | 0.1168493107 | same |
| Replay noise max abs | 0 | 0 |
| Replay log-prob max abs | 0 | 0 |

Every denoise step's latents/log-probs was finite. Both explicit replay
tolerances were1e-3, not the generic probe's looser default. This replay
check is an unsharded same-model step0 check, not a new CP parity result.

Actual MP4 decoding yielded93 RGB frames at512x512 and16fps, duration5.81s.
Mean adjacent-frame absolute pixel difference is4.2958574 on uint8 scale.
Both MP4 files have identical SHA-256:

`7f41e37dd66caefd90dd29bed27d0b324590c59aa70f1599b43d6e219c659e6c`

Saved real conditioning trees compare tensor-exactly between runs; prompt
embeddings have shape`[1,512,100352]`, BF16 dtype and all finite values.
The encoded trees are retained for later real-conditioning CP integration.

Visual inspection of six extracted frames shows a blurry block-like object
and changing framing. It does not establish the requested slide/deceleration
or a stationary camera. Nonblank, finite, moving output is **not** semantic
or physics-quality acceptance; no reward improvement is claimed.

## Loading bottleneck and remediation

The root-cache run spent about15minutes loading/moving weights. Live process
stacks showed `Module.to` first for the text encoder, then the DiT; Linux
reported `folio_wait_bit_common`. An iostat interval measured root reads at
11,776KiB/s,82.72ms read latency and95.8% utilization. Root disk was98% full.
These observations identify loading IO wait, not GPU computation or Ray
queueing, as the bottleneck in that phase.

Copied text_encoder/tokenizer/vae from that exact snapshot to the existing
NVMe snapshot containing the pinned transformer/scheduler/model_index.
The copy exited0; recursive byte comparisons for all three new components
also exited0. The original cache was not deleted or modified. The active
root-cache job continued using its original dependencies throughout.

The fresh repeat uses the complete NVMe snapshot and reaches native LoRA
setup about14seconds after the builder starts. The88.1s total is an observed
subsequent-run result. Cache warming and concurrent copying during the first
run prevent treating986.4/88.1 as an isolated disk benchmark, and this is
neither a GPU compute speedup nor a multi-GPU throughput comparison.

## Artifacts and reproduction

All output paths below are under`/mnt/nvme/outputs/wan22_i2v_cache/`:

- `cosmos_real_sequence_512p93f_l40s`: initial pre-load LoRA rejection only.
- `cosmos_real_sequence_512p93f_native_lora_l40s`: successful root-cache run,
  conditioning.pt, video.mp4, result.json and contact_sheet.png.
- `cosmos_real_sequence_512p93f_nvme_l40s`: successful NVMe repeat,
  conditioning.pt, video.mp4 and result.json.

From the candidate worktree:

```bash
env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-cosmos-cp \
  /home/ubuntu/VRL/.venv/bin/python -m vrl.scripts.generation.full_sequence_denoise_probe \
  --family cosmos-predict2.5 \
  --model-preset vrl/config/presets/model/cosmos/predict2_5_2b.yaml \
  --path /mnt/nvme/hf/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/0d37c7498f54cee3c599d438d895a0a4a8608064 \
  --prompt 'A wooden block slides across a smooth table, slows down, and comes to a stop. The camera remains still.' \
  --steps 20 --height 512 --width 512 --frames 93 --guidance-scale 1 --seed 42 \
  --dtype bf16 --float32-precision ieee --outer-autocast --device cuda:0 \
  --sde-type cps --offload --check-replay \
  --replay-noise-atol 0.001 --replay-logprob-atol 0.001 \
  --artifact-dir /mnt/nvme/outputs/wan22_i2v_cache/NEW_EMPTY_OUTPUT
```

CPU probe tests:13 passed in0.32s; Ruff and whitespace checks pass. Full
artifact writing and MP4 decoding exercised by both real GPU executions.
All model/copy/comparison processes terminal; fresh GPU compute inventory
empty afterwards. GPU2 released; candidate and frozen integration worktrees
clean. Shared Python dependencies and frozen runtime unchanged.

Remaining: real-conditioned CP updates/full trajectory integration, actual
reward scoring, production construction and sample ownership, original SFT/
compile/numerical gates, quality and paper-shaped workload. A single generated
clip fitting on one L40S is not a multi-GPU-only capacity claim.

## Native real reward scoring

The retained NVMe-run MP4 was decoded with decord into a real
`RewardSample` (`[3,93,512,512]`,16fps, actual generation prompt), then scored
twice through production `KlingVideoReward.score_batch`. This exercises
temporary MP4 materialization, the real in-process inference runtime, result
selection, cleanup and shutdown. It uses the decoded exported video, not the
original pre-encoding float VAE tensor; this introduces the usual additional
codec round trip compared with directly scoring fresh generation tensors.

Physical GPU3 was dedicated to this bounded scoring job. Worker settings
match the existing Cosmos reward preset: BF16, normalized `overall_reward`,
min_frame_pixels200704, local-only files. VideoReward is explicitly pinned
to4f26600130683e6f1de9f5d463887f28e8ef995c; its Qwen2-VL-2B-Instruct base
resolves local main to895c3a49bc3fa70a340399125c650a463535e71c. Both caches
are on NVMe; no shared package or model weights were modified.

Both calls and all four debug scores agree exactly:

| Score | Value |
| --- | ---: |
| visual_quality | -1.5804257179880763 |
| motion_quality | 0.0052675403663745715 |
| text_alignment | -1.921151626129533 |
| overall_reward | -3.4963098037512346 |

First score_batch took38.2534s including lazy model initialization. The warm
call took0.94855s: artifact materialization631.92ms, inference315.67ms.
Peak allocated memory was5,103,843,840bytes (about4.75GiB). Temporary MP4s
were absent after each call, shutdown completed, and the process exited0.
Fresh compute inventory was empty afterwards; GPU3 released.

The initial Qwen base load emitted old/new Transformers key-layout warnings.
The selected VideoReward checkpoint contains full `model.pth`, not an
adapter-only checkpoint. The actual loader remaps the complete state against
live keys and calls `load_state_dict(strict=True)` before inference, so this
run did not retain the initially missing base parameters. Relevant CPU
loading tests under the same Transformers5.13 overlay:12 passed,1 optional
test skipped in2.74s. The live-model strict old/new key-layout test was also
run explicitly and passed in2.54s. This full-checkpoint conclusion must not
be generalized to unverified adapter-only reward checkpoints.

Artifacts under `cosmos_real_reward_l40s`: result.json, two distinct request
and result debug records, and `executed_probe.py`, the exact executed script.
The reusable sibling `cosmos_real_reward_probe.py` was subsequently corrected
to name its preflight timer `preflight_seconds`. The original result's
`load_seconds` field (~37microseconds) actually measures preflight, NOT model
loading; retain that original artifact with this interpretation. The first
score_batch timing above includes real lazy loading. No second job was needed
for this label-only correction.

This establishes the real clip's native scoring path and repeatability, not
a quality threshold, reward discrimination across different samples, learned
improvement, HTTP/Ray delivery or integration with a CP optimizer update.
The remaining whole-training and paper-budget requirements stay open.

## Native collector launch preflight

The next bounded production collector run now has a resolved launch contract,
not an inferred topology. Preflight uses `load_config`, `resolve_online_run`,
`resolve_model`, `ResolvedOnlineRun.ray_launch_inputs` and
`GlobalRayPlacementOwner` directly from candidate `435c8fa2`.

The existing GRPO experiment is overridden only for one prompt group, eager
execution, the verified NVMe checkpoint, explicit placement and controlled
precision/storage. Geometry remains 512x512/93f, 20 CPS steps, guidance 1,
noise 0.7, eight samples per prompt, one sample per generation batch. Native
LoRA is rank 32/alpha 64 with the preset's six target patterns.

Resolution proves trainer reservation GPU 0, rollout GPU 2 and in-process
reward GPU 3, with no shared-device lifecycle. GPU 1 is unused by this
collector-only stage; this is preparation for CP replay, not CP execution.
The planned driver initial replay model is on CPU and its trainable state is
to be pushed through the native Ray syncer before collection.

The first preflight exposed inherited TF32 and preserve-device trajectory
storage. The corrected configuration fixes IEEE and CPU trajectory storage
without dtype conversion. The second preflight passed with those settings.
Checkpoint identity covered 20 files / 21,226,364,527 bytes, SHA256
`4fec540c29ab67ad37ad12262bdac706e6849c41912590893b71ad845a7d2010`.
The reward checkpoint is pinned to
`KlingTeam/VideoReward@4f26600130683e6f1de9f5d463887f28e8ef995c`.

Evidence: `cosmos_native_collector_preflight_ieee_cpu/{preflight.json,
resolved_config.yaml,executed_probe.py}` under the existing NVMe output root.
The first preflight is retained in `cosmos_native_collector_preflight`.
Reusable script: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_native_collector_probe.py`.
Preflight is the default; `--run` additionally requests real materialization,
private Ray launch, weight sync, generation, reward and typed-batch saving.
That execution branch has NOT yet been run or validated. No actors, model
weights, reward scoring or trajectories were produced by these preflights.
Both processes exited 0 and a fresh GPU compute inventory was empty.

## Real native Ray collector: one complete eight-sample group

The preflighted script's `--run` branch completed against candidate `435c8fa2`.
CPU replay materialization created the native rank-32/alpha-64 initial adapter;
the production Ray syncer pushed policy version 1 before collection. A private
Ray cluster probed physical bundle ownership `(0,2,3)` and loaded the real
Cosmos rollout worker on GPU 2. Real Kling reward ran on GPU 3. No CP trainer
or optimizer was launched, and GPU 1 was unused.

One actual prompt group produced eight distinct sample identities, each with
512x512/93f video and all 20 CPS denoising transitions. Native collector output
contains real rewards and genuine prompt embeddings, not synthetic conditioning
or injected reward values. Every chunk receipt identifies GPU 2 and policy 1.

| Native phase | Seconds |
| --- | ---: |
| Generation wall, eight samples | 492.601006 |
| Reward wall, one batched call | 37.171434 |
| Total measured collect call | 529.812738 |
| Generation/reward overlap | 0 |

Per-sample worker execution was approximately 61-62 seconds; native stage
receipts show about 50.5 seconds denoising and 9.2 seconds video decode per
sample. Warm prompt encoding was about 0.039 seconds. Queue waits rise with
sample index because this bounded baseline intentionally used one generation
worker. GPU snapshots were 100% busy during generation, with about 25.9 GiB
visible memory use. These timings do not establish multi-GPU speedup.

The reward wall includes lazy model loading. Reported inference was 2.863248s
for the batch and artifact materialization 6.621385s; these submetrics do not
sum to the cold reward wall. Rewards in sample order:
`[-2.443425, -2.946535, -3.793725, -4.589759, -3.887390, -3.915669,
-3.734280, -3.280910]`. All are finite and nonconstant; no quality threshold or
learning improvement is implied. The same full strict-loaded Kling checkpoint
and Transformers key-layout caveat documented above apply.

Independent CPU artifact audit passed the production trajectory validator,
eight unique sample IDs with sample indices 0-7, one reward group, 20-step axis
and finiteness of every segment tensor. A custom deserialization location
callback required every original serialized storage tag to be `cpu`, rather
than hiding device placement with forced CPU mapping. Observations and actions
are FP32 `[8,20,16,24,64,64]`; old log-probabilities are FP32 `[8,20]`; actual
prompt embeddings are BF16 `[8,512,100352]`. Estimated payload is
2,893,025,280 bytes. This is not independent model replay parity yet.

Evidence directory: `cosmos_native_collector_real_group` under the NVMe output
root, including `initial_trainable_state.pt`, `rollout_batches.pt`,
`result.json`, `artifact_audit.json`, resolved configuration, both executed
scripts, reward debug receipts and copied Ray logs. This retained group is the
input for the next real-conditioned replay/CP gate; do not regenerate it just
to recreate the same acceptance workload.

The private Ray session used `/tmp/ray` on the >95%-full root filesystem and
emitted repeated capacity warnings. Its directory was only about 920 KiB when
checked; no spill failure occurred. Future launches should set `RAY_TMPDIR`
to NVMe before Ray initialization. Both collector and artifact audit exited 0;
owned driver/worker/raylet PIDs are absent, fresh compute inventory empty,
and GPUs 0/2/3 released. Complete CP updates, controlled distributed throughput,
checkpoint/EMA/quality and full paper workload remain open.

## Independent released-model replay of the complete real group

A fresh process loaded the pinned released Cosmos replay model on physical
GPU 0 and the collector's saved initial LoRA. Native trainable-state validation,
load and readback verification completed before any replay forward. Checkpoint
identity matches the collected local-tree SHA256 exactly. No text encoder,
VAE, reward or Ray worker was rebuilt; the actual saved embeddings, observations
and actions were used directly.

All eight samples were rescored over all 20 stored transitions, visiting steps
19 down to 0 for each sample to avoid relying on previous forward order. Native
Cosmos state restoration and `sde_step_with_logprob` used the original CPS
noise level 0.7, scheduler schedule, BF16/IEEE outer-autocast configuration and
native rank-32/alpha-64 adapter. Scheduler timesteps were checked exactly
against the saved schedule. Each predicted tensor and recomputed log-probability
was finite. The predeclared absolute tolerance was 1e-3; it was not relaxed.

Result: **160/160 transitions, max log-probability difference 0, max
`abs(exp(new-old)-1)` 0**. This includes the terminal scheduler transition,
not 160 independent nondegenerate stochastic steps. A separate JSON audit
confirmed 160 unique `(sample, step)` pairs spanning samples 0-7 and steps 0-19,
not just a claimed count in the result summary.

Replay/verification wall time after loading was 405.980307 seconds; per-sample
time was 50.50-51.27 seconds. Peak allocated memory after the counter reset was
7,739,583,488 bytes. These are isolated replay measurements, not generation,
training or distributed speedup. Initial load and readback are outside timing.

Evidence: `cosmos_real_independent_replay/{executed_probe.py,transitions.json,
result.json}` under the NVMe output root. Process exited 0, fresh GPU compute
inventory empty, GPU 0 released. The prior collector group is retained for
the next real-conditioned CP gate. CP numerical behavior, gradients/optimizer
updates, recovery, quality and full paper-budget acceptance remain unverified
by this independent single-device replay result.

## Real-conditioned two-device CP admission check

The explicit native `ContextParallelStrategy(cp_size=2)` replayed saved real
samples 0 and 7 at steps 19, 10 and 0 on physical GPUs 0/1. Both ranks freshly
loaded the released replay model, validated/loaded/read back the saved initial
LoRA, then prepared the strategy (including its initial broadcast). Strict
determinism, IEEE, efficient SDPA and the strategy's fixed-row/FP32-LoRA
contract were enabled. This is six selected transitions, not the full group.

| Sample | Step | Max log-prob absolute error vs saved rollout |
| --- | ---: | ---: |
| 0 | 19 | 2.7750854e-8 |
| 0 | 10 | 6.5565109e-7 |
| 0 | 0 | 1.1920929e-7 |
| 7 | 19 | 5.8343669e-8 |
| 7 | 10 | 1.8477440e-6 |
| 7 | 0 | 5.9604645e-8 |

All predictions/log-probabilities were finite, both ranks' complete noise
predictions matched exactly, and all six log-prob errors passed the fixed
1e-3 absolute tolerance. Maximum error is 1.8477440e-6, not zero. Rank equality
is not equality with the original rollout noise prediction, which was not
stored. The original Ray rollout did not use fixed-row/FP32-LoRA compute;
this admission check must not be relabeled as exact rollout kernel parity.

Selected replay plus checks took 71.116996 seconds; individual transitions
took 11.71-12.26 seconds. Each rank's peak allocated memory after preparation
was 6,620,361,216 bytes. The prior native single-device replay averaged about
2.54 seconds per transition, but these runs have different compute contracts
and scopes. No fair speedup/slowdown factor or CP-only overhead attribution
is established. The next useful performance control uses the same six inputs
and fixed-row/FP32-LoRA/efficient-SDPA contract without CP sharding.

Evidence: `cosmos_real_cp_replay_admission/{executed_probe.py,transitions.json,
result.json}` under the NVMe output root. Torchrun exited 0, probe processes
are absent and fresh compute inventory empty; GPUs 0/1 released. This closes
only selected real-input CP log-prob compatibility and rank-consistency checks.
Full real-group CP gradients/updates, matched-compute control, production
configuration dispatch and end-to-end quality/performance remain open.

## Matched-compute unsharded control

The same six real `(sample,step)` inputs used by CP admission were replayed on
GPU 0 without CP hooks, while retaining strict determinism, IEEE, efficient
SDPA, fixed-row Linear and the same explicit FP32 LoRA branches. Initial
adapter validation/load/readback, source checkpoint and data were unchanged.
This removes the earlier mismatch in the core compute contract.

| Six-transition diagnostic | Total seconds | Per-step seconds |
| --- | ---: | ---: |
| Fixed-row unsharded control | 125.183727 | 20.63-21.33 |
| CP2 admission | 71.116996 | 11.71-12.26 |

The observed phase ratio is 1.760x, or 43.190% less elapsed time with CP2.
These are one execution per arm with no confidence interval. CP additionally
timed rank-consistency broadcasts/reductions and a final memory all-gather;
the unsharded control has no equivalent collectives. Both timings include
replay validation and exclude initial loading. This is a diagnostic comparison,
not a pure-forward benchmark, full training throughput or production speedup.

Every saved-rollout log-prob absolute error matches the corresponding CP
receipt's reported absolute error; maximum is 1.8477440e-6, below unchanged
1e-3 tolerance. Matching scalar absolute errors do not establish equality of
the full prediction tensors across these two runs. Unsharded peak allocated
memory was 7,977,218,048 bytes, versus 6,620,361,216 bytes per CP rank.

Important decision: fixed-row computation itself is expensive. The previous
ordinary native single-device replay averaged about 2.54 seconds per step,
far below either fixed-row arm (different scope/contract, not a controlled
speedup figure). CP distributes the expensive fixed-row path successfully,
but this evidence does not justify replacing the ordinary native path for
this geometry, which already fits on one GPU. Leave CP out of public/default
dispatch; real update semantics and the cost of its compute contract still
need resolution before production adoption.

Evidence: `cosmos_real_fixed_replay_control/{executed_probe.py,transitions.json,
result.json}` under the NVMe output root. Process exited 0 and fresh compute
inventory empty; GPU 0 released. No new video generation or reward scoring.

## Real CP single-slice update: probe capture failure retained

The next probe used all eight real rewarded samples at timestep 10. Native
algorithm construction and whole-group advantages were evaluated once per
rank, then each sample's native GRPO loss was weighted by 1/8 and passed to
`ContextParallelStrategy.backward`. Gradient checkpointing was enabled. The
native optimizer factory and `OnlineTrainer._clip_and_step` owned reduction,
clipping and update. This is one time slice and one pass, not the configured
multi-timestep/four-PPO-epoch update.

All eight forward/backward microbatches completed, totaling 767.116901 seconds.
Maximum pre-update saved-rollout log-prob error was 4.0829182e-6 (fixed limit
1e-3); all rank noise-prediction comparisons were exact. Both ranks reached
the code after the native optimizer boundary and passed assertions that the
step happened, its norm was positive and trainable parameters changed.

**The probe then failed and is not an accepted update result.** It attempted
to read parameter gradients after `_clip_and_step`, which calls
`optimizer.zero_grad()` before returning. The resulting gradient dictionary
was empty. This was not evidence of non-finite gradients, but the process did
not save `update.pt`, numeric gradient norm, optimizer state or final rank
parameter equality. Those gates remain unproven, not implicitly passed.

The reusable probe now registers an optimizer step pre-hook, capturing cloned
CPU gradients after native reduction/clipping but before the optimizer clears
them. It also writes each rank's raw accumulated gradients before the update
boundary so a later capture failure need not force another full backward pass.
A CPU regression using the real `_clip_and_step` verified clipped gradients
are captured, parameters update, and live gradients are cleared afterward:
**1 passed in 9.01s**. The corrected GPU probe has not yet been rerun.

Failed evidence: `cosmos_real_cp_single_slice_update/{executed_probe.py,
transitions.json,failure.json}`. Reusable corrected source:
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_real_cp_update_probe.py`; regression:
`/mnt/nvme/outputs/wan22_i2v_cache/test_cosmos_gradient_capture.py`.
The corrected run uses a new `cosmos_real_cp_single_slice_update_captured`
directory and must retain this failed artifact unchanged. Torchrun exited 1,
the CPU regression exited 0, and fresh GPU inventory was empty; GPUs 0/1
released. Full unsharded gradient/optimizer parity and full recipe remain open.

## Corrected real CP single-slice update and artifact audit

The corrected probe completed the unchanged eight-sample group at timestep
10, with one whole-group advantage calculation and eight `loss/8` microbatch
backwards through the native CP strategy. It captured gradients through the
validated optimizer pre-hook and saved each rank's raw accumulated gradients
before the native reduction/clip/update boundary. Original failed receipts
remain unchanged.

All eight pre-update log-prob errors match the failed run's receipts; maximum
is 4.0829182e-6 at the fixed 1e-3 threshold. Both ranks' predictions matched
exactly. Native gradient norm was 0.0008163046441, and the optimizer stepped.
560 gradient tensors were captured, 280 were nonzero and 280 trainable tensors
changed. All updated trainable parameters matched exactly across the two
ranks. Zero gradients on the other initial LoRA branch are not a missing-
gradient failure: all 560 gradients and optimizer entries were present.

An independent CPU audit loaded the saved files and verified eight distinct
sample indices at timestep 10, identical complete advantage vectors across
ranks, finite gradients/updated weights/Adam moments, 560 optimizer entries
each at step 1, and the recorded changed-parameter count. For every gradient,
the two saved raw rank gradients were summed and the native clipping factor
applied; their result matched the captured post-clip gradient exactly (maximum
absolute difference 0; declared audit tolerances 1e-8 absolute/1e-5 relative).
The measured norm is below max_norm=1, so clipping did not reduce this update.
This checks saved CP reduction accounting, not an unsharded reference gradient.

Measured region including backward, update and artifact checks/saving took
770.363525 seconds. Each rank's peak allocated memory was 10,225,359,872 bytes.
This includes diagnostic snapshot and capture work and is not production
throughput. No new rollout, reward call, EMA or second update was performed.

Evidence: `cosmos_real_cp_single_slice_update_captured` under the NVMe output
root, including both executed scripts, eight transition receipts, final result,
independent `artifact_audit.json`, per-rank `pre_step_rank_*.pt` and `update.pt`
(updated trainables, optimizer state, post-clip gradients and advantages).
The GPU torchrun and CPU audit exited 0, probe PIDs are absent and fresh GPU
compute inventory empty; GPUs 0/1 released.

This passes the **real whole-group, single-time-slice CP update boundary**.
It does not pass the complete multi-timestep/four-PPO-epoch recipe, matched
unsharded gradient/optimizer equivalence, checkpoint recovery, EMA, learning
quality or production performance. Those gates remain open; the captured
artifacts provide the fixed reference for the next unsharded comparison.

## Real single-slice unsharded gradient and optimizer reference

The unsharded fixed-row/FP32-LoRA/efficient-SDPA reference was parallelized
over samples with native DDP4, not over tokens. Each physical GPU processed
two distinct samples (rank r owns 2r and 2r+1). Every rank calculated the same
whole-group advantage vector once; local `loss/2` combined with DDP's four-rank
mean yields the average loss over the original eight samples. This is an
unsharded reference with a different reduction order, not a bitwise serial
single-device run or a single-device timing measurement.

All samples 0-7 at timestep 10 completed native backward and trainer
clip/update. All four final trainable replicas matched exactly. The gradient
norm, advantage vector, 560 captured gradient tensors and 280 changed
trainable tensors match the corresponding CP result's counts and norm.
DDP pre-step snapshots contain already-DDP-reduced accumulated gradients,
unlike the CP snapshots, which preceded their final CP reduction.

An independent CPU comparison used limits fixed before the results were read:
gradient/update relative L2 <=1e-4, parameter max absolute error <=1e-6,
Adam first-moment relative L2 <=1e-4 and second-moment <=2e-4. It required
identical advantage vectors, optimizer parameter groups, tensor name/order
and one-step optimizer counters. All tensors were checked finite.

| CP2 vs unsharded DP4 reference | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Gradients (560 tensors) | 3.4060644e-7 | 4.0927262e-12 |
| Actual parameter updates | 7.0991262e-7 | 1.1191332e-8 |
| Final parameters | 1.6667707e-9 | 1.1191332e-8 |
| Adam first moments | 3.4237987e-7 | 4.0500936e-13 |
| Adam second moments | 5.0066448e-7 | 6.7762636e-20 |

The update comparison subtracts the saved initial parameters before measuring
relative L2, so unchanged large base values cannot conceal update disagreement.
All declared tolerances passed without modification. This supports numerical
equivalence for the tested real eight-sample, one-time-slice accumulated update,
not exact equality or complete multi-timestep/PPO training equivalence.

Four-device reference region took 288.514487 seconds, with peak allocated
15,153,960,960 bytes per rank. Each of the two concurrent microbatch waves
took roughly 143 seconds. Do not compare this directly to CP2's 770 seconds
as a same-resource speedup: device count, task placement and diagnostic work
differ. Neither fixed-row path is a production-performance recommendation.

Evidence: `cosmos_real_dp4_unsharded_reference` under the NVMe output root,
including executed source, all rank transition/pre-step receipts, `update.pt`,
`result.json`, `cp_comparison.json` and the executed comparison script. Torchrun
and CPU comparison exited 0; fresh GPU inventory empty, all four GPUs released.
Full recipe, checkpoint recovery, EMA, quality, original native-compute parity
and production throughput gates remain open.

## Ordinary native-compute control: update equivalence fails

The same native DDP4 harness reran the eight real samples at timestep 10 with
ordinary Linear/autocast and default SDPA selection, removing the fixed-row,
explicit FP32-LoRA and efficient-SDPA overrides as one combined change. Model,
initial adapter, full-group advantages, deterministic IEEE setting, sample
ownership, gradient checkpointing, optimizer and artifact capture stayed the
same. This ordinary-native update itself completed successfully: all eight
saved-rollout log-prob differences were exactly 0, the four updated replicas
matched exactly, 560 gradients were captured and 280 trainable tensors changed.
Gradient norm was 0.0008508932078.

**Cross-contract update equivalence failed by a large margin.** The independent
comparison retained exactly the previously declared thresholds; neither
failure was relabeled as a pass:

| Comparison to ordinary native DP4 | Gradient relative L2 | Update relative L2 | Parameter max abs |
| --- | ---: | ---: | ---: |
| Fixed-compute CP2 | 0.5258839531 | 0.5617156138 | 1.9938813e-4 |
| Fixed-compute unsharded DP4 | 0.5258839531 | 0.5617156139 | 1.9938814e-4 |

Adam first/second moment relative L2 differences were approximately 0.525884
and 0.626079. Limits remain 1e-4 for gradient/update/first moment, 2e-4 for
second moment, and 1e-6 parameter max absolute error. These are relative L2
differences, not percentages of incorrect tensor elements. Comparing actual
updates (subtracting the common initial state) prevents unchanged adapter
values from concealing the discrepancy.

The matched DP4-versus-DP4 result isolates the **combined compute-contract
change**, not CP token partitioning. It does not identify which of fixed-row
GEMM shape, FP32 LoRA execution or attention backend is responsible, nor which
contract gives better learning/quality. The earlier CP-to-fixed-reference
agreement remains valid, but must not be generalized to ordinary native
training. Small saved-rollout log-prob drift did not guarantee gradient or
optimizer equivalence in this real group.

Native DP4 measured region was 36.435179 seconds versus fixed-compute DP4
288.514487 seconds, using the same device count and diagnostic capture scope.
This is one single-time-slice observation per arm, not an end-to-end or
confidence-qualified benchmark. Native peak allocated memory was
15,429,804,032 bytes per rank. Do not deploy/promote the fixed CP contract as
a semantics-preserving or performance-improving replacement for native mode.
Next diagnostic: isolate the individual compute overrides with the same saved
group, without changing tolerances or regenerating videos.

Evidence: `cosmos_real_native_dp4_update` under the NVMe output root, with
executed GPU source, native update/pre-step/transition artifacts, `result.json`
(`passed` execution only), and **failed** `cp_comparison.json` and
`fixed_comparison.json` with their executed comparison scripts. Torchrun exited
0; both parity comparisons exited 2 as intended on threshold failure. Fresh
GPU inventory empty, all four GPUs released. Full recipe, native-compatible
CP update semantics, recovery, EMA and quality gates remain open.

## Attention-only ablation: efficient SDPA is sufficient for parity failure

The same real DP4 single-slice harness changed only the attention selection
from the native default to `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`.
Ordinary Linear and LoRA autocast were retained; fixed-row execution and
explicit FP32-LoRA overrides were not installed. All other workload, state,
determinism, optimizer, sample ownership and capture settings were unchanged.

The update executed successfully and produced identical final replicas across
four ranks. All eight log-prob absolute errors were below 1e-3, with maximum
4.3213367e-6. Gradient norm was 0.0008289589896; 560 gradients were captured
and 280 trainable tensors changed. These execution checks did not imply
native-update parity: the independent unchanged-threshold comparison failed.

| Efficient-only vs native DP4 | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Gradients | 0.5228995987 | 1.1342578e-5 |
| Updates | 0.5470939425 | 1.9958774e-4 |
| Adam first moments | 0.5228995987 | 1.1342580e-6 |
| Adam second moments | 0.6227557954 | 8.8303137e-14 |

For this real group, forcing efficient SDPA alone is sufficient to reproduce
a large gradient/update mismatch. It is not necessary to combine it with
fixed-row GEMMs or FP32 LoRA to fail the native-parity gate. This does not
prove those other changes have no effect, apportion the combined error, or
establish which backend is more accurate against a higher-precision reference.
Do not describe efficient SDPA generally as broken based on this one workload.

Measured diagnostic region took 100.509438 seconds versus native DP4's
36.435179 seconds. Peak allocated memory was 15,429,801,984 bytes/rank. These
single-run, single-time-slice observations isolate a workload-specific cost,
not a general backend benchmark or end-to-end training speedup.

Evidence: `cosmos_real_efficient_only_dp4_update` under the NVMe output root,
including executed GPU source, all transition/pre-step/update artifacts,
execution `result.json`, failed `native_comparison.json` and its executed
comparison source. GPU job exited 0; unchanged-threshold CPU comparison exited
2. Fresh GPU inventory empty; all four GPUs released. Native-compatible CP
dispatch remains unapproved. Other individual overrides/interactions and the
full training/quality gates remain open.
