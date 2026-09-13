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

## Fixed-row-only ablation: a second independent native-parity failure

Candidate `435c8fa2` was unchanged. The native DP4 single-slice harness changed
only Linear execution to `fixed_row_linear_compute(base_handle, rows=64)`.
No FP32-LoRA module overrides were supplied, and default native SDPA selection
was retained. The saved eight real samples, initial adapter state, timestep 10,
whole-group advantages, two samples/rank with loss divided by two, strict IEEE
determinism, checkpointing and native optimizer boundary match the control.

Execution completed, all four final trainable replicas were exactly equal,
560 gradients were captured and 280 trainable tensors changed. Gradient norm
was 0.0008438613731. Maximum saved-rollout log-prob error was 3.8146973e-6,
below the admission tolerance, but the unchanged update-parity limits failed:

| Fixed-row-only vs native DP4 | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Gradients | 0.5711056042 | 1.0721618e-5 |
| Updates | 0.5755606012 | 1.9952650e-4 |
| Adam first moments | 0.5711056042 | 1.0721621e-6 |
| Adam second moments | 0.7067326491 | 1.0648253e-13 |

Limits remain gradient/update/first-moment relative L2 1e-4, second-moment
relative L2 2e-4, and parameter maximum absolute error 1e-6. Parameters had
maximum absolute error 1.9952650e-4. No thresholds were relaxed. This isolates
another override sufficient to fail native parity on this group: removing
efficient SDPA alone cannot make the fixed-row contract native-equivalent.
It does not identify higher-precision truth, prove a general Linear defect,
or establish learning quality. FP32-LoRA-only and interaction effects remain
unresolved; the separate single-factor errors must not be added together.

The measured diagnostic region took 244.957895 seconds versus native DP4's
36.435179 seconds, with peak allocated memory 15,246,827,520 bytes/rank.
These are single-run, single-time-slice observations, not complete training
throughput or a statistically qualified benchmark. The fixed-row path is not
approved as a semantics-preserving or performance-improving native replacement.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_real_fixed_rows_only_dp4_update`,
including executed GPU/comparison sources, four transition/pre-step receipts,
`update.pt`, execution-only `result.json` and failed `native_comparison.json`.
GPU process exited 0, CPU comparison exited 2 on threshold failure. Fresh GPU
compute inventory was empty; all four GPUs released. Full production training,
native-compatible CP, recovery, EMA and quality gates remain open.

## FP32-LoRA-only ablation: smaller mismatch, still outside native limits

The same native DP4 diagnostic changed only the LoRA A/B Linear branches:
autocast disabled inside each branch and input converted to FP32, with existing
FP32 parameters. Native full-row Linear and default SDPA were unchanged. The
probe-only wrapper remains active through checkpoint recomputation/backward
and restores forwards on exit. CPU tests covered FP32 execution under BF16
autocast, exact FP32 control output, finite gradients and restoration on both
normal and injected-exception exits: 2 passed in 8.19 seconds.

Same saved group, weights, timestep, loss averaging, optimizer and capture
settings as previous arms. Execution passed, all eight log-prob errors were
zero and final trainable replicas were exact across four ranks. There were
560 captured gradients and 280 changed tensors; gradient norm 0.0008509107865.
Nevertheless the unchanged native-parity comparison failed:

| FP32-LoRA-only vs native DP4 | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Gradients | 0.0029878823 | 5.0235485e-8 |
| Updates | 0.0066886213 | 7.4179014e-5 |
| Adam first moments | 0.0029878823 | 5.0235371e-9 |
| Adam second moments | 0.0041234047 | 4.5701832e-16 |

This effect is much smaller than the efficient-SDPA-only or fixed-row-only
differences, but still fails the original tolerances. Exact initial log-probs
do not establish derivative equivalence. No quality or higher-precision-truth
claim follows, and individual ablation errors cannot be summed to explain the
combined contract. The measured region was 38.485526 seconds; peak allocations
were 15,337,463,808 / 15,153,963,008 / 15,337,463,808 / 15,337,463,808 bytes.
Timing is one diagnostic update, not complete training throughput.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_real_fp32_lora_only_dp4_update`
contains the executed probe, precision helper, helper tests and comparator,
all per-rank receipts, `update.pt`, execution `result.json` and failed
`native_comparison.json`. GPU process exited 0, comparison exited 2. Fresh GPU
inventory empty before transferring all four GPUs to an unchanged native
fresh-process repetition to check reference reproducibility. Production CP,
full recipe, recovery, EMA and quality remain open.

## Native reference repetition: exact update reproducibility

A fresh four-process launch repeated the original native DP4 script with only
its output directory changed. Same candidate, model/adapter, saved real group,
default SDPA, native Linear/autocast, precision/determinism and optimizer.
All eight log-prob errors were zero, all four final parameter replicas exact,
and the native comparison found zero relative L2 and zero maximum absolute
error for all 560 gradient, update and final parameter tensors and both Adam
moments. Optimizer parameter groups, advantages and step counters matched.

The measured region was 36.454606 seconds versus the original 36.435179;
all ranks peaked at 15,429,804,032 allocated bytes. Gradient norm again was
0.0008508932078, with 280 changed tensors. This repetition supports reference
stability for the tested diagnostic; it is not a broad reproducibility or
statistical throughput guarantee.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_real_native_repeat_dp4_update`,
with executed probe/comparator, per-rank receipts, update artifact, execution
result and passed `native_comparison.json`. GPU job and CPU comparison both
exited 0, fresh compute inventory empty, all four GPU claims released.

The three single-factor ablations all fail unchanged native-update limits,
while the unchanged reference repeats exactly. Keep the fixed-compute CP
contract experimental; none of these ablations warrants silently changing
native rollout/training precision. The matched fixed CP/DP proof remains only
within its explicit compute contract. Full native DP training semantics versus
single-device accumulation, production integration and full-recipe quality
remain required; these diagnostic slices do not close them.

## Native DP4 versus single-device accumulation: real-slice parity passed

The native control was run with `SingleProcessStrategy` on GPU 0, accumulating
all eight saved real samples at timestep 10 with each loss divided by eight.
The reference DP4 arm has two disjoint samples/rank with loss divided by two
and DDP averaging across four ranks. Both use the same complete eight-sample
advantages, initial adapter, released backbone, checkpointing, native Linear,
LoRA autocast, default SDPA, strict deterministic IEEE and native optimizer
boundary. No fixed-compute overrides or regenerated trajectories were used.

All eight saved-rollout log-probs matched exactly. The single-device gradient
norm was 0.0008508932078, identical to the reported DP4 norm; 560 gradient
tensors were captured and 280 trainable tensors changed. The unchanged parity
limits passed for all captured gradients, updates, parameters and Adam moments:

| Single-device vs native DP4 | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Gradients | 8.5849871e-11 | 2.8421709e-14 |
| Updates | 3.3058375e-10 | 2.9103830e-11 |
| Parameters | 7.7877419e-13 | 2.9103830e-11 |
| Adam first moments | 9.5091321e-11 | 3.5527137e-15 |
| Adam second moments | 2.8007779e-11 | 2.6469780e-23 |

Advantages and optimizer parameter groups matched exactly, and all optimizer
step counters were one. Relative errors use DP4 as the denominator. This
establishes native data-parallel accumulation equivalence within the original
tolerances for this real group and single selected timestep, not bitwise
equivalence or the entire multi-timestep, multi-epoch online training recipe.

Single-device measured region was 140.464884 seconds and peak allocated memory
15,063,064,576 bytes. Original DP4 was 36.435179 seconds; its independent repeat
was 36.454606 seconds. Observed phase speedup is approximately 3.85x for equal
eight-sample work, using four GPUs versus one. The regions include forward,
backward, optimizer and diagnostic capture; DP4 also performs replica checks
and distributed collection. They exclude initial model/data loading, rollout,
reward and production weight synchronization. This is not an end-to-end speedup
or a statistically qualified benchmark; single-device has one observation.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_real_native_single_update`,
including executed probe/comparison sources, transition/pre-step receipts,
`update.pt`, execution result and passed `native_comparison.json`. GPU process
and CPU comparison exited 0, fresh compute inventory empty; GPU 0 released.
No other GPUs were claimed by this arm. Prefer this native DP path for the next
full-recipe semantic/integration checks; the fixed-compute CP contract is still
experimental. Full update cadence, streaming/global advantage normalization,
weight synchronization, recovery, EMA and quality remain separate open gates.

## Native OnlineTrainer full cached-group cadence: setup and scope

The external `cosmos_native_trainer_probe.py` connects the saved real group to
`OnlineTrainer.collect_training_batch` through its prepared-iteration input,
then calls the actual `train_on_rollout_batch`. It uses the native algorithm
and evaluator factory, DDPStrategy, optimizer construction, mandatory first
update parity gate and EMA. No replacement loss, timestep loop, optimizer
boundary or precision override is installed in the trainer.

CPU preflight passed: the actual cached eight-sample group has finite rewards
and nonzero full-group advantages; samples 0/1, 2/3, 4/5, 6/7 are assigned to
ranks 0-3. Full-group advantages are computed before slicing and supplied to
collection. Native timestep selection is [0,2,4,6,8,10,12,14,16,18], with four
PPO epochs and replay batch size one: 80 training evaluations per rank and
four optimizer boundaries. First-step diagnostics can add no-grad evaluations.
EMA remains enabled with decay 0.9 and update interval one.

This is explicit sample sharding of one cached group, not the production
prompt-group dispatcher. Each local slice is treated as one two-sample batch;
DDP averaging recovers the full-group loss for this balanced arrangement.
Prepared full-group reward statistics are supplied, but local group-size and
prompt-count metrics do not represent a production global prompt batch.
There is no collector or weight syncer: the native post-training schedule hook
has no weights to push. It does not test live rollout, rewards, queue behavior,
version synchronization, recovery, production ownership or quality. This
cached configuration has global_std=false and no SFT regularizer; it is not
the entire paper recipe or a real-GPU global_std streaming acceptance.

While the GPU integration ran, existing CPU-only streaming regressions were
rerun from unchanged candidate 435c8fa2: test_streaming_global_std.py and
test_fsdp_streaming_equivalence.py, 12 passed in 11.34 seconds. These cover
streaming/full-batch advantage, gradient and Adam agreement and four-process
Gloo behavior including rejection of uneven post-filter group counts. They
are not GPU FSDP/Cosmos proofs. CPU checks overlapped the GPU run, so its time
is an execution receipt, not an isolated performance benchmark.

Preflight evidence: cosmos_native_trainer_preflight/preflight.json under the
NVMe output root. The first torchrun launch exited 2 in argument parsing before
starting workers because --run was an ambiguous launcher abbreviation. A --
separator before the script corrected command dispatch without changing the
experiment. The subsequent run uses cosmos_native_trainer_full_ppo; completion
must be established from its final artifacts and independent audit below.

## Native OnlineTrainer cached-group cadence: execution and audits passed

The unchanged native trainer completed all four configured PPO epochs on the
ten selected timesteps. All ranks reached four optimizer boundaries, each
capturing 560 finite post-clip gradient tensors. At the end, all trainable
parameters were exactly equal across four ranks. Native trainer state recorded
step=1, global_step=4, all 560 Adam states at step four, and EMA num_updates=4
with 560 finite shadows. Nonzero gradient tensor counts by update were
280, 560, 560, 560. No training-quality inference follows from those counts.

The mandatory initial global replay gate recorded finite=true and maximum
log-prob difference 0.0. Its native configured limit was 0.01; the actual zero
also satisfies the earlier diagnostic 0.001 limit. No tolerance was changed
for this run. Subsequent policy log-prob changes were finite, as expected when
replaying fixed old actions after weight updates. The rank-0 aggregate maximum
over the training call was 0.000171456486; this is not an independent final-step
replay comparison or a quality measure. Rank-local loss/reward-component and
advantage metrics must not be presented as global batch metrics for this
explicit sample-sharded harness (see scope above).

| Rank-0 optimizer boundary | Cumulative seconds |
| --- | ---: |
| First | 351.827761 |
| Second | 700.426706 |
| Third | 1048.986795 |
| Fourth | 1397.490415 |

Training-call elapsed time was 1397.785117 seconds (23.30 minutes), including
native initial checks and gradient capture but excluding model/data setup and
final snapshot serialization/audit. Peak allocated memory was 15,665,102,848
bytes/rank. Repeated spot checks showed all four GPUs at 100% utilization with
stable approximately 18GB process residency. No end-to-end or isolated speedup
claim: CPU regression/audit preparation overlapped this correctness run.

Independent CPU artifact audit passed: exact boundary sequences on four ranks,
finite gradients/parameters/moments/EMA, optimizer and EMA counts, and the actual
zero-error global replay record. A second CPU audit reconstructed four unfused
AdamW updates from the saved post-clip gradients and native hyperparameters,
then reconstructed the native EMA warmup/lerp history. Comparison to the GPU
final snapshot passed its predeclared limits:

| CPU reconstruction vs GPU final state | Relative L2 | Max absolute error |
| --- | ---: | ---: |
| Parameters | 5.1765096e-9 | 7.4505806e-9 |
| EMA | 4.3095946e-9 | 1.4901161e-8 |
| Adam first moments | 2.1534541e-7 | 6.8212103e-13 |
| Adam second moments | 1.2921403e-5 | 4.0115480e-18 |

Limits: parameter/EMA maximum absolute error 1e-6, Adam first-moment relative
L2 1e-4 and second-moment relative L2 2e-4. This checks application of captured
gradients and EMA history, not independently computed full-recipe gradients,
fresh-process recovery or production weight synchronization.

Evidence under `cosmos_native_trainer_full_ppo`: executed GPU probe and both
CPU audits, four gradient snapshots, per-rank boundary records, native debug
gate, final_state.pt, result.json, artifact_audit.json and optimizer_replay.json.
GPU job and both CPU audits exited 0. Fresh compute inventory empty; all four
GPU claims released. Production group dispatch, live rollout/reward/weight
sync, real-GPU global_std streaming, full checkpoint recovery and quality
remain open. Fixed-compute CP remains experimental and was not used here.

## Fresh four-rank strict trainer state restoration passed

A new torchrun launch rebuilt the released backbone and native DDP trainer
in four fresh processes, loaded the trained adapter through native validation,
load and readback APIs, then called OnlineTrainer.load_state_dict(strict=True)
on the saved four-update trainer state. No model or optimizer update ran.

Each rank exported its restored state and recursively compared every tensor,
shape, dtype, dictionary key, sequence type and scalar metadata against the
saved snapshot. All 560 trainable model tensors and 2240 trainer tensors were
exactly equal on every rank. Optimizer identity manifests and hyperparameters
matched; trainer step=1, global_step=4 and EMA num_updates=4 were preserved.
Final trainable replicas also matched across ranks by direct broadcast/compare.

Before loading, the probe deliberately marked rollout weights initialized,
replay parity passed, and precision guard not pending. Strict loading correctly
reset these to false, false and true respectively. Thus a restored process
does not inherit stale rollout-sync or replay-admission readiness.

The timed load/readback/replica-comparison region was 0.740620-0.741820 seconds
across ranks, excluding backbone construction, input loading and process start.
This is not a complete restart latency or a full recovery performance result.
Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_native_trainer_fresh_restore`,
including the executed probe and imported setup source, four rank receipts
and result.json. GPU process exited 0, fresh compute inventory empty; all four
GPUs released. Candidate remained unchanged.

This verifies fresh-process state rehydration, not uninterrupted-versus-resumed
future training equivalence. The source snapshot did not capture process RNG,
recipe progress, data iterator or live rollout runtime state. Generic checkpoint
writer/loader integration, per-rank RNG continuation, next-update equivalence
and worker synchronization remain required for complete recovery acceptance.

## Native checkpoint file and fresh-process per-rank RNG roundtrip passed

Two separate four-process launches exercised the generic production checkpoint
APIs on the real four-update model/Adam/EMA state. The writer reconstructed the
native trainer, strictly loaded that state, and called save_training_checkpoint
on every rank. The reader started with fresh backbone/adapter construction,
loaded TrainingCheckpoint, then used restore_training_checkpoint(strict=True)
with the runtime's actual resolved model identity. It did not first initialize
the reader's adapter from the saved trained snapshot.

The schema-v2 checkpoint.pt is 918,715,353 bytes with matching metadata and
resolved model identity (local-tree SHA256
4fec540c29ab67ad37ad12262bdac706e6849c41912590893b71ad845a7d2010).
No optional warm-start adapter exports were requested. Probe progress explicitly
records next_epoch=1 and next_step=4 with a scope marking it as a cached-group
test position, not production data iterator state. The reader checked these
values, identity and metadata through native validation.

Each rank compared all 560 trainable model tensors and 2240 trainer-state
tensors exactly to the trained source, including dtype/shape and every scalar,
sequence and mapping field. Model/optimizer/EMA/counters and manifests passed
in both writer and reader. Reader rollout-sync/replay readiness was correctly
invalidated and precision guard rearmed.

The writer captured rank-distinct Python, NumPy, Torch CPU, all four visible
CUDA devices and a named CPU generator. It recorded 16 future draws per stream
and verified checkpoint writing did not advance them. Native checkpoint RNG
gather retained four rank trees; the fresh reader restored its own rank tree
with strict world_size=4 and reproduced every recorded draw exactly. Independent
CPU inspection confirmed all four rank-specific future CPU streams were
distinct, so equal seeding did not conceal rank-selection errors.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_native_checkpoint_roundtrip`,
including executed probe/imported sources, native checkpoint and metadata,
writer future-draw references, per-rank write/read receipts and both results.
Both GPU launches and the CPU distinct-stream assertion exited 0; fresh compute
inventory empty, all GPU claims released. This closes the tested native file,
state and RNG roundtrip, not full recovery acceptance: uninterrupted versus
resumed future training, actual recipe/data cursor continuation and live
rollout/weight synchronization remain open. No training or quality result was
generated by this roundtrip itself.

## Actual next-update save/continue versus fresh resume: protocol

The next probe starts from the saved real four-update trainer state. A control
process reconstructs that state, captures rank-distinct RNG, writes a new native
checkpoint and continues without reloading it. A separate fresh four-process
launch restores that same checkpoint through native strict APIs and continues
the same computation. The control is uninterrupted across this new save boundary,
not the original 23-minute training process, which had already exited.

Both arms retain native Linear/autocast/default attention and checkpointing.
They use OnlineTrainer.begin_optimizer_update, backward_on_training_batch and
finish_optimizer_update for one next optimizer boundary over all ten selected
timesteps. No alternate loss or optimizer loop is substituted. The original
four-PPO-epoch configuration is not rewritten; the explicit streaming API
performs exactly one optimizer update, advancing global_step 4 to 5 and EMA
num_updates 4 to 5. This is a next-update recovery test, not another complete
four-epoch recipe call.

Inputs remain the same real cached eight samples with old actions/log-probs
and full-group advantages supplied before balanced two-sample/rank slicing.
No new policy-version-matched rollout or production iterator runs. Both arms
consume the same saved future RNG reference before training and capture their
future RNG after training. The comparator requires exact gradients, model,
optimizer, EMA, counters/metadata and each rank's future random draws, retaining
any mismatch instead of relaxing limits. Completion is determined from the
GPU receipts and independent comparison, not this protocol description.

## Actual next-update save/continue versus resume: exact match passed

Both four-rank launches completed the native next optimizer update over the
same ten selected timesteps and cached eight samples. Each arm verified changed
trainable parameters and exact final trainable replicas across ranks. All ranks
reported gradient norm 0.0012481594458, global_step=5 and EMA updates=5.
Rank-0 training regions were 346.828555 seconds for save/continue and 346.742082
seconds for fresh resume, excluding setup/checkpoint I/O and final serialization.
These are single correctness-run timings, not restart-latency or speedup claims.

Independent CPU comparison required exact equality, without relaxed tolerances:

| Compared final artifact | Exact tensor count | Result |
| --- | ---: | --- |
| Trainable model state | 560 | Exact |
| Trainer state, including Adam and EMA | 2240 | Exact |
| Captured post-clip gradients | 560 | Exact |
| Post-update RNG draws, each of four ranks | 7 tensor streams plus Python draws | Exact |

The full gradient/model/trainer artifact comparison uses the saved rank-0
snapshots; each GPU arm separately checks final trainable parameter agreement
across ranks, and all ranks' future RNG references are compared independently.
Recursive comparison also checks every scalar, dtype, shape, sequence type and
dictionary key. Explicit assertions require trainer step=2, global_step=5,
560 optimizer entries each at step five and EMA num_updates=5 in both arms,
so equality alone cannot hide a jointly omitted update.

This advances recovery evidence beyond loading and random-draw probes: an actual
full ten-timestep optimizer update from this saved real state is identical
across the save/continue versus fresh-resume boundary. It remains a cached-group,
explicitly sample-sharded diagnostic. It does not establish production iterator
progress, new rollout version synchronization, full recipe scheduling, recovery
of a live continuous queue, or learning quality.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_native_next_update_resume`,
with executed probe/comparator and imported sources, native checkpoint, both
arms' final states and per-rank receipts/future RNG references, and passed
comparison.json. Both GPU launches and the CPU comparator exited 0. Fresh GPU
compute inventory empty, all four claims released. Production rollout/weight
sync, actual data progress and remaining quality gates stay open.

## Trained checkpoint delivery to four real Ray workers passed

The existing native vrl.scripts.perf.weight_delivery_probe ran unchanged on
candidate 435c8fa2 with four isolated single-GPU acceptance actors. It strictly
restored the real four-update checkpoint through native model identity checks,
exported the sender snapshot, and loaded full real Cosmos generation models
on all four L40S devices. The tool launches its own actor fleet directly; this
is not the production trainer/rollout/reward placement in the source config.
No GPU trainer or reward worker was active. Ray temporary files used NVMe.

Each receiver first installed and verified an intentionally different parameter
payload: zero entries replaced by one and nonzero entries replaced by zero.
Two subsequent native snapshot syncs installed the trained sender payload with
versions 1 and 2, requiring parameter-content readback and matching version ACK
on every receiver. This rejects a no-op update even if its version echo is right.
Both delivery versions refer to the same four-update trained snapshot, not two
additional training updates or a continuous retained-version-slot experiment.

All four receivers passed for 560 tensors and 183,500,800 bytes each. The shared
snapshot path was used (bucket_bytes=null). Fleet-wide sync plus readback took
0.687883 seconds on first install and 0.503922 seconds on repeated install.
Sender build/restore was 18.861041 seconds; snapshot export 0.068314 seconds.
Per-receiver poison install/readback took 0.215234-0.229403 seconds. Timings
exclude worker/model startup and are not video throughput, network bandwidth
or a controlled scaling benchmark. Source initialization reported seed zero,
deterministic=false; this parameter-byte test does not imply deterministic
forward execution. Actual GPU processes were observed at approximately 21GB
each during worker loading.

Existing test_weight_delivery_probe.py regressions also passed: 10 tests in
62.99 seconds, with one Ray future-behavior warning. They ran CPU-only and
overlapped source setup. The production tool publishes its report only after
successful fleet cleanup. It exited 0; fresh GPU inventory and raylet/GCS/probe
process queries were empty. All four GPU claims released.

Evidence: `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_trained_four_worker_delivery`,
with native result.json, executed source and command/scope receipt. This proves
trained parameter delivery and in-place verification on four real receivers.
No new video was generated, so fresh rollout/replay correctness, reward/quality,
production scheduling, restored-worker rollout initialization and continuous
queue/version retention behavior remain separate open gates.

## Fresh trained two-worker rollout, reward and independent replay passed

A bounded native collector restored the real step-four checkpoint and exported
its trained policy on CPU. All 560 trainable tensors differed from the original
untrained adapter snapshot. Native placement preflight and physical probes
confirmed rollout workers on GPUs 1/2 and reward on GPU 3; GPU 0 was reserved
as the trainer role but no GPU trainer ran during collection. The private Ray
session used /mnt/nvme/ray, not the near-full root filesystem.

Native weight sync installed collection version 1 on the fresh fleet. This
version number is fleet-local, not training global_step. The collector generated
eight new 512x512, 93-frame videos with 20 CPS steps, guidance 1.0 and noise 0.7
for the original block-sliding prompt. Runtime receipts prove four samples per
worker, eight unique batch keys and version 1 throughout, with physical GPU
IDs 1 and 2. After generation, both active workers independently verified their
full installed parameter content against the sender snapshot and returned
version 1. The file initial_trainable_state.pt in this output means the trained
policy at collection start, not zero-initialized LoRA.

Collection took 286.722211 seconds: generation phase 248.959041 seconds,
reward phase 37.722631 seconds, measured generation/reward overlap 0.0.
Executor dispatch wall was 248.728 seconds; each worker ran four approximately
62-second clips. Reward used the pinned full Kling checkpoint on GPU 3;
batch inference was 2.882527 seconds and artifact materialization 7.098953
seconds, with cold-load costs in the larger reward phase. Existing TF5 base
Qwen key warnings preceded the native full reward checkpoint strict remapped
load; they are not evidence of an adapter-only reward model being accepted.

These are bounded phase observations, not a controlled one-worker-versus-two
speedup result: the earlier one-worker run used different policy weights.
The zero overlap and reserved idle trainer GPU must not be described as a
continuous four-GPU pipeline. Reward outputs were finite and are retained in
result.json; their means are not a controlled quality comparison or evidence
of improved learning.

CPU artifact audit passed native trajectory validation, eight distinct sample
IDs, all finite tensors/rewards, twenty steps and CPU-only serialized storage.
Estimated payload was 2,893,025,280 bytes. Additional JSON assertions checked
all worker assignments, physical GPU IDs, batch-key uniqueness and versions.
Ray collection/readback exited 0 and fresh compute/Ray process inventories
were empty before replay started.

Four new independent replay processes then loaded the trained collection
snapshot with native parameter readback. Each replayed two disjoint samples
in reverse timestep order, using the actual saved observations, actions and
prompt embeddings. No DDP/CP or optimizer was involved. All **160 distinct
sample/step pairs** passed: maximum log-prob difference 0.0 and ratio deviation
0.0 under the unchanged 1e-3 limit. Coverage includes the terminal transition,
not 160 nondegenerate stochastic transitions. Per-process measured replay
regions were 101.395516-101.748304 seconds, excluding loading; peak allocated
memory was 7,739,583,488 bytes each. The independent CPU coverage audit passed.

Evidence under the NVMe output root: cosmos_trained_collector_preflight,
cosmos_trained_collector_real_group (executed source, trained policy, trajectory,
native result, artifact audit and its source), and
cosmos_trained_independent_replay (executed replay/audit sources, per-rank
transitions/results and audit.json). Replay launch and all CPU audits exited 0;
fresh compute inventory empty, all four GPUs released. This establishes a
trained-checkpoint -> native multi-worker generation/reward -> independent
replay chain. Production iterator continuation, live trainer-to-worker loop,
continuous queue recovery, full recipe and quality acceptance remain open.

### Trained collector overlap capability preflight

The saved trained-collection configuration was resolved on CPU against the
unchanged Cosmos candidate 435c8fa2. Its native reward runtime reports
scoring_is_nonblocking=false and external_accelerator_isolation_verified=true;
the collector consequently reports supports_reward_generation_overlap=false.
The resolved batch plan contains one prompt group. Thus a separate reward GPU
does not establish asynchronous execution, and this single-group collection
has no following group whose generation could overlap scoring.

An explicit PER_GROUP_STREAMING request with two placeholder prompts was
rejected by the native capability guard before generation runtime access.
These prompts were never generated or scored. The successful probe loaded no
models and used no GPUs. Its receipt is
cosmos_reward_overlap_preflight_validated/result.json under the NVMe output
root, with executed source preserved beside it. Earlier empty preflight output
directories are not passes: the recovered retry failed because the probe
omitted three required collector arguments; the final probe supplies them.

Focused CPU regression coverage passed: 68 tests in 1.29 seconds across
tests/rollouts/collector/test_runtime.py,
tests/rollouts/orchestration/test_prompt_collection.py, and
tests/scripts/perf/test_reward_overlap_benchmark.py. This is capability and
scheduling coverage, not a GPU performance result or a production fix.

Next performance gate requires a native asynchronous reward service with
verified accelerator isolation, at least two real prompt groups, and equal-work
A/B/C arms: batched serial, per-group serial, and per-group streaming. Streaming
must beat the batched baseline, not just the extra per-group-call overhead.
Keep policy, prompts, seeds, sample count, generation parameters, reward model,
warmup and timing boundaries fixed; measure actual overlap and validate outputs.
Do not force the capability flag or claim these CPU checks removed GPU bubbles.

### Real HTTP reward capability admitted

On unchanged candidate 435c8fa2, an owned native Kling HTTP service was launched
with CUDA_VISIBLE_DEVICES=3, pinned reward revision
4f26600130683e6f1de9f5d463887f28e8ef995c, BF16 and min_frame_pixels=200704.
The client had CUDA_VISIBLE_DEVICES empty and never initialized CUDA. Service
isolation is operator-attested and enforced by the mask; the wire protocol
does not independently verify physical GPU UUIDs. No generator was running.

The actual saved Cosmos collector configuration and lifecycle were resolved,
and its native collector was constructed with the HTTP-backed reward runtime.
Overlap capability was false before preflight and true after native service
preflight, with nonblocking scoring and accepted isolation. No capability
property was overridden. This verifies admission, not actual concurrent work.

Two real HTTP requests scored the SHA-256-verified historical Wan reference
video through native MP4 materialization and Kling motion_quality. Both scores
were exactly -0.6127294366809066, matching the preserved historical in-process
runtime component score for that same source and reward setup. This is not a
fresh local-versus-HTTP controlled comparison or a Cosmos quality result.
First-call wall was 36.454959 seconds including lazy service model loading;
second-call wall was 1.017598 seconds, including 0.707802 seconds artifact
materialization and 0.305766 seconds reported model inference. These are two
observations, not a confidence interval or generation speedup benchmark.

Artifacts were cleaned after both calls. Client and service exited 0, followed
by empty GPU compute and reward-service process inventories; GPU 3 released.
Native service CPU regressions passed 55 tests in 0.88 seconds. Evidence:
cosmos_http_reward_capability/{result.json,service.yaml,service.log,executed_probe.py}
under the NVMe output root. The service was shut down, not left listening.
The next real gate remains two-or-more-group equal-work A/B/C generation and
reward scheduling, using the same service across warmup and measured arms.

### Real equal-work multi-group A/B/C pilot

Unchanged candidate 435c8fa2 completed all three native collection modes on the
same two-worker fleet and owned pinned Kling HTTP service. Native placement
probed GPUs 1/2 for generation and GPU 3 for reward; GPU 0 was a reserved trainer
role with an actual CPU sender, not an active GPU trainer. A live compute-process
inventory independently showed the two workers and reward service on separate
GPU UUIDs. This is not a four-GPU training or 3-rollout/1-trainer benchmark.

Each measured arm generated two native PromptExample groups of eight clips,
512x512, 93 frames, 20 CPS steps, noise 0.7, guidance 1.0, generation batch one.
Both groups used explicit request seeds (12340 and 22340), distinct fixed prompts
and the same trained step-four checkpoint. Plain string prompts were deliberately
not used: the native collector coalesces consecutive strings into one generation
request, which can remove the intended inter-group overlap opportunity.
CPU preflight validated seed forwarding before GPU launch. Two unmeasured clips
warmed both workers and the service before the fixed A -> B -> C order. Fifty
clips total were generated: two warmup and sixteen per measured arm. No model,
worker, reward or policy reload occurred between arms.

| Arm | Collection seconds | Generation seconds | Reward phase seconds | Native overlap seconds |
| --- | ---: | ---: | ---: | ---: |
| A: batched serial | 520.154111 | 496.502157 | 23.571573 | 0.0 |
| B: per-group serial | 518.685593 | 497.090159 | 21.515220 | 0.0 |
| C: per-group streaming | 508.315438 | 497.029815 | 21.223076 | 10.018006 |

Each arm's native counters confirm two groups and sixteen samples. A made one
reward call; B/C each made two. All three arms passed native trajectory structure
validation and finite-tensor/reward checks. Every stored trajectory tensor hash
matched exactly across both groups in A/B/C, and every reward matched exactly
(maximum difference zero). Both real workers passed final trained-weight content
readback at fleet-local version 1. These comparisons do not constitute a new
independent replay or optimizer-equivalence run for these seeds.

Streaming reduced this observed collection wall time by 11.838673 seconds versus
A, or 2.276% (1.02329x), and by 10.370155 seconds versus B. Native overlap measures
the complete score operation, including CPU artifact preparation and transport,
not ten seconds of overlapping GPU kernels. Two-second nvidia-smi snapshots
also captured GPU 1/2 at 100% while GPU 3 was at 70% and 58%, at 08:52:42 and
08:52:44 respectively. These utilization samples support concurrent device work
but do not precisely measure kernel overlap duration.

The score-phase composition explains the limited headroom: A's model inference
was 4.736752 seconds while MP4 materialization was 16.988891 seconds; B/C total
inference was 4.800866/4.827384 seconds and materialization 14.857914/14.545546
seconds. A/B already differ by 1.468518 seconds without any overlap. Fixed order,
one run per arm, artifact warmup and a lightweight utilization sampler during C
remain measurement limitations. No confidence interval, generation-p95 gate,
10% performance acceptance, quality or end-to-end training speedup is claimed.
The pilot does not justify automatically launching the much longer confidence
campaign; generation throughput and production training orchestration have
higher priority than repeating reward compatibility tests.

Evidence under cosmos_overlap_abc_real: executed_probe.py, executed_audit.py,
resolved_config.yaml, prompts.json, service.yaml/log, trained policy snapshot,
all three full rollout_batches.pt files and per-arm receipts, result.json,
audit.json and C_gpu_timeline.csv. Separate CPU preflight evidence is in
cosmos_overlap_abc_preflight. Driver, service cleanup, utilization monitor and
independent CPU audit all exited 0. Fresh compute and Ray/service inventories
were empty; all GPU claims released. No background training queue was started.
