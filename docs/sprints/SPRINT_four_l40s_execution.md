# Four L40S hardware execution

Status: active. Execute hardware workloads sequentially. A completed smoke run
does not establish learning quality, resume correctness, or another topology.

## Current machine (2026-09-11 UTC)

- Four NVIDIA L40S, each reporting 46,068 MiB; driver 580.173.02.
- Host RAM: 372 GiB. Mounted `/mnt/nvme`: 1.7 TiB, about 1.6 TiB free.
- Root filesystem has only about 20 GiB free; large new model downloads and
  artifacts should use the existing NVMe mount.
- Initial inspection found no GPU jobs. Old shell pollers were present, but
  their target training/generation processes were absent.

## Execution queue

The source sprints retain their complete acceptance criteria. This queue does
not replace them or narrow their scope; add newly discovered hardware gates.

1. `SPRINT_engine_worker_vocabulary.md` P6: SD3.5 N=1/N=2 output parity,
   throughput and peak memory, then online 2-engine/1-rank versus 1-engine/2-rank.
2. Audit the existing four-rank Wan FSDP smoke artifacts and exercise required
   checkpoint/resume gates (`done/SPRINT_multi_gpu_training.md`).
3. SD3.5 dedicated-rollout strict/continuous comparisons and repeatable recipe
   evidence (`planned/SPRINT_miles_recipe_evidence.md`,
   `parked/SPRINT_async_rollout_train_overlap.md`).
4. Real GPU weight-delivery verification and transport measurements
   (`planned/SPRINT_miles_weight_delivery_verification.md`).
5. Wan 2.1 I2V real distributed update and checkpoint/resume
   (`planned/SPRINT_wan_2_1_i2v_proof_run.md`).
6. Wan 2.2 dual-expert update, lifecycle evidence, and resume
   (`planned/SPRINT_wan_2_2_proof_run.md`). Its historical disk blocker must be
   rechecked against the already-mounted NVMe, without formatting any device.
7. Video training context parallelism numerical and memory gates
   (`parked/SPRINT_video_context_parallel.md`).
8. Cosmos Predict2.5 numerical gates and full paper-shaped workload
   (`parked/SPRINT_cosmos_predict25_rl_paper_parity.md`). Preserve its stated
   batch, frames, resolution, and update budget; smoke is not completion.
9. Reassess Qwen Image and Cosmos3 hardware-triggered work against actual model,
   dependency, and capacity prerequisites before launching their proof runs.

## Observations

- Previous SP baseline `outputs/sp_acceptance/n1.log` terminated at
  `rollout.startup.load_policy`: the 600-second RPC deadline expired. No output
  tensor or parity evidence was produced.
- Retry uses the existing CLI with the sole extra override
  `distributed.rollout.worker_rpc_timeout_s=1800`, matching the established
  Wan cold-start allowance. Seed 7 and the original four prompts are preserved.
  Log: `outputs/sp_acceptance/n1_retry.log`; target dump:
  `outputs/sp_acceptance/n1_retry/`. Completion remains unverified.
- Existing `outputs/wan_hpsv3_flash_grpo/fsdp_smoke_main/` contains epoch 0/1
  metrics, final checkpoint, and rank verdict files. Rank 0 reports success
  with world size 4; full artifact/resume audit remains pending.

## SP execution results (2026-09-12 UTC)

- N=1 retry completed: four 512x512 RGB uint8 images, generation 5.323 s,
  reported rank peak 16,790 MiB. Cold startup took 345.181 s.
- A live native stack located startup inside `cuMemcpyHtoDAsync_v2` while
  transferring the root-disk mmap of the transformer. GPU 0/1 independent
  256 MiB host-to-device probes took 0.178/0.208 s. Sequentially reading the
  4.939 GB mapped weight file took 188.144 s; model loading then completed.
  Future cold starts should use the existing NVMe model cache explicitly.
- N=2 completed: generation 8.121 s. Default BF16 parity **failed** at the
  unchanged 0.02 absolute threshold: max difference 0.482353, mean 0.007060,
  mismatch fraction 0.070023. See `outputs/sp_acceptance/compare_retry.json`.
  These are single-request observations, not a warmed throughput benchmark.
- N=1 repeat completed in 5.076 s and matched the first N=1 output exactly
  (`outputs/sp_acceptance/compare_n1_repeat.json`, tolerance zero).
- FP32/IEEE N=1/N=2 diagnosis is next. Passing a different precision does not
  close the default BF16 gate. Online topology comparison is still pending.
- The existing engine result aggregation returns the first rank's metrics;
  the acceptance harness therefore does not yet provide every rank's peak.
- Fixed a false-pass hole in dump comparison: empty/nonfinite outputs are
  rejected. Focused verification: 11 tests passed; touched-file Ruff passed.
- All four existing Wan FSDP rank verdicts report success. Final checkpoint
  metadata records `global_step=2`, `completed_epoch=2`, pinned Wan 1.3B model
  identity and a 378,397,551-byte checkpoint. Resume remains untested here.
- FP32 N=1 attempt `outputs/sp_acceptance/n1_fp32.log` exited with OOM while
  another session's rollout actor (PID 72587) occupied 16.38 GiB on the same
  card. This is resource contention, not an exclusive-capacity measurement.
  Do not rerun until the existing hardware job exits.
- Confirmed live at 00:03 UTC: driver PID 71799 runs
  `vrl.scripts.train --config experiment/sd3_5/online_grpo_ocr_continuous_4gpu_acceptance`
  with the 1800-second RPC override. Its three rollout actors occupy GPUs 1-3;
  its trainer uses GPU 0. Log:
  `outputs/sd3_5_ocr_continuous_4gpu_acceptance.launch.log`. This is related
  acceptance work started by another session; preserve it and inspect its
  process plus artifacts before launching the next GPU task. Its configured
  three epochs are scheduling acceptance, not the full learning A/B.

## Coordination (session vrl-74, 2026-09-12 00:05 PDT)

- `n1_retry` completed: dump at `outputs/sp_acceptance/n1_retry/` (4x3x512x512
  uint8, generate_wall 5.3 s, launch 345 s).
- Session vrl-74 launched the N=2 arm at 00:04 PDT: pid 67464, GPUs 1-2,
  `outputs/sp_acceptance/n2.log` -> `outputs/sp_acceptance/n2/`, same seed 7,
  same prompts, same `worker_rpc_timeout_s=1800`. Do not launch a second N=2.
- vrl-74 owns the GPU queue after N=2, in this order: `compare n1_retry n2`;
  `online_grpo_ocr_continuous_4gpu_acceptance` 3 epochs (continuous x streaming
  gap smoke); online 2x1 vs 1x2 small runs; then the sd3_5 dedicated 3x1
  strict / continuous / dynamic arms (`experiment/sd3_5/online_grpo_ocr_dedicated_3x1`);
  then wan physics strict vs continuous with the UnifiedReward service on GPU 3
  (weights already in the NVMe HF cache). Other sessions: please claim a stage
  here before launching anything on the GPUs.

## CPU engine acceptance follow-up (2026-09-12 UTC)

- This Codex session owns the CPU-only engine result aggregation fix, not the
  GPU queue reserved above. `execute_batch` now checks every rank's returned
  worker/request/batch identity and successful policy version, propagates typed
  nonprimary errors/stale-slot results, and retains every rank's metrics for
  `runtime_debug.ray_chunks`. Existing single-rank calls are unchanged.
- The nonprimary OOM regression exercises the production executor and verifies
  that both ranks retry each child batch, output sample coverage is complete,
  and both GPU identities/peak measurements appear in the final diagnostics.
  Focused CPU verification: 34 passed, one GPU test deselected. Real hardware
  telemetry verification remains part of the next coordinated P6 run.
- The existing continuous driver PID 71799 is now absent. Its log records
  completion at 00:04:21 UTC. Metrics contain three epochs, checkpoint metadata
  records `global_step=3`, and an artifact receipt exists. Epoch staleness is
  0/1/1. However, all three logged rewards and gradient norms are zero, so this
  establishes scheduling execution only, not learning quality or useful updates.
- Engine fix committed as `234c6abf`. Broader CPU Ray generation regression:
  227 passed, 23 GPU/slow tests deselected; touched-file Ruff and diff checks
  passed. Hardware execution was intentionally excluded from that test command.
- The queue owner is now running the real SD3.5 FP32 two-rank forward probe,
  confirmed by live driver PID 80505. Observe that process and its result before
  claiming additional GPUs; do not duplicate this diagnosis.

## Wan 2.2 preparation ownership

This Codex session owns downloading and validating the pinned Wan 2.2 T2V
dual-expert model on NVMe. This is CPU/network preparation only; the GPU queue
above is not claimed. Hugging Face metadata resolves revision
`5be7df9619b54f4e2667b2755bc6a756675b5cd7` to itself and reports 126,200,628,126
bytes for the repository. Cache destination: `/mnt/nvme/hf/huggingface/hub`.
Download log: `outputs/perf/wan22_model_download.log`. Do not duplicate download.

Download completed successfully. All 49 files passed size and digest checks
(SHA-256 for LFS files, Git blob SHA-1 for ordinary repository files), totaling
126,200,628,126 bytes. Each expert has 12 indexed shards and the text encoder
has three. Verification receipt: `outputs/perf/wan22_cache_verification.json`.
The storage/download blocker is resolved; the sprint moved to `planned/`.
Two-rank config resolution passed, but actual loading, gradients, boundary
parity, I2V proof and checkpoint/resume remain unverified. No GPU was used by
this preparation task.

### P6 results so far (vrl-74, 2026-09-12 00:20 PDT)

- N=1 vs N=2 decoded images (seed 7, 4 prompts, 512px, 10 steps, native
  sampler, bf16): PSNR 36.3 / 42.0 / 42.9 / 32.8 dB, mean abs diff 0.007,
  diff on high-frequency edges only (`outputs/sp_acceptance/n1_vs_n2.png`).
- N=1 self-repeat (`n1_rep`) is bit-exact against `n1_retry`, so that gap is
  not run-to-run noise.
- One-forward probe (`sequence_parallel_forward_probe`, real SD3.5 weights,
  2 ranks): float32 rel_l2 2.5e-7, bfloat16 rel_l2 3.4e-3. The
  small FP32 forward difference supports precision amplification as the
  cause of the image gap. This is a one-forward observation, not a completed
  FP32 denoise/image comparison. The original BF16 pixel gate remains FAILED
  at atol 0.02 with zero permitted mismatches; no replacement tolerance has
  been established. P6 numerical acceptance remains open.
- Throughput/memory at this geometry (4 single-sample batches): N=1 generate
  5.3 s, N=2 8.0 s; reported primary-rank peak about 16.8 GB in both. The
  original reports predate complete per-rank telemetry. Sequence parallel does not
  pay off for 512px single-sample batches (all-gather overhead dominates);
  the win, if any, needs long sequences (video) — recorded, not pursued here.
- continuous x streaming (Gap 3) smoke: `online_grpo_ocr_continuous_4gpu_acceptance`
  3 epochs, verdict success, prefetch engaged from step 1 (queue_wait 0 s).
- Online 2x1 vs 1x2: first 2x1 launch died from driver/worker import skew
  during a concurrent Codex edit to vrl/generation (not a repo bug); relaunched
  on the consistent tree at 00:18 PDT.

## Wan 2.1 I2V preparation ownership

This Codex session owns CPU/network preparation of the pinned Wan 2.1 I2V
checkpoint. The old root-disk snapshot contains only `model_index.json`, not
the weights. Repository metadata for revision
`b184e23a8a16b20f108f727c902e769e873ffc73` lists 46 files totaling
90,104,322,037 bytes. Download destination is the existing NVMe HF cache;
log: `outputs/perf/wan21_i2v_model_download.log`. Do not duplicate the download.
The GPU queue remains with vrl-74. Driver 86070 was confirmed live inside
trainer backward while its first epoch metrics had not yet been written.

The download completed. All 46 files totaling 90,104,322,037 bytes passed
size and repository-digest verification; transformer index has 14 shards,
text encoder index five. Receipt: `outputs/perf/wan21_i2v_cache_verification.json`.

Dataset readiness is not complete. The old local manifests contain only 7/2
rows and lack `report.json`; current split files require 309/35 rows. All 344
prompts match the official CSV (3,360 candidate rows). A full importer run into
`/mnt/nvme/data/external` decoded the first training reference, then exited 1
with HTTP 403 for the selected second video's Luma CDN URL (CSV row 2673,
caption `Hand flipping open book cover.`). Log:
`outputs/perf/videophy_i2v_setup.log`. There is no completed new manifest/report.
The official CSV contains alternative videos for this same caption on S3;
availability-aware selection must retain the actual chosen source identity
and avoid relabeling cached frames when retrying. Do not substitute synthetic
images or silently use the old partial manifests.

The 2x1 online driver 86070 has now exited. Its log ends before the first
optimizer update with replay parity `max_abs_diff=0.0148427`, limit `0.01`;
`metrics.csv` has only its header. This is a failed numerical gate, not a
completed five-epoch run. Preserve that limit and coordinate the next diagnosis
with the GPU queue owner.

## I2V dataset completion

The importer now tries the remaining officially listed videos for the same
caption after HTTP 403/404/410, preserving the existing quality ordering and
recording rejected sources. Other HTTP failures propagate. Cache filenames
bind the source URL and output dimensions; original frame dimensions are kept
in source sidecars so repeated imports cannot relabel a different cached video.
Eight focused tests passed, including recovered-source cache separation and
HTTP 503 fail-through; touched-file Ruff passed.

Full import into `/mnt/nvme/data/external/videophy_i2v` completed: 309 train,
35 eval rows, with 80/12 fallback selections. Every image was decoded and
checked for RGB/832x480, every caption/order/source row/URL was checked against
the source files, and train/eval caption intersection is empty. Digests are
recorded in `outputs/perf/videophy_i2v_verification.json`. The canonical I2V
recipe resolves its dataset plan as ready with explicit NVMe manifest/root/
source-report overrides (`outputs/perf/wan21_i2v_dataset_plan.json`). Independent
count/content checks above are necessary because the dataset-plan tool does
not infer expected row counts for absolute manifest paths.

The I2V proof sprint moved to `planned/`: weights and full data are ready;
real multi-rank training and checkpoint/resume are still pending the GPU queue.

## Next hardware slot: Wan I2V (Codex)

The SD3.5 single-GPU control driver 91845 is now absent. It failed before its
first update with replay parity 0.0173313 > 0.01 (log:
`outputs/sp_online/ctrl_1gpu_nocompile.launch.log`). Thus the observed parity
failure is not unique to the dedicated multi-GPU topology. All four GPUs were
observed idle after its exit.

Codex attempted to claim the next hardware slot for the prepared Wan I2V
two-rank FSDP proof on physical GPUs 0 and 1. This claim is now withdrawn:
the existing queue owner launched a compiled single-GPU control in the same
window. Use the pinned NVMe cache and verified full
VideoPhy dataset; run directory `outputs/wan_i2v_14b_l40s_proof/epoch1`, launch
log `outputs/perf/wan_i2v_l40s_epoch1.log`. This starts the real update gate;
it does not claim that training or resume has passed.

The prelaunch query showed a new competing driver (131204), but the launch
was incorrectly batched after that query before its result was inspected.
The I2V torchrun supervisor 132795 was explicitly terminated with SIGTERM;
its tool session exited 1 and both rank processes 132903/132904 are absent.
Cancellation occurred during checkpoint loading, before training. This is a
scheduling cancellation, not a model failure or memory-capacity result.
The compiled-control driver 131204 remains the queue owner's current job.
Do not infer a free execution slot from one idle-GPU observation; inspect the
preflight result before any dependent launch and wait for the queue release.

### Discovered hardware gates (vrl-74, 2026-09-12 00:35 PDT)

- **Replay parity gate fails for the SD3.5 OCR recipe at its production
  geometry on this stack**, independent of topology: 2 engines x 1 rank
  (`outputs/sp_online/2x1`, max_abs_diff 0.0148) and single-GPU colocated
  control (`outputs/sp_online/ctrl_1gpu_nocompile`, 0.0173) both exceed
  `trainer.replay_parity.max_abs_logprob_diff=0.01` with torch.compile off.
  The 128px/4-step continuous gate passes with 0.0. The May L4 run recorded a
  2-sample probe at 6e-7; the gate since b8c1f766 (2026-09-04) takes the max
  over the whole first replay. Under test now: compile on (recipe default,
  `ctrl_1gpu_compile`) and `actor.samples_per_replay_batch=16` matching the
  generation batch (`ctrl_1gpu_nocompile_rb16`).
- **Single-process launches only work when the intended GPU set is a prefix of
  the physical ids.** `CUDA_VISIBLE_DEVICES=3` (or `=1`) fails in
  `GlobalRayPlacementOwner.assign_roles` with "rollout device GPU 0 has no
  bundle in the probed placement group (probed GPUs=[3])"; pinning
  `distributed.resources.visible_devices=[1]` with all GPUs visible fails the
  mirror way ("GPU 1 ... probed GPUs=[0]"). The resolver's device ordinals and
  Ray's probed ids live in different id spaces once the visible set is not
  `0..k-1`. The torchrun path avoids it by remapping per rank explicitly.
  Not fixed here; it blocks running two single-GPU experiments side by side
  on GPUs other than 0.

### Masked GPU placement fix staged separately (Codex, 2026-09-12 00:44 UTC)

Commit `2ca99be6` in isolated worktree `/home/ubuntu/VRL-gpu-placement`
preserves integer `CUDA_VISIBLE_DEVICES` IDs and ordering during automatic
resource resolution and translates the trainer device back to its local Torch
ordinal. It is deliberately not applied to the shared runtime while the
compiled-control job is running. The focused resource/placement suite passed
85 tests (3 deselected); touched-file Ruff and formatting checks passed.

A private, metadata-only Ray probe with `CUDA_VISIBLE_DEVICES=3` succeeded:
`visible=[3]`, `trainer=cuda:0`, `probed_gpus={0: 3}`,
`expected_ray_ids=[3]`, rollout bundle `[0]`. The driver asserted CUDA remained
uninitialized, and the probe exited 0 after owner and cluster shutdown. No
model was loaded. This validates the masked-ID placement path, not a complete
training run. The separate explicit-subset case with all GPUs visible remains
open; the fix must not be treated as closing that scheduler constraint.

### Explicit GPU subset fix and compiled-control result (Codex, 2026-09-12 UTC)

The compiled single-GPU control driver 131204 is terminal. Its launch log
ends with first-update replay parity failure: finite=True,
`max_abs_diff=0.0155535 > 0.01`. Compilation therefore did not resolve this
gate; no optimizer update is established by that run. No training/torchrun
driver was found in the subsequent process check. This is not a release of
the queue owner's remaining stages.

Commit `254be02d`, on top of `2ca99be6` in
`/home/ubuntu/VRL-gpu-placement`, addresses the separate explicit-subset case.
Owned local Ray nodes inherit only the placement layout's actor GPU IDs;
the driver mask is restored even if initialization fails. The node cannot
expand the inherited mask. Attached and preinitialized clusters retain their
existing ownership and device view. Both commits remain isolated from the
shared training worktree, pending a coordinated runtime update.

Validation: 109 focused CPU tests passed (21 deselected), touched-file Ruff,
formatting and diff checks passed. The real cluster ownership suite passed
13 tests before adding the second GPU parameter and mask-boundary guard;
both additions were subsequently exercised by the focused runs. Metadata-only
real Ray probes passed for explicit subsets `(3,)` and `(3, 1)` with no driver
mask: probed bundles were `{0: 3}` and `{0: 3, 1: 1}` respectively, the trainer
remained `cuda:3`, and CUDA initialization state was unchanged. Both private
clusters shut down and the test process exited 0. These are scheduler and
lifecycle results, not full online training, weight-sync, or resume evidence.

### Current hardware claim: Wan I2V two-rank proof (Codex)

Following the compiled control's terminal failure and repeated checks with no
live training/torchrun drivers, Codex claims the next two-GPU proof window on
physical GPUs 2 and 3. Please do not launch overlapping hardware stages during
this claim. The run uses the isolated `/home/ubuntu/VRL-gpu-placement` tree at
`254be02d`, verified NVMe weights/data, and the canonical two-rank I2V FSDP
recipe. Output target: `outputs/wan_i2v_14b_l40s_proof/epoch1_retry` in the main
workspace. A separate GPU/process preflight must pass before launch. This
claim is a queued attempt, not evidence of training success.

The `epoch1_retry` attempt exited 1 before generation: both ranks initialized
the real model and FSDP, then initial rollout weight export called
`DTensor.full_tensor()` on CPU-offloaded shards belonging to a CUDA/NCCL mesh.
The error was `No backend type associated with device type cpu`. This was not
OOM or a placement failure, and no optimizer update occurred. Both rank
processes and their GPU allocations were gone after terminal cleanup.

Isolated commit `0235947e` stages one selected offloaded DTensor at a time on
the current CUDA device for its full gather, retaining CPU snapshots without
gathering the frozen base. The expanded real Wan dual-expert CPU-offload tests
passed for one and two CUDA ranks (initial sync, nonzero expert gradients,
checkpoint and optimizer export); 57 FSDP CPU/distributed tests also passed.
Touched-file Ruff, format and diff checks passed.

After a separate clean GPU/process preflight, the same canonical production
I2V recipe restarted on GPUs 2/3 from `0235947e`. New output directory:
`outputs/wan_i2v_14b_l40s_proof/epoch1_gather_fix`; log:
`outputs/perf/wan_i2v_l40s_epoch1_gather_fix.log`. The claim remains active
through this retry. Full production update/resume evidence is still pending.

The `epoch1_gather_fix` retry exited 1 after crossing the previous gather
failure and loading both rollout pipelines. Initial weight installation then
failed in `pipeline.remove_all_hooks()`: Accelerate's
`remove_hook_from_module` uses delegated `hasattr(_hf_hook)` on the PEFT
`LoraModel` wrapper and calls the underlying hook with that wrapper, which has
no directly registered `scale_shift_table` parameter. The raised Wan error
correctly marks the pipeline unusable rather than continuing with partial
hooks. This is a separate hook-ownership failure, not an OOM or a successful
update. No sample, optimizer step, or checkpoint/resume acceptance was produced.

The torchrun session is terminal (exit 1), and subsequent process and GPU
queries found no training drivers or allocations. The hardware claim is now
released while this hook-ownership path is diagnosed in the isolated tree.
Next I2V retry must follow a new preflight; do not report this stopped job as
still running or duplicate it based on an observation timeout.

### Current claim: I2V hook-ownership retry (Codex)

Isolated commit `195cfc14` removes explicitly owned transformer hooks
child-first before public pipeline cleanup, avoiding PEFT's delegated hook
lookup. Adding a root `scale_shift_table` parameter to the real Accelerate/PEFT
offload regression reproduced the production failure before the fix; all 25
loading/offload tests then passed. Touched-file Ruff and formatting passed.
Codex claims GPUs 2/3 for the same canonical two-rank production proof after
another separate preflight. Target: `outputs/wan_i2v_14b_l40s_proof/epoch1_hook_fix`,
log `outputs/perf/wan_i2v_l40s_epoch1_hook_fix.log`. No completion is implied.

The complete Wan family CPU regression passed 40 tests. The production
`epoch1_hook_fix` run crossed hook removal and weight installation, then exited
1 on `NameError: ftfy is not defined` in Diffusers' Wan prompt cleanup. The
environment lacked an already-declared dependency; installed `ftfy==6.3.1`
from the repository lock and verified the actual `prompt_clean` function in
a fresh process. No source dependency or sampling configuration was changed.
No training process or GPU allocation remained after cleanup. After a separate
preflight, the claim continues for output `outputs/wan_i2v_14b_l40s_proof/epoch1_ftfy`
and log `outputs/perf/wan_i2v_l40s_epoch1_ftfy.log`, on the same isolated commit.

The `epoch1_ftfy` run completed two generated samples per rank (35.747/35.866 s
generation wall time). Thus real-model initial weight installation, prompt
cleanup, conditioning, denoise and decode crossed the earlier failures.
It then exited 1 at the strict rollout-to-trainer GPU parking check:
worker physical residual 2,657,091,584 bytes, pre-load baseline 446,693,376
bytes, allowed excess 268,435,456 bytes. No backward/optimizer update or
checkpoint was reached. The 378 MB batch allocator peak is not a proof of
total physical release; NVML accounts for process-owned CUDA memory outside
that allocator too. Keep the parking threshold unchanged and diagnose both
live tensors and runtime/workspace allocations before the next retry.

The torchrun tool session exited 1, no training driver remains, and all four
GPUs reported 0 MiB after cleanup. The hardware claim is released during CPU
diagnosis. The isolated hook fix remains at `195cfc14`; no shared runtime
source was modified and no training acceptance gate is marked complete.

### Current claim: I2V parking diagnostics (Codex)

Codex claims GPUs 2/3 for one canonical two-rank rerun with failure-only
allocator/component residency diagnostics in the isolated worktree. The
parking threshold and workload are unchanged; 44 focused worker tests passed.
Output: `outputs/wan_i2v_14b_l40s_proof/epoch1_parking_diag`, log:
`outputs/perf/wan_i2v_l40s_epoch1_parking_diag.log`. Launch follows a separate
process/GPU preflight; the claim does not establish update or resume success.

The diagnostic run (`fe6a1b0e`) exited 1 after successful generation, reproducing
the exact 2,657,091,584-byte physical residual. Both worker diagnostics agree:
Torch allocated=9,719,296 bytes, reserved=2,105,540,608 bytes; component CUDA
parameter/buffer bytes: VAE=0, text_encoder=0, image_encoder=2,056,
transformer=1,048,576. These observations narrow the failure to allocator
reservation retention with small live allocations, not full model weights.
They do not yet identify which live allocation prevents segment release.
Inspect segment/block residency and the surviving buffers/runtime workspaces;
do not increase the physical parking allowance or call this gate passed.

The diagnostic helper is failure-only and cannot replace the original error;
44 worker tests passed, touched-file Ruff/format checks passed. The torchrun
session is terminal with exit 1. Post-exit process and GPU allocation checks
are empty. This hardware claim is released pending the next targeted probe.

### Current claim: I2V BLAS-workspace parking fix (Codex)

A standalone CUDA probe reproduced the retention: a live 8,519,680-byte BLAS
workspace pinned a 2,147,483,648-byte allocator segment after empty_cache;
clearing the idle workspace released the entire segment. The GPU regression
also verifies a subsequent matrix multiplication recreates working state.
CPU-offload handoff now opts into this cleanup; CuMem and other callers retain
the existing default. One GPU regression and 64 CPU/worker/reward tests passed.
The physical parking threshold is unchanged.

Codex claims GPUs 2/3 for the canonical two-rank I2V retry from the isolated
worktree after a separate preflight. Output:
`outputs/wan_i2v_14b_l40s_proof/epoch1_blas_fix`; log:
`outputs/perf/wan_i2v_l40s_epoch1_blas_fix.log`. Production validation remains open.

The real two-rank retry from `d9c66e4d` passed GPU parking on both workers:
physical residual=555,745,280 bytes, baseline=446,693,376 bytes, excess about
104 MiB, below the unchanged 256 MiB limit. This confirms idle BLAS workspace
retention was the large parking blocker on the production I2V path.

The run then entered trainer replay and exited 1 before the first optimizer
update: Diffusers Wan attention's `apply_rotary_emb` multiplies CUDA query/key
tensors with CPU rotary cosine/sine tensors. Inspect the CPU-built FSDP replay
model's rotary buffers and their device movement; do not bypass replay parity
or change the workload to avoid it. Checkpoint/resume remains untested.

The tool session is terminal with exit 1, post-exit training-process/GPU
allocation queries are empty, and this hardware claim is released while the
new replay-device failure is diagnosed. No full I2V training pass is claimed.

### Current claim: I2V mixed-residency restore fix (Codex)

Trainer parking recorded one device per module from its first parameter.
CPU-offloaded FSDP parameters live on CPU while rotary buffers live on CUDA;
restoring everything to that one CPU device lost the buffer placement.
The isolated fix records named parameter/buffer devices and restores current
objects by name (Module.to can replace buffers). One- and two-rank CUDA Wan
tests now park/restore before both expert backwards, verifying CUDA buffers
and CPU parameter shards survive. Both tests passed; 60 strategy/FSDP CPU
tests and touched-file Ruff/format checks passed.

Codex claims GPUs 2/3 for the same production recipe after a separate preflight.
Output `outputs/wan_i2v_14b_l40s_proof/epoch1_restore_fix`; log
`outputs/perf/wan_i2v_l40s_epoch1_restore_fix.log`. No training pass is implied.

The production retry from isolated commit `f01c3625` completed successfully
(torchrun exit 0; both rank verdicts success, world_size=2). The real I2V
generate -> park -> replay -> backward -> optimizer -> checkpoint gate now
has evidence. Full-precision epoch metrics: reward_mean=0.8894408941,
reward_std=0.1914940476, grad_norm=0.0601723505, zero-advantage rate=0,
pre-update max logprob difference=0.0000662058592 (unchanged limit 0.01).
Both workers passed physical parking at residual=555,745,280 bytes.

Final checkpoint metadata records global_step=1, next_epoch=1, next_step=1,
pinned production I2V identity and checkpoint size 2,098,666,415 bytes.
`TrainingCheckpoint.load` validated the actual checkpoint and its progress;
both checkpoint-1 and checkpoint-final exist. This is one small-geometry
real update with CPU motion reward, not the full physics-reward learning gate
or a resume pass. Next required hardware steps are strict resume to step 2
and comparison with an uninterrupted two-update control, then remaining
source-sprint quality/reward gates.

The job has exited and no training process or GPU allocation remains. This
hardware claim is released. Root storage fell to about 13 GiB free while
writing the two checkpoints; future run output directories must use the
existing NVMe mount rather than accumulating more root-disk checkpoints.

### Current claim: I2V strict resume (Codex)

Codex claims GPUs 2/3 for strict resume from
`outputs/wan_i2v_14b_l40s_proof/epoch1_restore_fix/checkpoint-final` to
`trainer.total_epochs=2`, preserving the canonical recipe, pinned model and
verified manifests. Isolated runtime remains `f01c3625`. Output:
`/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/resume_step2`; log:
`outputs/perf/wan_i2v_l40s_resume_step2.log`. Separate GPU/process preflight
precedes launch. Resume success and uninterrupted equivalence remain unproven.

Strict resume completed with torchrun exit 0 and success verdicts from both
ranks (world_size=2). `TrainingCheckpoint.load` validated the actual final
checkpoint with next_step=2 and next_epoch=2. All 800 transformer LoRA tensors
are finite and all differ from the step-1 checkpoint, proving a continued
parameter update. Full-precision resumed metrics: grad_norm=0.0220371052,
pre-update max logprob error=0.0000896602869, reward_mean=1.0,
reward_std=0.0. The lack of reward variation means this update is not learning
quality evidence; the recipe also has a reference-KL objective.

This establishes strict checkpoint loading and continued two-rank training,
but not equivalence to uninterrupted training. The next required run is a
fresh canonical two-update control, comparing its step-1 and step-2 model,
optimizer, progress and RNG/sampler state against the split run. Use NVMe
for both outputs and Ray temporary files. No I2V training process or GPU
allocation remained after this run; the hardware claim is released.

### Current claim: I2V uninterrupted control (Codex)

Codex claims GPUs 2/3 for a fresh canonical two-update control from isolated
`f01c3625`, with the same pinned model, dataset and seed as the split run.
Output `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/control_step2`; log
`outputs/perf/wan_i2v_l40s_control_step2.log`; Ray temporary root on NVMe.
Separate GPU/process preflight precedes launch. Compare both intermediate
and final checkpoints with the first-update and resumed artifacts; matching
successful exit codes alone do not prove numerical equivalence.

The uninterrupted control completed with exit 0 and two success rank verdicts.
Both updates had finite nonzero gradients (0.0959043399 / 0.8270630873), and
first-replay errors remained below 0.01. Ray temporary files and checkpoints
were written on NVMe. This adds continuous two-update execution evidence.

Numerical split/control equivalence did NOT pass. The actual checkpoint
comparison is `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/split_control_comparison.json`,
with its executable `compare_split_control.py` alongside it. It checks model,
trainer/optimizer/EMA, progress and RNG payloads and rejects nonfinite values.
Step 1 already differs in 400 model tensor leaves and 1,600 trainer leaves,
while progress and stored RNG match; step 2 differs in 800 model and 3,200
trainer leaves, with one RNG leaf difference too. These are not equivalent
starting trajectories, so the final differences cannot isolate resume behavior.

The resolved configurations differ only in total epochs/output directory,
but neither specifies `sampling.seed`: the diffusion request parser forwards
`sampling.get("seed")`, independently of trainer RNG. Thus the claim of a
controlled stochastic trajectory was too strong. Next run: explicitly seeded
two-update control, then resume its own published checkpoint-1 to step 2 and
compare all states. Keep the original recipe geometry/objectives; explicit
sampling control is for this numerical audit, not a substitute quality test.
The trainer RNG difference also needs inspection in that controlled audit.

The control process is terminal and post-exit process/GPU queries are empty.
This hardware claim is released. Resume equivalence remains open.

### Current claim: seeded I2V equivalence control (Codex)

Source inspection confirms `sampling.seed` controls I2V initial latents,
per-sample denoise SDE generators and stochastic-window selection. Codex
claims GPUs 2/3 for the canonical two-update control with `sampling.seed=7`,
otherwise unchanged, from isolated `f01c3625`. Output:
`/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/control_seed7`; log:
`outputs/perf/wan_i2v_l40s_control_seed7.log`. Ray temporary files use NVMe.
After completion, resume this control's own checkpoint-1 for a matched branch.
Launch follows a separate process/GPU preflight; equivalence is still unproven.

While the seeded control runs, an independent CPU-only probe of the actual
`MotionDynamicsModel._module_for_inference()` confirmed that first lazy load
changes Torch CPU RNG state (2,494 state bytes changed after seed 42). The
base `LazyTorchModule` directly calls `_load_module`; no RNG guard surrounds
RAFT construction. This is a concrete resume-lifecycle asymmetry: the
uninterrupted second update reuses the reward model while the resumed process
loads it anew after trainer RNG restoration. Runtime source remains unchanged
during the current job. Fix/verify this effect before claiming full RNG
equivalence; changing only `sampling.seed` does not isolate trainer RNG.

The seeded control completed with exit 0 and both rank verdicts success, but
both logged gradient norms are zero. Reward standard deviations are 0 and
0.0329200812; pre-update replay errors are 0.0000657960773 and 0.0000554621220.
This is not a meaningful parameter-update equivalence control. Do not accept
unchanged checkpoints as evidence that trained-state resume is correct.

After the job exited, isolated commit `63ff171a` wrapped RAFT's CPU weight
construction in a CPU-only Torch RNG fork. Nine focused reward tests passed
(one optional quality test skipped), including successful/failed lazy load
and cache reuse; the real cached RAFT probe now preserves CPU RNG exactly.
Touched-file Ruff and formatting passed. GPU/process queries are clear.

Additional source diagnosis: `FullSequenceDenoiseBatchExecutor` passes the
unchanged request seed to `model.prepare_sampling` for every sample batch,
while the denoise loop offsets its separate SDE generator by sample_start.
Thus the two one-sample batches reuse the initial latent seed. The resolved
SDE window_size is zero, so a randomly selected terminal window is not the
explanation. Fix and test deterministic per-sample initial-noise semantics
before selecting the next nondegenerate seeded control. Also audit per-rank
checkpoint RNG ownership: the runner initializes rank-distinct streams but
the current primary-only checkpoint publication stores the primary RNG tree.

This hardware claim is released. Seeded resume equivalence and learning
quality remain open; no zero-gradient result is treated as completion.

### Initial latent seed offset fix (Codex)

Isolated runtime commit `99633226` offsets the request seed passed to
`model.prepare_sampling` by the denoise batch's `sample_start`. It uses a
replacement request, leaving the original request and loop seed unchanged,
so the existing SDE generator offset is not applied twice. This prevents
identical initial noise in successive one-sample native batches with a fixed
seed. Unseeded requests retain their existing behavior.

The full-sequence binding and denoise-step suites passed: 154 tests. New
coverage checks the initial batch, later batch, deterministic repeat,
unseeded requests, unchanged input/config seeds, and prepare keyword
forwarding. Touched-file Ruff/format and diff checks passed. This establishes
batch-start seed semantics, not arbitrary batch-partition invariance and not
a hardware learning or resume acceptance result.

Before another expensive equivalence run, fix the confirmed per-rank RNG
checkpoint gap: `save_training_checkpoint` publishes only the primary
process RNG tree, while `scripts/common/online.py` restores that tree on
every rank after initializing rank-distinct streams. Audit strategy-owned
collectives, legacy checkpoint compatibility, and topology validation, then
test distinct rank streams across save/resume. Only after runtime fixes are
frozen should the seeded, nonzero-update GPU control and matched resume run.
No GPU job was launched for this CPU-only change; the hardware claim remains
released. All broader acceptance gates remain as recorded above.

### Per-rank checkpoint RNG ownership fix (Codex)

Isolated runtime commit `6bc896f0` gathers every training rank's RNG tree
through the strategy's CPU coordination group before primary-only checkpoint
publication. Multi-rank checkpoints now store `rng.world_size` and ordered
`rng.by_rank`; the online runner restores its own entry. RNG capture joins
the existing setup-failure agreement before any later collectives. The
initial-noise and reward-construction fixes remain in this runtime lineage.

Strict multi-rank resume rejects legacy primary-only RNG trees. Non-strict
legacy resume emits an explicit nonequivalence warning; single-process legacy
restore remains supported. Rank bounds, malformed rank lists and changed
world sizes fail before RNG mutation, including in non-strict mode. Thus the
older I2V checkpoint artifacts remain historical update/resume evidence but
cannot satisfy the new strict per-rank RNG contract.

Verification: 191 tests passed, 5 skipped, with CUDA hidden. The new real
two-process Gloo test writes and loads the actual checkpoint, proves the two
saved Torch states differ, and reproduces each rank's subsequent Torch,
named prompt-generator, Python and NumPy random draws exactly. Checkpoint
publication failure, strategy/MRO and online lifecycle suites also passed.
Touched-file Ruff/format and diff checks passed. This is CPU distributed
evidence, not CUDA RNG/NCCL or full trained-state equivalence acceptance.

Preflight found a separate main-worktree SD3.5 job (PID 192781,
`outputs/sp_online/ctrl_1gpu_nocompile_gb1`) running on GPU 0; GPUs 1/2/3
reported idle. No GPU workload was launched or interrupted by this change,
and no shared runtime source was edited. Next: fresh hardware preflight,
CUDA/NCCL per-rank RNG round trip on free GPUs, then a newly seeded I2V
nonzero-update control and strict resume from its own new-format checkpoint.
All full-workload, learning-quality and remaining family gates stay open.

### Current claim: CUDA per-rank RNG checkpoint test (Codex)

Codex claims GPUs 2/3 for the real two-process CUDA RNG checkpoint test from
isolated runtime `6bc896f0` with test-only extensions. It initializes NCCL and
the CPU coordination subgroup through production code, writes/loads the
checkpoint, and checks subsequent random draws on every visible CUDA device
for each rank. GPU 0's separate SD3.5 process is not touched. No production
training acceptance is inferred from this bounded checkpoint test.

The CUDA test process exited 0: 14 tests passed in 6.74 s, including real
Gloo and NCCL two-rank checkpoint round trips. Log:
`outputs/perf/checkpoint_rng_cuda_2rank.log`. Test commit `5f3d0d50`.
Post-exit GPUs 2/3 are clear. The bounded test claim is released.

### Current claim: corrected seeded I2V control (Codex)

Codex claims GPUs 2/3 for two canonical I2V updates from frozen isolated
runtime `5f3d0d50`, including initial latent seed offset, reward RNG isolation
and per-rank checkpoint RNG fixes. Set `sampling.seed=7`, total_epochs=2,
and the verified full VideoPhy I2V manifests; keep other recipe settings.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/control_seed7_rank_rng`.
Log: `outputs/perf/wan_i2v_l40s_control_seed7_rank_rng.log`.
Preflight: GPUs 2/3 idle, 334 GiB host memory available, 1.3 TiB NVMe free;
the separate GPU 0 job remains untouched. Acceptance requires actual nonzero
updates before this branch can serve as a trained-state equivalence control.

Control completed with torchrun exit 0 and success verdicts on both ranks.
Step 1 still saturates the motion reward (mean 1/std 0), gradient 0. Step 2
has reward mean 0.7610435486/std 0.3768313527 and nonzero gradient norm
0.27305263635055477. Pre-update max logprob errors are 0.0000657960773 and
0.0000565871596, below the unchanged 0.01 limit. Actual checkpoint loading
validated next_epoch/next_step=2, all 800 LoRA tensors finite, 400 tensors
changed from checkpoint-1 to final, and two distinct rank CPU RNG trees.
This control can test a nonzero second update across resume, although its
first update is degenerate and it does not establish learning quality.
The process is terminal; GPU/process preflight is clear on all four cards.
The control claim is released.

### Current claim: matched strict I2V resume (Codex)

Codex claims GPUs 2/3 for strict resume from
`control_seed7_rank_rng/checkpoint-1` to total_epochs=2, with all control
settings and frozen runtime `5f3d0d50` unchanged except resume/output fields.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/resume_seed7_rank_rng`.
Log: `outputs/perf/wan_i2v_l40s_resume_seed7_rank_rng.log`.
Compare final model, trainer/optimizer/EMA, progress and every rank's RNG
against the uninterrupted control, not just process success or rank 0 RNG.

The matched resume exited 0 with both rank verdicts success. Its reward mean
0.7610435486, reward std 0.3768313527 and max pre-update logprob difference
0.0000565871596 exactly match control step 2, but gradient norm is
0.27392399005551543 versus control 0.27305263635055477. Resolved configuration
comparison confirms only output_dir and added resume_from/resume_strict differ.

Full checkpoint comparison FAILED exact trained-state equivalence. Every
rank's RNG tree (1,273 leaves) and progress match exactly, as do family/schema;
395 model tensor leaves and 1,580 trainer leaves differ. First differing model
leaves are LoRA B weights, with examples around 7.4e-5 maximum absolute error.
Model comparison covers 131,072,000 elements; trainer comparison covers
524,288,800 elements, including optimizer/master/EMA state. No relaxed
tolerance is substituted for the exact comparison.

Artifacts: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/seeded_resume_comparison.json`
and `.log`, generated by `compare_seeded_resume.py` alongside the validated
recursive comparator. This controlled hardware result verifies the per-rank
RNG correction but leaves gradient/model/optimizer equivalence unresolved.
Next isolate the restored model/master/optimizer state before the second
backward pass and distinguish state restoration from numerical repeatability.
Do not rerun unchanged jobs or mark the I2V quality/physics gates complete.
Both jobs are terminal, process query is empty and all four GPUs are clear.
The matched-resume hardware claim is released.

### Current claim: FP32-master restore isolation (Codex)

The actual control checkpoint-1 has no model/master rounding disagreements
across 800 parameter bindings, zero Adam first/second moments, and optimizer
step=1. This rules out malformed saved master rounding as an explanation,
but does not yet prove the resumed runtime state.

Codex claims GPUs 2/3 for extended real two-rank BF16/FP32-master checkpoint
tests, covering CUDA with and without CPU offload. Each branch starts with a
zero-gradient update, saves/restores model and optimizer state exactly, then
checks a nonzero second update for exact full optimizer/model equivalence.
Runtime remains `5f3d0d50`; only the diagnostic test is extended. Fresh
preflight found no training/Ray processes and all four GPUs clear.

Isolated test commit `26faff39` completed this diagnostic: 4 tests passed in
19.64 s with `CUDA_VISIBLE_DEVICES=2,3 pytest --distributed`, including real
two-rank CUDA with CPU offload both disabled and enabled. Both CUDA branches
verify finite zero first-step gradients and finite nonzero second-step
gradients, exact restored model/master/moment/group state before the second
step, and exact optimizer/model state after the second step. Touched-file
Ruff/format and diff checks passed. Log:
`outputs/perf/fsdp_master_zero_step_resume_verified.log`.

Two earlier diagnostic harness errors were corrected before this result:
Torch's nested assert_close cannot compare optimizer-type strings, and raw
DTensor.full_tensor on a CPU-offloaded shard cannot use an NCCL-only mesh.
The final harness handles scalar metadata explicitly, uses production state
gathering for full-state checks, and compares local master shards on every
rank. GPU tests require the explicit --distributed switch; the initial
default invocation's skips are not GPU acceptance evidence. Production
runtime code was not changed by this diagnostic.

The simple BF16/FP32-master CUDA path did not reproduce the production Wan
resume discrepancy. This is not proof that every optimizer restoration path
is correct. Next extend the isolation to Wan LoRA, activation checkpointing,
and parking/restoration, then inspect production-scale pre-backward state if
needed. Existing training_debug records also show the same rank-local
trainable SHA256 (`2669ebe8410f1e50004356ff672c6b77fe3f17c461a7786abdfe73a44f319a52`)
at the control's initial gate and resumed step-1 gate; the control's step-1
gate was not separately captured, so this does not prove equality of every
runtime state at the matched boundary. Exact Wan equivalence remains open.
The test process is terminal and GPU memory queries are clear. This claim
is released; no expensive production rerun was launched unchanged.

### Current claim: Wan-specific master resume isolation (Codex)

Codex claims GPUs 2/3 for tiny real Wan I2V BF16 LoRA/FSDP tests, with FP32
masters, CPU offload, parking/restoration and activation checkpointing off/on.
After a zero-gradient first step, serialize/reload model and optimizer, check
exact restored state, and compare predictions, nonzero gradients and final
state for the second update. Runtime is unchanged from `26faff39`; only the
existing Wan test is extended. Fresh preflight found all GPUs clear and no
training/Ray/pytest processes. This is diagnostic, not the 14B acceptance gate.

Both branches passed (2 tests, 19.00 s), isolated test commit `9a2b01d2`.
Log: `outputs/perf/wan_bf16_master_resume_isolation.log`. Exact comparisons
cover restored model/optimizer, second-step predictions and nonzero gradients,
and final model/optimizer with checkpointing disabled/enabled. The production
Wan forward checks gradient-enabled plus its checkpointing flag; the enabled
branch exercises that path. The tiny model still cannot exclude scale-specific
kernel behavior or full online lifecycle differences. Test process is terminal;
post-exit preflight is clear. This diagnostic claim is released.

### Current claim: independent I2V resume repeatability (Codex)

Codex claims GPUs 2/3 for one independent strict resume from the same
`control_seed7_rank_rng/checkpoint-1`, with the same seed and configuration
as `resume_seed7_rank_rng`, output `resume_seed7_rank_rng_repeat` under the
same NVMe root. Log: `outputs/perf/wan_i2v_l40s_resume_seed7_rank_rng_repeat.log`.
Runtime source is unchanged; the intervening commits only add tests. This
repeat has a specific diagnostic purpose: compare the two cold resumes to
separate warm-vs-resumed state differences from production-scale numerical
nonrepeatability before adding instrumentation or changing runtime behavior.
No relaxed tolerance or zero-update result will count as exact equivalence.

The independent resume completed with exit 0 and both rank verdicts success.
Reward mean/std and pre-update max error match the first resume exactly, but
gradient norm is 0.27371560909433384 versus 0.27392399005551543. The full
comparison again differs in 395 model and 1,580 trainer leaves, while both
rank RNG trees and progress are exact. Even the first replay-gate debug
record, including the rank-local trainable fingerprint, is identical.
Thus warm-vs-resumed state alone cannot explain the observed discrepancies:
independent cold resumes from the same checkpoint also fail exact repeatability.
This does not yet identify a particular kernel or exclude unrecorded state.

Evidence: `resume_repeatability_comparison.json` and `.log` beside the prior
comparison artifacts on NVMe. `compare_seeded_resume.py` now accepts explicit
left/right/output arguments so the original control comparison is retained.
The process is terminal, process query empty and GPUs clear; claim released.

Source inspection found an existing strict deterministic training policy in
`OnlineRunConfig.initialize_process_rng`: `trainer.deterministic=true` enables
deterministic algorithms with warn_only=False and deterministic cuDNN settings.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launch because CUDA is initialized
before the runner's second RNG initialization. Previous controls/resumes did
not enable this option. Next test strict deterministic production repeatability
with the existing option before adding new runtime mechanisms. Rollout/reward
process determinism remains a separate consideration; this policy explicitly
owns trainer processes only. Exact resume and all broader quality gates remain
open, and no tolerance was relaxed.

### Current claim: strict deterministic I2V resume (Codex)

Codex claims GPUs 2/3 for strict resume from
`control_seed7_rank_rng/checkpoint-1`, now adding `trainer.deterministic=true`
and launch-time `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Other model/data/sampling
and update settings are unchanged. Isolated runtime `9a2b01d2` is frozen.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/resume_seed7_deterministic`.
Log: `outputs/perf/wan_i2v_l40s_resume_seed7_deterministic.log`.
Fresh preflight found all GPUs clear and no training/Ray/pytest processes.
This strict mode must either complete or expose an unsupported nondeterministic
operation; do not downgrade to warn-only to make the diagnostic pass.

First strict-mode run exited 0 with success verdicts on both ranks. Launch
evidence confirms deterministic_algorithms=true, warn_only=false,
cudnn_deterministic=true, cudnn_benchmark=false and the expected cuBLAS
workspace environment. Reward mean/std and replay error match previous runs;
gradient norm is 0.2728397151080096. The run completed without a deterministic
algorithm exception. This alone does not establish repeatability. Process/GPU
preflight is clear; the first-run claim is released.

Codex now claims GPUs 2/3 for the identical strict-mode independent repeat,
changing only output to `resume_seed7_deterministic_repeat` in the same NVMe
root, log `outputs/perf/wan_i2v_l40s_resume_seed7_deterministic_repeat.log`.
Compare complete final checkpoint payloads before declaring repeatability.

The strict deterministic repeat completed with exit 0 and success verdicts
on both ranks. Launch evidence again confirms strict deterministic settings.
Both runs have exactly the same nonzero gradient norm 0.2728397151080096,
reward mean 0.7610435486/std 0.3768313527 and pre-update max error
0.0000565871596. Full checkpoint comparison PASSED exact repeatability:
zero mismatches in model (131,072,000 elements), trainer/optimizer/EMA
(524,288,800 elements), both rank RNG trees (1,273 leaves), progress,
family and schema. No comparison tolerance was relaxed.

Evidence: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/deterministic_repeatability_comparison.json`
and `.log`, produced with the existing comparator's explicit branch arguments.
These runs establish deterministic cold-resume repeatability, not yet the
uninterrupted-vs-resumed gate: previous uninterrupted control used default
nondeterministic training. Next run the two-update continuous control with
the same strict deterministic flags; compare its checkpoint-1 and final
against the tested resume source and outputs. If checkpoint-1 differs, use
the new control's own checkpoint for a fresh matched resume. Retain nonzero
second-update and exact full-state requirements; first-step saturation and
the separate quality/physics gates remain open.

Both processes are terminal, process queries empty and all GPUs clear. The
repeat hardware claim is released. No runtime code change was needed for this
result; the existing deterministic training option was used as designed.
