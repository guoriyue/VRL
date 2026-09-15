# Four L40S hardware execution

## Current Handoff (2026-09-13)

USER PAUSE: Wan was intentionally cancelled at the user's request on 2026-09-13
around 22:16 UTC to release hardware for another agent's cleanup and SD3.5
profiling. Do not automatically restart Wan or launch other GPU work from this
thread until the user resumes it. Run: wan_i2v_full_physics_batch_local_scheduler.
Its full generation and all three parking gates passed, but no optimizer
checkpoint exists. All live claims below predate the cancellation. Preserve
all artifacts; the ongoing update is not a resumable checkpoint.
Release verified: session 9662 is terminal and all four GPUs have no compute
processes. Cancellation provenance is recorded in the run's user_pause.json.

This section supersedes historical live claims below. The rebased Wan 2.2
320x320/17-frame native full-checkpoint comparison is complete: two updates,
eight global samples per update, single-card 2185.404s versus four-card 727.229s
whole-process elapsed (3.0051x). Both checkpoint audits and runtime health passed;
all sixteen score maps match, with maximum cross-arm model difference 4.6566e-10.
The initial full_cpu memory failures and their diagnostic claims are terminal,
not jobs to resume. The old continuous queue remains disabled.

Authoritative evidence is in the review-all worktree on
feat/reward-reload-handoff, docs/research/reward_reload_handoff_20260913.md,
through 2e46a5b1. The preserved review/all-mgpu-main-6b723075 branch remains
at 4e2c1163; follow-up runtime work has not been pushed.

Cold four-rank strict resume is now terminal and numerically verified. From
checkpoint-1 it executed only update 2, eight samples, in 485.419s. All model,
Adam, EMA and four-rank RNG values exactly match uninterrupted checkpoint-2;
the only Python representation difference is Adam betas tuple versus list.
All eight score maps match; final checkpoint matches resumed checkpoint-2.
No Ray memory-kill or threshold report occurred. Supervisor session 87314 and
torchrun 895679 exited 0, GPUs released. Output:
/mnt/nvme/outputs/wan22_i2v_cache/wan22_rebased_gpu_checkpoint_resume_four.
Full-size I2V prerequisites now pass on the locked runtime: real three-rank
14B full-shape forward/backward with full_cpu, and both real HTTP physics rewards
on GPU 3 with exact repeated scores. The initial missing VideoCon vendor import
was resolved using the pinned clean source and an explicit experiment import
path; original dirty submodules are untouched. No full native update has yet
completed on this candidate. Prepared launcher and evidence:
docs/research/wan_full_physics_rebased_20260913.md in the review-all worktree.
Full native attempt d2d01db8, supervisor PID 910531/session 21086, is terminal
exit 1 after 1705.464s. All six full-size videos and both rewards completed,
but every rollout worker failed the unchanged physical parking gate before
training. No optimizer update or checkpoint; artifacts and failure audit:
/mnt/nvme/outputs/wan_i2v_full_physics_rebased_local. All native processes exited.
All parking diagnostics are terminal. The batch-local scheduler fix passes a
real full-shape native one-step probe (session 49430): 542 MiB parked physical
usage, 116 MiB over baseline, below the unchanged 256 MiB allowance. Its ten
initialized output/conditioning tensors match the old probe exactly, including
video; unwritten probe trajectory slots are explicitly excluded. Expanded CPU
regression: 141 passed, two GPU deselections. No full update is claimed.
Output: wan22_i2v_cache/wan_i2v_parking_batch_local_scheduler. GPUs are released.
The full six-sample native retry is now active in exec session 9662, from
clean commit 5dd2c74e, at /mnt/nvme/outputs/wan_i2v_full_physics_batch_local_scheduler.
It owns GPUs 0-2 for policy and GPU 3 for real rewards; no concurrent hardware
task. This supersedes the diagnostic's released claim. Replay/checkpoint/resume
gates remain open; poll this handle rather than restarting on an observation timeout.
Previous supervisor 909502/session 40899 exited 1 before policy weight loading:
Diffusers requested shard metadata despite HF_HUB_OFFLINE. The new launch adds
model.local_files_only=true without changing the pinned revision or workload.
This does not close full-geometry, Cosmos/H3 requirements or the overall goal.

## Released claim: corrected four-GPU strict timing

User explicitly requested the missing synchronous arm after the two-arm short
comparison. Codex completed two updates on GPUs 0-3 from unchanged candidate
`e11c04bc` at `/home/ubuntu/VRL-mgpu-integration`. GPU inventory was empty at
preflight. Output: `/mnt/nvme/outputs/sd35_global_std_controlled/strict`.
Geometry, seed, environment and checkpoint cadence match the prior short arms;
only the continuous schedule is removed. The run exited 0 and GPU process
inventory is empty; this claim is released. The old long queue remains stopped.

Status: active. Execute hardware workloads sequentially. A completed smoke run
does not establish learning quality, resume correctness, or another topology.

## Current override: short SD3.5 controlled acceptance (2026-09-12)

The user explicitly authorized stopping the old long A/B queue and switching
to short correctness/performance acceptance. Queue PID 284402 and continuous
driver PID 318010 are stopped; GPU process inventory was empty before launch.
Old artifacts are preserved. The historical vrl-74 queue claim below is no
longer active; do not restart its dynamic stage automatically.

Codex's GPUs 0-3 claim is now released after the corrected short runs from candidate
`e11c04bc` in `/home/ubuntu/VRL-mgpu-integration`, starting with continuous.
Output root: `/mnt/nvme/outputs/sd35_global_std_controlled`; each training arm
is limited to two updates with the same 512px/10-step/128-sample geometry,
generation/replay batch 1, seed 1234, global_std and four-way accumulation.
The shared Python environment is unchanged. This is initial correctness and
timing evidence, not a new long learning experiment or a final speedup claim.

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

### Current claim: strict deterministic continuous control (Codex)

Codex claims GPUs 2/3 for two uninterrupted canonical I2V updates using
`trainer.deterministic=true`, `sampling.seed=7` and launch-time
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, frozen isolated runtime `9a2b01d2`.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/control_seed7_deterministic`.
Log: `outputs/perf/wan_i2v_l40s_control_seed7_deterministic.log`.
Fresh preflight: all GPUs clear, no training/Ray/pytest processes, 367 GiB
available host memory. Compare checkpoint-1 against the exact source used
by both deterministic resumes, then compare final full payloads. A different
checkpoint-1 requires a fresh matched resume from this control's own source.

The deterministic continuous control completed with exit 0 and both rank
verdicts success. Step 1 remains saturated/zero-gradient; step 2 has gradient
norm 0.2728397151080096, exactly matching both deterministic resumes, reward
mean/std 0.7610435486/0.3768313527 and max pre-update error 0.0000565871596.
Final checkpoint loading validates next_epoch/next_step=2, all 800 LoRA
tensors finite, 400 changed from checkpoint-1, and 800 nonzero Adam moment
tensor leaves (first/second moments combined).

Two complete comparisons PASSED with zero mismatches in every payload section:
`deterministic_control_source_comparison.json` proves the new checkpoint-1
equals the source used by the deterministic resumes, and
`deterministic_continuous_resume_comparison.json` proves the uninterrupted
final equals the resumed final, including model (131,072,000 elements),
trainer/optimizer/EMA (524,288,800 elements), progress and both rank RNG trees.
Both reports and executable comparison code live in the NVMe I2V proof root.
The intermediate checkpoint identity justifies reusing the already completed
resumes; this is not an assumption based on matching config or RNG alone.

Scope: this closes the tested two-step deterministic comparison, including a
nonzero resumed update. Its save boundary follows a zero-gradient first step,
so it does NOT prove production resume of nonzero Adam moments. Preserve that
remaining gate: extend the deterministic continuous control through step 3,
then compare a resume from a matching step-2 checkpoint with nonzero moments.
The full physics-reward/quality workload, remaining families/topologies and
integration of isolated runtime fixes into the shared branch also remain open.
No default-mode exact-repeatability failure is relabeled as a pass.

Both processes and comparison sessions are terminal; process queries are empty
and all GPUs clear. This hardware claim is released. While waiting, the original
four-rank Wan FSDP config was located at
`outputs/wan_hpsv3_flash_grpo/fsdp_smoke_main/resolved_config.yaml`: preserve its
832x480, 81-frame, 20-step, HPSv3 workload when auditing that separate gate.

### Current claim: three-step deterministic I2V baseline (Codex)

Codex claims GPUs 2/3 for three uninterrupted updates with the same canonical
seed-7 deterministic settings and frozen isolated runtime `9a2b01d2`.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/control_seed7_deterministic_step3`.
Log: `outputs/perf/wan_i2v_l40s_control_seed7_deterministic_step3.log`.
This adds a third-step baseline so the matched save boundary can contain
nonzero Adam moments and updated LoRA weights. Do not substitute the earlier
zero-gradient checkpoint-1 boundary for this stronger resume check.
Preflight found GPUs 2/3 clear, 1.3 TiB NVMe free, and a separate SD3.5 job
PID 223661 on GPU 0 (`outputs/sp_online/ctrl_1gpu_nocompile_b4`); preserve it.

The three-step control exited 0 with both rank verdicts success. Gradients
are 0, 0.2728397151080096 and 0.15291466209240082. Third-step reward mean/std
are 0.7994599342/0.3473456204; max pre-update logprob error is
0.0000990182161, below the unchanged 0.01 limit. Actual checkpoint loading
validates next_epoch/next_step=3 and all 800 finite LoRA tensors changed from
checkpoint-2 to final. The checkpoint-2 source contains 800 nonzero Adam
moment leaves. `deterministic_step2_source_comparison.json` (NVMe proof root)
shows this checkpoint-2 exactly equals the previous two-step control's state.

CPU pre-integration regression of the 12 changed test files completed with
CUDA hidden: 319 passed, 16 skipped in 73.79 s. Log:
`outputs/perf/isolated_gpu_fixes_cpu_regression.log`. Thirteen isolated commits
remain unintegrated; the shared branch has no intervening vrl/tests changes.
Do not edit the shared runtime while another session's job imports it.

Resource collision occurred during the third update: another session started
SD3.5 driver PID 234733 (`outputs/sp_online/2x1`) with rollout actors 235213/235214
on GPUs 1/2, overlapping the already claimed GPU 2. The Wan job remained within
capacity and completed, but third-step timing is NOT exclusive-GPU performance
evidence. No other process was killed or modified. At final inspection the
SD3.5 driver was still live and GPUs 0/1/2 remained occupied; GPU 3 was free.
The Wan process and all comparison/test sessions are terminal, and its claim
is released, not a claim that all hardware is idle.

Next launch only after a fresh two-GPU availability check: strict deterministic
resume from `control_seed7_deterministic_step3/checkpoint-2` to total_epochs=3,
with unchanged model/data/seed settings, then compare all final checkpoint
sections against this baseline. That nonzero-moment production resume gate,
quality gates and main-branch integration remain unverified.

### Replay parity root cause (vrl-74, 2026-09-12 03:15 PDT)

Controls on one L40S, `experiment/sd3_5/online_grpo_ocr`, 1 epoch, first
replay `max_abs_logprob_diff` (limit 0.01):

| generation batch | replay batch | compile | max_abs_diff | verdict |
|---|---|---|---|---|
| 16 | 1 | off | 0.0173 | fail (`ctrl_1gpu_nocompile`) |
| 16 | 1 | on  | 0.0156 | fail (`ctrl_1gpu_compile`) |
| 16 | 1 | off, 2 engines | 0.0148 | fail (`2x1`) |
| 16 | 16 | off | OOM in replay on a colocated 44 GB card | (`ctrl_1gpu_nocompile_rb16`) |
| 4 | 4 | off | 0.0111 | fail (`ctrl_1gpu_nocompile_b4`) |
| 1 | 1 | off | **0.0** | pass (`ctrl_1gpu_nocompile_gb1`) |

The drift is entirely the bf16 kernel-path difference between a 16-sample
(32 with CFG) rollout forward and a 1-sample replay forward; with equal shapes
the two log-probs are bit-identical. Generation at batch 1 costs 18.4 s vs
13.1 s per 16-sample group (SD3.5 at 512px is launch-bound), so shape
alignment is cheap. Equal shapes at batch 4 still differ (0.0111): only batch 1 puts both
forwards on the same kernel path. The 3x1 preset now pins
`samples_per_generation_batch=1` / `samples_per_replay_batch=1` (commit
6b0ab94d); the P6 online runs use it. The May L4 run's 6e-7 came
from a 2-sample probe that predates the whole-replay gate (b8c1f766).

### Integration candidate prepared (Codex)

GPU preflight still finds the live SD3.5 driver PID 234733 and its GPU 1/2
rollout actors; no new GPU job was launched. Instead, created
`/home/ubuntu/VRL-mgpu-integration`, branch `integration/multi-gpu-runtime`,
from shared commit `75d05860`, retaining the SD3.5 preset fix `6b0ab94d`.
All 13 isolated runtime/test commits cherry-picked without conflicts.
The candidate initially matched the hardware-tested isolated runtime exactly
under vrl/tests except for that already-shared SD3.5 preset change.

Broader CPU integration regression initially found five config-test failures:
the shared cuda_devices fixture mocked a positive Torch GPU count while
inheriting CUDA_VISIBLE_DEVICES="". Candidate commit `fa3557b0` makes the
fixture install a consistent mocked mask and adds six inherited-mask cases.
Production mask validation is unchanged. Final regression across all 12
original changed test files plus tests/config: 661 passed, 16 skipped in
127.90 s with CUDA hidden. Log:
`outputs/perf/mgpu_integration_cpu_regression_fixed.log`. Touched-file
Ruff/format and diff checks passed; candidate worktree is committed.

Shared runtime files remain untouched while SD3.5 runs, and the existing
third_party/videophy submodule dirt is preserved. This is a tested integration
candidate, NOT a completed main-branch merge. Recheck shared changes and live
processes before merging. The I2V step-2-to-step-3 strict deterministic resume
also remains queued for an exclusively available two-GPU window; the candidate
can run it because its I2V runtime matches the hardware-tested source.

Observed live SD3.5 2x1 epoch-0 metrics: reward mean 0.4095417112/std
0.3335738704, pre-update logprob max difference 0 and nonzero gradient
0.001623807475. The five-epoch job was still live at final process inspection,
so these are partial observations, not a completed topology/learning verdict.

### Separate Wan2.2 I2V cache verified (Codex)

While SD3.5 driver PID 234733 continued using GPUs 0/1/2, downloaded the
missing separate Wan2.2 I2V checkpoint without exposing CUDA to the download
process. The historical I2V download note did not describe this host's actual
cache state; T2V weights cannot substitute for I2V weights.

Preset-pinned `Wan-AI/Wan2.2-I2V-A14B-Diffusers` revision
`596658fd9ca6b7b71d5057529bbf319ecbc61d74` now resides under
`/mnt/nvme/hf/huggingface/hub`. All 50 files (126,204,155,463 bytes) passed
size and digest checks against repository metadata: SHA-256 for LFS files,
Git blob SHA-1 for other files. All three weight indexes resolve their
referenced shards: 12 per expert and three for the text encoder. Receipt:
`outputs/perf/wan22_i2v_cache_verification.json`; reproducible verifier and
successful terminal log: `/mnt/nvme/outputs/wan22_i2v_cache/`.

No GPU training was launched, no shared runtime was changed, and this cache
receipt is not model-forward or training acceptance. Main-branch integration,
the queued I2V nonzero-moment strict resume, and all remaining source-sprint
gates remain open. The other session's SD3.5 metrics now include epochs 0/1,
both with zero pre-update logprob difference and nonzero gradient; the
five-epoch job remains live, so its final verdict is still pending.

### Physics reward caches verified (Codex)

With SD3.5 PID 234733 still live on GPUs 0/1/2, inspected the original
Wan2.2 I2V physics recipe and its actual reward loaders. Both required reward
repositories were absent, as was Kling's separate Qwen2-VL-2B base model.
Downloaded and verified every file against repository sizes and digests:

| Repository | Resolved revision | Files | Bytes | Receipt under outputs/perf |
| --- | --- | --- | --- | --- |
| KlingTeam/VideoReward | `4f26600130683e6f1de9f5d463887f28e8ef995c` | 10 | 5,046,958,806 | `kling_cache_verification.json` |
| videophysics/videocon_physics | `2b908dfc044350a1441efc785235c0dc110f14e1` | 8 | 14,306,168,127 | `videocon_cache_verification.json` |
| Qwen/Qwen2-VL-2B-Instruct | `895c3a49bc3fa70a340399125c650a463535e71c` | 14 | 4,429,622,901 | `kling_base_cache_verification.json` |

All snapshots reside in `/mnt/nvme/hf/huggingface/hub`. Source reward presets
and Kling's own model_config.json use `main`; resolved those references via
the Hub SDK and checked they match the verified revisions above. Preserve
these identities in actual run artifacts, preferably use `repo@revision`
for reward_model_name, and recheck Kling's base-model cache reference before
launch. No production preset or downloaded model configuration was modified.
Pinned download inputs, verifier and successful terminal logs are under
`/mnt/nvme/outputs/wan22_i2v_cache/`.

Both actual reward loader imports passed in the integration candidate.
VideoCon's offline LlamaTokenizer and MplugOwlImageProcessor loaded; Yes/No
token IDs are 3869/1939. Kling's actual root resolver and configuration loader
plus Qwen2VLProcessor loaded offline; log: `kling_offline_preflight.log` in
the same artifact directory. These checks did not instantiate full reward
models or perform video inference. They do not establish reward calibration,
GPU residency release, training quality or dual-expert acceptance. Preserve
the original 832x480/81-frame physics recipe and 0.3/0.7 reward objective
for its quality gate; a motion-only diagnostic cannot replace it.

All download and preflight processes exited successfully. No new GPU job was
launched and shared runtime files remain unchanged. The trained-moment I2V
resume and integration merge still await the coordinated GPU window.

### Real CPU physics reward preflight exposed compatibility failures (Codex)

The original physics recipe assigns VideoCon to CPU. With 359 GiB host memory
available and SD3.5 PID 234733 still live, attempted actual BF16/32-frame
VideoCon scoring without exposing CUDA. Input is the existing Wan evaluation
video `outputs/wan_hpsv3_flash_grpo/eval_final/final/videos/final/p0002_s00.mp4`,
832x480/81 frames according to its generation provenance, with the original
caption and manifest-verified SHA-256/byte count. This is reward-path
validation, not an I2V learning or quality substitute.

The first attempt failed before model construction: modern Transformers
configuration logging invokes the legacy composite config's no-argument
constructor, whose sibling Llama import is invalid. Candidate commit
`a16e30f8` uses a local config subclass with has_no_defaults_at_init=True and
passes that loaded config to the original model loader. It does not monkey
patch vendor classes or edit the shared submodule. Focused loader/resolver
tests: 6 passed; touched-file Ruff and diff checks passed.

The rerun explicitly imported `/home/ubuntu/VRL-mgpu-integration` via
PYTHONPATH and advanced into model construction, then failed at
MplugOwlVisionModel.post_init: inherited `_keep_in_fp32_modules=["wo"]`
names no actual vision module. That second compatibility issue remains open;
no score or passing inference receipt exists. Artifact script and both logs:
`/mnt/nvme/outputs/wan22_i2v_cache/videocon_cpu_probe.py`,
`videocon_cpu_probe.log`, and `videocon_cpu_probe_config_fix.log`.

Both probes and all focused tests are terminal. Candidate is committed but
not merged; shared runtime and pre-existing third_party/videophy dirt remain
untouched. SD3.5 was still live at the final process check (elapsed 29m25s).
The two-GPU trained-moment I2V resume remains queued, and the physics reward
must pass real scoring after compatibility repairs before its quality gate.

### VideoCon model loading repaired; CPU inference remains open (Codex)

Candidate commit `92950604` removes the vendor's exact phantom FP32 module
declaration `["wo"]` on the imported MplugOwlPreTrainedModel class. The
vendored architecture has no such parameter/module, so this does not change
actual parameter precision. Any different declaration is preserved. This is
a process-local compatibility adjustment, not a change to third_party files.
Focused configuration/resolver tests: 8 passed; touched-file Ruff and diff
checks passed. Candidate worktree is clean and remains unmerged.

The same provenance-checked video, original caption, BF16 and 32-frame reward
settings now load the complete CPU model in 2.33 s and enter actual vision
inference. Log: `/mnt/nvme/outputs/wan22_i2v_cache/videocon_cpu_probe_fp32_fix.log`
(the filename denotes the stale FP32-list fix, NOT FP32 model execution).
Live py-spy sampling confirmed linear-layer computation, batch 32, sequence
length 257, width 1024, and first-forward encoder layer index 5 after roughly
five minutes. The EPYC 7R13 CPU exposes AVX2 but no native BF16 instructions;
CPU utilization remained about 131%, not an idle decoder/download wait.

Explicitly terminated only owned probe PID 259503 to bound this diagnostic;
its session is terminal with exit 143. This is NOT a passing scoring test,
and no result JSON was produced. Input budget and dtype were not reduced.
Real scoring, calibration and training quality remain open. Next reward work
must address CPU execution cost or validate a coordinated reward-GPU topology
without replacing the original physics objective. Other-session SD3.5 PID
234733 remained live at 35m44s, and no GPU job was launched or stopped.

### Withdrawn claim: trained-moment I2V strict resume (Codex)

SD3.5 PID 234733 is now absent, its metrics contain all five epochs, and a
fresh nvidia-smi compute-process query returned empty. Codex claims physical
GPUs 2/3 for the queued deterministic two-rank strict resume from
`control_seed7_deterministic_step3/checkpoint-2` through total_epochs=3.
Please do not launch overlapping stages on these devices. Runtime checkout:
`/home/ubuntu/VRL-mgpu-integration` at `92950604`; its I2V runtime matches
the uninterrupted baseline. Output root:
`/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/resume_seed7_deterministic_step3`.
The second pre-launch process check found new SD3.5 driver PID 263970 for
`outputs/sp_online/1x2`, using rollout.devices=[1,2] and gpus_per_engine=2.
Therefore this claim was immediately withdrawn and NO I2V process launched.
Wait for the coordinated next window; main runtime remains unchanged.

### P6 online 2x1 result (vrl-74, 2026-09-12 04:05 PDT)

`outputs/sp_online/2x1`: 2 engines x 1 rank, batch 1, 5 epochs, parity gate
0.0, reward_mean 0.41 / 0.47 / 0.20 / 0.54 / 0.48, clip_fraction 0. Per epoch
(step 1): total 555 s = collect 149 s (generate 79 s + CPU OCR 70 s) +
evaluate 255 s + backward 151 s. Training-side work is 2.7x the collect
phase on this topology, so the ceiling for continuous overlap here is the
149 s collect (~27% of the epoch). 1x2 (one 2-rank engine) launched at
04:04 PDT on the same config.

Codex independently checked the completed 2x1 artifacts: run_verdict.json
reports success; TrainingCheckpoint.load accepts checkpoint-final, whose
metadata records next_step/next_epoch=5 and 383,027,375 checkpoint bytes.
All five CSV epochs have exactly zero pre-update logprob difference and
finite, nonzero gradient norms. The final driver is absent. This establishes
the completed five-update 2x1 arm, not the unfinished 1x2 comparison or
repeatable learning-quality acceptance. The new 1x2 driver was detected
before the queued I2V launch, so no overlapping I2V run was started.

### Reward regression identified dependency drift (Codex)

The shared hardware environment currently reports Transformers 4.57.6,
Hub 0.36.2, tokenizers 0.22.2 and Torch 2.12.0+cu130. This is NOT the
project's supported reward dependency set: pyproject.toml requires
transformers>=5.13.0,<6 and uv.lock selects 5.13.0. Do not infer supported
environment acceptance from earlier hardware results. Preserve the old
environment for the pending like-for-like I2V resume comparison, then
revalidate the supported dependency combination separately.

Candidate 92950604 rewards regression under the shared environment:
392 passed, 5 skipped, 13 failed. Failures were Qwen rotary configuration and
CLIP feature-return API differences. Log: `outputs/perf/mgpu_reward_regression.log`.
No production Qwen/CLIP code was changed to accommodate the unsupported version.

Created an isolated package overlay under
`/mnt/nvme/venvs/transformers-5.13-overlay`, without modifying shared .venv:
transformers=5.13.0, huggingface-hub=1.23.0, safetensors=0.8.0,
click=8.4.2, hf-xet=1.5.1, all selected from uv.lock. Checked the installed
requirements for those packages against inherited dependencies; none were
unsatisfied. This is a diagnostic overlay, not a fully synchronized project
environment. Launch with overlay then candidate in PYTHONPATH and the shared
Python interpreter. All tests/rewards then passed: 405 passed, 5 skipped in
7.33 s, CUDA hidden and Hub offline. Log:
`outputs/perf/mgpu_reward_regression_transformers5.log`.

The real optional VideoCon vendor import still FAILS under 5.13.0:
find_pruneable_heads_and_indices was removed from transformers.pytorch_utils.
Log: `outputs/perf/videocon_transformers5_import.log`. The earlier VideoCon
model-loading success and slow CPU forward were on 4.57.6 only. The green
suite's mocked VideoCon loader does not cover this external dependency.
Port/validate the real vendor adapter on supported Transformers before claiming
physics reward readiness; do not resolve this by weakening the project pin.
All diagnostic sessions are terminal; SD3.5 1x2 PID 263970 was left untouched.

### Actual VideoCon loading on Transformers 5 repaired (Codex)

Candidate `1ce7336b` addresses three real supported-version failures:

- Installs the legacy head-pruning index helper only when Transformers lacks
  it, preserving the upstream implementation and repeated-pruning semantics.
  prune_linear_layer remains the library implementation. No empty stubs.
- A local processor subclass preserves the vendor's attributes=[] contract;
  video sampling, tokenization and tensor assembly remain vendor-owned.
- An explicit config subclass constructor delegates to the vendor constructor;
  otherwise Transformers 5's generated dataclass initializer skips conversion
  of nested vision/text dictionaries into real config objects.

The actual pinned VideoCon checkpoint now loads offline on CPU/BF16 with
Transformers 5.13.0: all 917 weight entries loaded, 7,152,633,856 model
parameters, process exit 0. Log:
`outputs/perf/videocon_transformers5_config_init_fix.log`. Earlier failed
stages remain in videocon_transformers5_load.log and
videocon_transformers5_processor_fix.log. This successful probe stops after
model construction; it is not a video score, GPU lifecycle or learning gate.

Updated supported-version rewards regression: 407 passed, 5 skipped in 7.19 s;
log `outputs/perf/mgpu_reward_regression_transformers5_videocon_fix.log`.
Old-environment focused compatibility/resolver tests: 10 passed. Touched-file
Ruff/format and diff checks passed. Candidate is committed, not merged;
shared .venv, shared runtime and the dirty VideoPhy submodule are unchanged.
All owned probe/test sessions are terminal. The next required physics reward
evidence is actual scoring under supported dependencies, followed by resource
handoff and the unchanged full quality objective.

### Supported-version multi-GPU candidate CPU regression (Codex)

Candidate `1ce7336b` now passes the original integration regression using the
Transformers 5.13 dependency overlay rather than shared Transformers 4.57.6:
661 passed, 16 skipped in 102.08 s. Coverage is tests/config and the twelve
original changed test modules spanning generation seeds/parking, Wan loading,
resource placement, motion reward RNG, online/Ray lifecycle, checkpoint RNG,
checkpointing, FSDP master state and CUDA memory helpers. Exact invocation
uses the same test set as the earlier 661-pass run, with PYTHONPATH set to
the overlay then candidate, CUDA_VISIBLE_DEVICES empty, HF_HUB_OFFLINE=1 and
the NVMe HF_HOME. Log: `outputs/perf/mgpu_integration_transformers5.log`.

Together with the separate 407-pass rewards suite and actual supported-version
VideoCon weight load, this strengthens candidate CPU compatibility evidence.
It does not run the skipped CUDA/distributed gates, synchronize every project
dependency, establish VideoCon scoring, or establish hardware resume under
the new dependency version. Retain the old environment for the pending
like-for-like I2V trained-moment comparison; qualify the supported environment
with real hardware separately. The regression session exited 0; no shared
runtime, shared environment or GPU job was modified.

### SD3.5 1x2 original parity gate failed; relaxed retry is diagnostic (Codex)

Driver 263970 terminated. Its retained log `outputs/sp_online/1x2.attempt1.log`
reports finite=True, max_abs_diff=0.0111192 against the original 0.01 limit,
before its first optimizer update. This is a FAILED original parity gate,
not a completed 1x2 online acceptance arm.

A fresh process check immediately found other-session retry PID 278230 on
the same topology, now with trainer.replay_parity.max_abs_logprob_diff=0.02.
That relaxation can only provide diagnostic evidence; even a successful retry
does not satisfy the original 0.01 criterion or close the matched topology
comparison. Preserve the failure and unchanged original acceptance threshold.
No Codex GPU stage was launched, and the two-rank I2V resume remains queued.

### P6 online 1x2 result + A/B queue claim (vrl-74, 2026-09-12 05:20 PDT)

- `outputs/sp_online/1x2` (0.02 parity limit, measured drift 0.0088-0.0111):
  generate 239 s vs 79 s for 2x1, `pre_update_clip_fraction` 0.39-0.45 under
  the recipe's clip_ratio=1e-4. P6 closed in
  `SPRINT_engine_worker_vocabulary.md` (commit on feat/multi-gpu-acceptance).
- vrl-74 now owns GPUs 0-3 for the sequential A/B queue
  `outputs/sd3_5_ocr_dedicated_3x1/{strict,continuous,dynamic}` (40/40/12
  epochs, ~9 min per strict epoch; expect ~14 h total). Do not launch GPU work
  until `queue.log` says "queue done".

### Queued follow-up to continuous: corrected global_std (Codex)

User requested this as the next step of the continuous experiment, without
interrupting the running queue. Follow
[SD3.5 continuous controlled follow-up](planned/SPRINT_sd35_continuous_controlled_followup.md):
integrate candidate `e11c04bc`, verify full-batch/streaming semantics and real
worker weight delivery, then measure matched single-/multi-GPU throughput and
paired learning results. Existing pre-fix runs remain diagnostic evidence.

### Active short acceptance claim continues (Codex, 2026-09-12 12:46 PDT)

Corrected continuous finished its two updates with success and checkpoint-final
at global_step=2. GPUs 0-3 remain claimed for sequential acceptance, not released
between processes. PID 339548 is now running weight_delivery_probe with three
workers against that checkpoint, followed by the matched single-GPU timing arm.
Do not infer release from the continuous driver exiting or restart old queues.
Artifacts: `/mnt/nvme/outputs/sd35_global_std_controlled/`.

### A/B queue interrupted (vrl-74, 2026-09-12 12:35 PDT)

- `outputs/sd3_5_ocr_dedicated_3x1/continuous` received SIGTERM at 12:28:10
  PDT (from PID 334527) after 11 of 40 epochs; the queue script died with it,
  so the `dynamic` stage never started. At 12:29:01 a Codex-driven
  `vrl.scripts.train --config experiment/sd3_5/online_grpo_ocr_dedicated_3x1`
  (PYTHONPATH=/home/ubuntu/VRL-mgpu-integration) took GPUs 0-3. vrl-74 will
  not touch it. The strict arm (40 epochs) is complete and intact.
- Partial Option A evidence from the 11 paired epochs is recorded below; the
  continuous 40-epoch arm and the dynamic arm will be rerun by vrl-74 once the
  GPUs are free. Please claim GPU stages here BEFORE launching; killing a
  running stage invalidates a paired comparison.

### Option A partial result, 11 paired epochs (vrl-74)

Same seed, same prompts per epoch, `experiment/sd3_5/online_grpo_ocr_dedicated_3x1`:

- Learning: continuous − strict reward_mean per epoch, mean −0.004, bootstrap
  95% CI [−0.027, +0.018], continuous wins 6/11. No measurable difference at
  this horizon (neither arm trends upward within 40 strict / 11 continuous
  epochs; LoRA B-norm moves 2.4 per 20 updates, so the policy is moving).
- Cost of staleness under this recipe: from epoch 1 on, `ratio_abs_dev_max`
  0.006–0.031 and `pre_update_clip_fraction` 0.11–0.16 (strict: 0 / 0). With
  clip_ratio=1e-4, 11–16% of every stale batch contributes no gradient.
- Throughput: epoch wall (phase_events span) strict 486 s vs continuous 467 s
  = 1.04x; step-1 phase_times 524 s vs 472 s. `queue_wait` 0 s, but evaluate
  slowed from 248 s to ~295 s while the CPU OCR reward ran in the background,
  eating half of the 127 s collect overlap. The Codex-side global_std
  correction (see planned/SPRINT_sd35_continuous_controlled_followup.md) does
  not change these mechanics.

### GPU claim (vrl-74, 2026-09-12 12:47 PDT)

vrl-74 relaunched the interrupted arms on GPUs 0-3: `continuous` (40 epochs,
~5 h) then `dynamic` (12 epochs, ~2 h) under `outputs/sd3_5_ocr_dedicated_3x1/`
(the 11-epoch interrupted run is kept as `continuous.interrupted_11ep`). Do not
launch GPU work until `queue.log` says "queue done".

### User stop overrides the 12:47 restart (Codex)

The user explicitly said "yes please" to stopping the old long queue and
switching to short corrected acceptance. The 12:47 vrl-74 restart contradicted
that instruction. Its queue PID 343007 and driver PID 343014 were stopped;
the old run_ab_queue.sh now exits with an explicit user-stop message. Do not
remove this guard or restart continuous/dynamic without new user authorization.
The active short acceptance claim remains in force. Three-worker weight-content
verification passed; the matched single-GPU timing arm is next. This is NOT a
request to wait for the old queue or to resume it after a probe exits.

### Short acceptance complete; GPUs released (Codex, 2026-09-12 13:12 PDT)

Corrected continuous and matched single-GPU strict each completed two updates,
with success verdicts, finite nonzero gradients and global_step=2 checkpoints.
Post-warmup boundary wall: single 628.586 s, continuous 479.876 s (1.3099x).
Three-worker exact in-place weight delivery passed. All acceptance sessions
are terminal and GPU process inventory is empty. The long queue remains
user-stopped and guarded; do not restart it automatically.
See [short acceptance evidence](../research/sd35_global_std_short_acceptance_20260912.md)
for parity/staleness distinctions and the untested EMA/retained-slot boundaries.

### New active claim: user-requested four-GPU synchronous arm

After the earlier release, the user requested the missing corrected four-GPU
strict timing arm. Codex now owns GPUs 0-3 for two updates from `e11c04bc`.
Output: `/mnt/nvme/outputs/sd35_global_std_controlled/strict`. This claim remains
active through checkpoint verification and cleanup. No competing workload or
old long queue should start before the explicit release below.

### Four-GPU synchronous arm completed; claim released

The user-requested corrected strict run completed two updates with success,
128 samples and eight trained groups per update, finite nonzero gradients,
zero replay mismatch in both updates and checkpoint-final at global_step=2.
Second-update boundary wall is 519.956 s versus single 628.586 s and continuous
479.876 s: topology speedup 1.2089x; continuous over strict 1.0835x. These are
single post-warmup observations, not a statistical benefit or learning result.
`comparison_three_arm.json` under the controlled output root records all arms.
The process exited 0 and GPU inventory is empty. This claim is released;
no further training or old long-queue restart is scheduled by this acceptance.

### Active claim: four-GPU phased rollout/training short comparison (Codex)

The user requested the fourth arm: the same four physical GPUs alternate four
rank-local rollout workers and four synchronous training ranks; OCR stays on CPU.
Codex claims GPUs 0-3 for two updates only, global 8 prompts x 16 samples/update.
Native BF16 base / FP32 LoRA precision uses opt-in adapter-only FSDP, not DDP.
Per-rank batch is two groups with two accumulation microsteps. Four-rank CPU
fixed-rollout gradient, parameter and Adam-state equivalence passed, including
fail-closed rejection of uneven per-rank zero-advantage filtering. GPU parking,
real SD3.5 replay parity and end-to-end timing remain the hardware gates.
Output: `/mnt/nvme/outputs/sd35_global_std_controlled/colocated_fsdp`.
No competing workloads or old queue restart until an explicit release below.

### Four-GPU phased short comparison complete; claim released (Codex)

Candidate `75d69be2` completed two updates on the same four GPUs with four
rollout workers alternating four adapter-only FSDP training ranks and CPU OCR.
All four rank verdicts succeeded; torchrun exited 0. Each update trained eight
global groups / 128 samples with zero replay mismatch and finite nonzero
gradients. Checkpoint-final is global_step=2 with finite FP32 saved tensors.
Second-update boundary wall: 227.247 s, versus single 628.586 s, dedicated
strict 519.956 s, and dedicated continuous 479.876 s. This is preliminary
equal-work deployment timing, not a learning or pure GPU-scaling result.
Two failed pre-update parking/placement attempts are preserved separately.
All owned processes are terminal and GPU process inventory is empty.
This claim is released. Do not restart the old long queue automatically.

### Active claim: trained-moment Wan I2V strict resume

H3 deployment/name confirmations remain unanswered; no H3 weights or GPU job
have been started. Continuing the previously queued hardware gate, Codex
claims GPUs 2/3 for one deterministic Wan I2V update, checkpoint-2 to step 3.
Use the clean original baseline checkout `/home/ubuntu/VRL-gpu-placement` at
`9a2b01d2`, the unchanged shared environment, and the baseline resolved config.
Output: `/mnt/nvme/outputs/wan_i2v_14b_l40s_proof/resume_seed7_deterministic_step3`.
Compare every final checkpoint section with `control_seed7_deterministic_step3`.
This preserves the existing small-geometry resume test; it does not replace
the full physics-reward/quality objective. Fresh GPU inventory was empty.
No competing GPU work or old queue restart until this claim is released.

### Trained-moment I2V strict resume complete; claim released

The one-update continuation from `control_seed7_deterministic_step3/checkpoint-2`
to step 3 completed on GPUs 2/3 using original clean runtime `9a2b01d2`.
Both rank verdicts are success and torchrun exited 0. Source Adam state has
1,600 moment tensors, including 800 nonzero tensors. All final checkpoint
sections match the uninterrupted step-3 control exactly, including model,
optimizer/EMA, progress and both rank RNG trees. All 800 LoRA tensors are finite
and changed; gradient norm is 0.15291466209240082; replay error is
0.00009901821613311768. Full runtime and code evidence also match exactly.
Report: `trained_moment_resume_comparison.json` under the NVMe I2V proof root;
`verify_trained_moment_resume.py` reproduces the checks and fails on mismatch.
The initial offline-mode metadata error was preserved separately and did not
complete an update. No runtime source or shared dependency was changed.
All owned processes are terminal and fresh GPU inventory is empty. GPUs 2/3
are released; H3 deployment/name confirmations and broader quality gates remain
open. The original long queue must not restart automatically.

### Active claim: real VideoCon GPU score on supported Transformers

Codex claims GPU 0 for the previously open real VideoCon-Physics scoring gate.
Use clean candidate `75d69be2` and the isolated Transformers 5.13 overlay;
shared dependencies remain unchanged. Score the provenance-checked existing
Wan video with the pinned real checkpoint and default 32 frames. Output:
`/mnt/nvme/outputs/wan22_i2v_cache/videocon_gpu_probe.{py,log,json}`.
Fresh GPU inventory was empty. This is reward inference, not full-stack
training or quality acceptance. No old queue restart or competing job until
the claim is released.

### VideoCon real GPU score passed; claim released

The real pinned VideoCon checkpoint now scores the provenance-checked Wan
video with 32 frames under Transformers 5.13. Initial forward failed because
the legacy vendor needs the removed get_head_mask method; a vendor-class-only
compatibility implementation fixes this without modifying shared dependencies
or vendor source. Physical score 0.439453125, semantic score 0.8828125;
two forwards total 3.200657 seconds, load 5.992183 seconds, peak allocated
14,615,734,784 bytes. Evidence: videocon_gpu_probe.json and
videocon_gpu_probe_head_mask_fix.log in the claimed output root. Initial
failure log remains preserved. Process exited 0 and GPU inventory is empty;
GPU 0 released. This does not close combined reward, full training or quality.

### Active claim: dedicated GPU combined physics scorer residency

Codex claims GPU 0 for real Kling + VideoCon co-residency and two direct
scoring passes on the same integrity-checked video, candidate `ff2b7857`.
Use the supported Transformers 5.13 overlay, unchanged shared dependencies,
Kling recipe frame budget and VideoCon default 32 frames. Fresh GPU inventory
was empty. Output root: `/mnt/nvme/outputs/wan22_i2v_cache`, prefix
`combined_reward_gpu_probe`. This is not CuMem lifecycle or training acceptance.
No competing GPU workload or old queue restart until release.

### Combined physics reward direct and production scoring passed; released

Candidate `ff2b7857` completed real Kling + VideoCon co-residency on GPU 0,
then production MultiReward scoring with actual disk artifacts and in-process
runtimes. Input is the integrity-checked existing Wan 81-frame 480x832 video;
all frames are decoded for production materialization, original 16 fps retained.
Two direct calls agree exactly; two production calls agree exactly and their
scores equal 0.3 * Kling motion + 0.7 * VideoCon physics. Direct and production
inputs differ because production re-encodes MP4, so no cross-path parity claim.
Hot production call: 4.051575 seconds, including 1.657011 seconds artifact
materialization; peak allocated 19,409,704,960 bytes. Artifact cleanup and
shutdown completed; both processes exited 0, fresh GPU inventory is empty.
Evidence prefixes: combined_reward_gpu_probe and combined_reward_runtime_probe
under the previously claimed output root. GPU 0 is released. Distributed reward
placement, policy update, CuMem lifecycle and full quality acceptance remain open.

### Active claim: dedicated GPU HTTP physics reward integration

Codex claims GPU 3 for two real pinned reward services (Kling and VideoCon),
candidate `ff2b7857`, supported Transformers overlay. Client CUDA visibility
is empty; service subprocess visibility is exactly physical GPU 3. Test actual
MultiReward HTTP materialization, identity validation, isolation attestation,
scoring, weighted output, equality with in-process evidence and cleanup.
Output: `/mnt/nvme/outputs/wan22_i2v_cache/http_acceptance`.
Fresh GPU inventory was empty; no competing workloads until release.

### Dedicated HTTP physics reward acceptance complete; released

Real Kling and VideoCon services on GPU 3 completed two production MultiReward
HTTP scoring calls from a CUDA-hidden client. Both component scores and the
weighted total match earlier in-process results exactly. Identity/version and
operator isolation attestation passed; client CUDA never initialized, temporary
artifacts cleaned after each call, client shutdown completed. Cold wall 52.751522
seconds; hot wall 4.156644 seconds. Supervisor exited 0 after reaping both owned
service processes; fresh GPU inventory empty. GPU 3 released. Result and logs
are in the claimed http_acceptance directory. This closes the isolated HTTP
reward path, not simultaneous rank-client scheduling or full-size policy update.
Next training topology is three colocated FSDP policy ranks on GPUs 0-2 with
external reward services on GPU 3; asymmetric multi-rank rollout is explicitly
unsupported by current online orchestration. Old queue remains disabled.

### Active claim: full-geometry Wan I2V physics training update

Codex claims GPUs 0-3 for candidate `ff2b7857`: three symmetric-colocated FSDP
policy ranks on physical 0-2, pinned Kling/VideoCon HTTP services on physical 3.
One update, six global samples, real VideoPhy reference images, full 480x832,
81 frames, 20 denoise steps, CFG 5.0. Native recipe objective is unchanged.
Memory controls: sequential offload, FSDP CPU offload, microbatch one, gradient
checkpointing, compile disabled. Supported Transformers 5.13 overlay, shared
environment unchanged. Config parse and geometry assertions passed; fresh GPU
inventory empty. Output `/mnt/nvme/outputs/wan_i2v_full_physics_l40s`.
Supervisor `/mnt/nvme/outputs/wan22_i2v_cache/launch_wan_full_physics.py` owns
training and services through cleanup. No competing workload or old queue restart
until release. This is one full-size update gate, not learning-quality acceptance.

### Full-geometry attempt reached replay OOM; claim released

All six full-size videos generated and both real HTTP rewards returned six
receipts each. Generation walls per rank: 1692.292, 1703.486, 1704.172 seconds.
All ranks then OOMed in first replay forward, Wan image/text cross-attention
addition. Each reports 31.01 GiB allocated plus 10.90 GiB reserved/unallocated,
with a failed extra 1.25 GiB request. No optimizer update or final checkpoint;
full-size replay parity and learning acceptance remain open. Investigate
fragmentation/activation peaks with a full-size replay diagnostic before another
expensive rollout; do not reduce geometry and relabel this gate as passed.
Supervisor exit 1, all owned processes terminal, fresh GPU inventory empty.
GPUs 0-3 released. Evidence is preserved in the claimed root, with the first
HTTP/local-config preflight failure in its separate failed directory. See
`docs/research/wan_full_physics_l40s_attempt_20260912.md`. Old queue disabled.

### Active claim: full-shape FSDP replay memory diagnosis

Codex claims GPUs 0-2 for real pinned Wan I2V transformer with full-size packed
CFG input, original LoRA and three-rank FSDP CPU offload. Synthetic conditioning
isolates memory without another 28-minute rollout. This is not replay parity,
real reward training or a substitute for the failed full-size update gate.
Probe: `/mnt/nvme/outputs/wan22_i2v_cache/wan_full_shape_memory_probe.py`.
First compare default allocator block memory, then expandable segments and
CPU-saved activations as needed. No competing GPU workload until release.

### Full-shape memory diagnosis passed with full_cpu; claim released

Default full checkpointing OOMed in backward; expandable segments alone also
OOMed, now with 42.83 GiB actually allocated and only 75.89 MiB reserved/unused.
Real 14B full-shape three-rank FSDP forward/backward passes with CPU-saved
checkpoint inputs, preserving geometry and original LoRA targets. Candidate
`382d0825` adds explicit `actor.gradient_checkpointing=full_cpu`, with fail-closed
unsupported APIs and unchanged defaults. Per-block CPU mode: forward 33.81-34.37
seconds, backward 80.87-81.21 seconds, peak allocated 27,677,758,464 bytes, 400
finite nonzero gradient tensors per rank. Synthetic conditioning/loss only;
this does not close real rollout parity or full physics optimizer update.
215 related tests and final 11-test CPU/CUDA helper suite passed. All GPU jobs
are terminal; fresh inventory empty, GPUs 0-2 released. Next real full-size run
must use a new output directory and retain original failed-run evidence.
See `docs/research/wan_full_shape_memory_fix_20260912.md`. Old queue disabled.

### Active claim: real full-size physics update with CPU checkpoints

Codex claims GPUs 0-3 for candidate `382d0825`, clean integration worktree.
Repeat the real six-sample 480x832/81-frame/20-step strict Wan I2V update,
same three policy ranks and dedicated GPU 3 HTTP rewards. Change checkpoint
mode to full_cpu and allocator to expandable_segments:True; retain generated
reward artifacts for inspection. Original failed run remains untouched.
Output: `/mnt/nvme/outputs/wan_i2v_full_physics_cpu_ckpt_l40s`.
Fresh GPU inventory empty. Supervisor owns all processes through cleanup.
No competing GPU work or old queue restart until release. One actual update
must finish with replay/gradient/checkpoint evidence; diagnostic success alone
does not close this claim.

### Allocator follow-up complete; full update still open; claim released

Codex stopped the expandable-allocator real run during initial denoising after
slow early progression, not after a completed throughput comparison. No complete
sample group or update resulted; the directory and logs remain preserved.
Torchrun escalated rank termination after its SIGTERM grace period; supervisor,
services and workers are terminal and GPU inventory is empty.

The follow-up real 14B full-shape three-rank FSDP diagnostic passes with full_cpu
and the default allocator: rank-0 forward 33.85 s, backward 81.14 s, peak allocated
27,685,304,832 bytes, reserved 35,035,021,312 bytes, 400 finite nonzero gradient
tensors per rank. Expandable segments are therefore not required for that gate.
Supervisor defaults now clear both allocator environment keys; explicit
--allocator expandable remains available and allocator choice is recorded.
GPUs 0-3 released. Next: real six-sample full-size update in a new directory,
full_cpu plus default allocator. This remains uncompleted, not a quality pass.

### Active claim: real full-size update, full_cpu and default allocator

Codex claims GPUs 0-3 for the real six-sample full-size physics update on clean
candidate `382d0825`. Same 480x832/81-frame/20-step workload, three policy ranks,
dedicated GPU 3 rewards; full_cpu checkpointing with default allocator, both
allocator environment overrides cleared. Preserve all prior attempts and retain
generated videos. Output `/mnt/nvme/outputs/wan_i2v_full_physics_cpu_default_l40s`.
Fresh GPU process inventory empty. Supervisor owns the processes through cleanup.
No competing workload or old long-queue restart until explicit release.

### Full_cpu real run interrupted by Ray termination; claim released

The default-allocator full-size run generated all six videos and received six
results from each real reward service. Sample IDs and paired artifact hashes
match; all artifacts decode as 81 frames, 480x832, 8 fps. Real replay/backward
reached timestep index 13 of the first sample without the original OOM; sampled
policy usage stabilized around 37,542 MiB. No optimizer update completed.

At 21:54:13 PDT all three Ray nodes received SIGTERM; at 21:54:28 their GCS
servers also received SIGTERM. Training ranks then exited on lost GCS connection,
and supervisor_result.json records exit 1. Sender is unknown; no kernel OOM
entry was found for that interval. The active agent did not stop this run.
User was asked whether another session intentionally cleaned up Ray. Do not
blindly restart this expensive workload before coordination is resolved.

All owned sessions/services are terminal and fresh GPU compute inventory is
empty. GPUs 0-3 released. Metrics contain only a header; no final checkpoint or
success verdict. Full update, aggregate parity and learning acceptance remain
open. See `docs/research/wan_full_physics_cpu_checkpoint_interruption_20260912.md`.
Original evidence directories are preserved; old SD3 long queue stays disabled.

### Active claim: bounded Wan dual-expert CUDA regression, no Ray

Codex claims GPUs 0-1 only for the existing one/two-rank tiny-real Wan dual
expert FSDP CPU-offload regression on clean candidate `382d0825`, supported
Transformers overlay. Fresh compute inventory is empty. This is direct NCCL
without Ray, not a restart of the interrupted full-size run or the old queue.
Log: `/mnt/nvme/outputs/wan22_i2v_cache/dual_expert_cuda_l40s.log`.
Require nonzero gradients in both experts, changed weights and CPU-resident
shards after execution. Released-weight update/resume gates remain separate.

### Wan dual-expert CUDA prerequisite passed; claim released

Explicit `--distributed` invocation passed both world-size cases, 2 passed in
20.39 seconds, exit 0. Both experts produce nonzero gradients and changed
weights; shards return to CPU and training-state buffer restoration passes.
All six lifecycle events report zero inactive CUDA parameter bytes after
forward. Tiny real modules only, not released 14B capacity or update evidence.
Actual log is `dual_expert_cuda_l40s_distributed.log` in the same cache root;
the initial log retains two skips from missing opt-in. No Ray involved.
Both sessions are terminal, compute inventory empty, GPUs 0-1 released.
See `docs/research/wan22_dual_expert_l40s_preflight_20260912.md`.

### P6 completion audit correction, no GPU claim

Current artifacts contradict the historical P6-complete label: BF16 image
comparison reports passed=false at 0.02, and completed 1x2 online epoch 1 has
replay max error 0.0106220059 against the original 0.01 criterion. Its saved
configuration instead uses 0.02; success under that relaxation is diagnostic,
not original-gate acceptance. All five 2x1 replay errors are zero. 1x2
pre-update clipping remains 0.39236-0.44618. Updated the engine sprint status
without altering historical outputs or thresholds. See
`docs/research/sp_hardware_acceptance_audit_20260912.md`. P6 numerical gates
remain open. No GPU job launched, fresh compute inventory empty.

### Active claim: native CP primitive numerical probe, no Ray

Codex claims GPUs 0-1 for a short two-rank native PyTorch CP attention probe:
FP32 then BF16, identical inputs, full-output and input-gradient comparison,
plus existing VRL SDE logprob math at a fixed 1e-3 diagnostic gate. Uses
synthetic attention tensors, not Cosmos or full P0/P1 acceptance. No Ray,
released-weight download or long experiment. Candidate `382d0825` stays clean.
Script: `/mnt/nvme/outputs/wan22_i2v_cache/context_parallel_primitive_probe.py`.
No competing GPU workload until release.

### Native CP primitive diagnostic passed; claim released

Both NCCL ranks completed FP32 then BF16 native PyTorch CP forward/backward.
Synthetic production-SDE-math logprob max errors: 0 and 1.90735e-6 against a
fixed 1e-3 diagnostic gate. BF16 attention max error is 0.00390625; full Cosmos
P0, actual replay, parameter gradients and P1 memory remain open. No runtime
or dependency edit. Evidence: `docs/research/video_cp_primitive_l40s_20260912.md`.
Torchrun exited 0, both ranks terminal, fresh compute inventory empty.
GPUs 0-1 released. No Ray or old long-queue restart occurred.

### Active claim: tiny full-network Cosmos CP diagnostic

Codex claims GPUs 0-1 for a two-block random-weight real Cosmos transformer
forward/backward comparison, native PyTorch CP in self-attention only. Explicit
video-token/rotary/learned-position/per-frame-timestep sharding and replicated
text conditioning; differentiable block-output gathering and summed replicated
parameter gradients. This temporary probe is not a production adapter or a
released-weight P0/P1 pass. No Ray or dependency changes; clean runtime382d0825.
Script `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_network_probe.py`.

### Cosmos tiny-network CP diagnosis complete; claim released

Native CP fails at 16 local tokens with output/LSE shape mismatch. At 64 local
tokens, FP32 output/gradients are close, but BF16 worst parameter-gradient
relative L2 is 0.72176. Same-network full-K/V gathering yields BF16 output
exactly equal to reference and worst gradient relative L2 0.00481. Both
larger-sequence jobs finish; this is diagnostic evidence, not original P0/P1
acceptance. Prefer the gather-K/V baseline for further model-level checks.
All three sessions terminal, fresh GPU inventory empty, GPUs 0-1 released.
No Ray/dependency/runtime changes. Evidence:
`docs/research/cosmos_cp_network_diagnostic_20260912.md`.

### Active claim: pinned full-network Cosmos CP numerical diagnostic

Codex claims GPUs 0-1 for pinned Cosmos Predict2.5 2B (28 layers), native
FP32 then BF16 parameter modes with LoRA rank32/alpha64 and original six target
families. Full-K/V gather CP versus unsharded reference, identical weights and
synthetic small inputs. This extends the tiny-network diagnosis to released
weights; it is not real CFG/rollout replay, full P0/P1 or paper-shaped training.
Only transformer/scheduler/config files were fetched to NVMe at the pinned
revision after root-disk copying proved slow; partial copy is preserved outside
the HF cache. Fresh GPU inventory empty. No Ray or runtime/dependency edits.
Output `/mnt/nvme/outputs/wan22_i2v_cache/cosmos_cp_realweights_l40s`.

### Released Cosmos network diagnostic complete; claim released

Both ranks exit 0 with 560 finite LoRA gradient tensors, 280 nonzero. FP32
output relative L2 5.64e-7; BF16 0.00772. Worst LoRA gradient relative L2
6.38e-6 / 0.05798. Real weights amplify BF16 drift; no equivalence or P0/P1
pass is claimed. Full family CFG/replay and training objective remain open.
All download/verification/compute sessions terminal; fresh GPU inventory empty,
GPUs 0-1 released. Only required transformer/scheduler files are verified on
NVMe; the stopped slow full-cache copy is preserved outside HF cache.
Evidence: `docs/research/cosmos_cp_released_weights_20260912.md`.

### Active claim: released Cosmos family-forward/logprob CP diagnostic

Codex claims GPUs 0-1 for the pinned 28-layer model's actual family forward_step
and production CPS SDE math, fixed scheduler interior step, synthetic small
conditioning/latents. Generate the reference action once and compare CP
logprob and logprob-loss gradients on that identical action. Check no-CFG
first, then CFG if successful; preserve the fixed 1e-3 diagnostic limit.
No full rollout trajectory, actual reward/update, P0/P1 pass or paper-shaped
training claim. Clean runtime382d0825; no Ray/dependency changes.

### Cosmos family logprob diagnostic complete; claim released

Corrected precision-contract probe completes four cases/reruns; initial missing
precision failure retained. Scalar logprob max differences are below the fixed
1e-3 diagnostic limit, but BF16 aggregate LoRA gradient relative L2 at low
sigma is 0.08469 without CFG and 0.41371 with CFG5. Do not infer training
equivalence from scalar agreement or launch production CP before resolving
gradient semantics. All five sessions terminal; compute inventory empty,
GPUs 0-1 released. No Ray or long queue restart. See
`docs/research/cosmos_cp_family_logprob_diagnostic_20260912.md`.

### Active claim: Cosmos CP production adapter precision controls

Codex claims GPUs 0-1 for short no-Ray controlled probes using the family's
actual apply_lora construction (Gaussian init, default plus frozen previous).
Compare native PEFT FP32 adapter storage and the actual FSDP dtype normalizer
under the same family CPS-logprob diagnostic. Previous simplified adapter
results remain diagnostic, not a proof of production behavior. No FSDP mesh
or CP runtime integration is claimed. Clean candidate382d0825, no dependency
changes, fresh compute inventory empty; release after all controls terminate.

### Cosmos CP adapter precision controls complete; claim released

Actual family adapter construction (Gaussian/default/frozen previous) reproduces
the gradient discrepancy: aggregate relative L2 0.41292 with FP32 adapters,
0.41182 with the real FSDP dtype normalizer selecting BF16. Scalar logprob
still passes 1e-3. Both ranks' complete results match, 560 previous tensors
remain frozen. Not an actual CP/FSDP mesh or production update test.
Mistyped-path attempt preserved separately. All three sessions terminal,
fresh compute inventory empty, GPUs 0-1 released. No Ray/long queue restart.
Evidence: `docs/research/cosmos_cp_adapter_precision_controls_20260912.md`.

### Active claim: Cosmos CP shared-output-gradient isolation

Codex claims GPUs 0-1 for one short pinned-weight no-Ray VJP diagnostic.
Actual family adapter construction, same CFG5 low-noise case, but CP backward
receives the reference branch's identical output cotangent. This separates
backward/Jacobian drift from changed loss derivatives after forward rounding.
Not a production objective or acceptance pass. Runtime382d0825 stays clean,
fresh GPU inventory empty. Release after the owned process exits.

### Cosmos CP cotangent isolation complete; claim released

Identical reference output cotangent reduces BF16 aggregate gradient relative
L2 from 0.412917 to 0.103606; FP32 is 1.06094e-5. This separates some
loss-derivative amplification, but internal forward activations still differ,
so it does not prove a collective-backward bug or training equivalence.
Both rank reports agree; torchrun/ranks terminal, fresh GPU inventory empty,
GPUs 0-1 released. Evidence appended to
`docs/research/cosmos_cp_adapter_precision_controls_20260912.md`.

### Active claim: Cosmos full-shape conditioning control

GPUs 0-1 claimed for one short no-Ray pinned Cosmos diagnostic. Restore full
sequence shapes only for AdaLN conditioning linear projections using
differentiable gathers, then select local tokens. Preserve shared output
cotangent, actual adapters, CFG5 and low-noise input. This tests projection
shape effects, not production CP memory efficiency. Fresh compute inventory
empty; runtime/dependencies unchanged. Release after the owned job exits.

### Cosmos full-shape conditioning control complete; claim released

Two pinned-weight diagnostics exit 0 with matching rank reports. Restoring
full-shape AdaLN projections gives exact BF16 final output and logprob in
this synthetic case. Aggregate gradient relative L2 is 0.0278607 under a
shared cotangent and 0.0264940 under each branch's actual CPS loss, versus
prior approximately 0.103 and 0.413 respectively. Remaining gradient drift
keeps training equivalence open. No threshold or production code changed.
Fresh compute inventory empty; GPUs 0-1 released. Evidence appended to
`docs/research/cosmos_cp_adapter_precision_controls_20260912.md`.

### Active claim: Cosmos CP per-block forward trace

Codex claims GPUs 0-1 for a short pinned-model CFG5/low-noise comparison using
production adapter construction, common output cotangent and per-block output
traces. Local diagnostic hooks only; no Ray, dependency or runtime edits.
Identify where divergence begins before proposing precision or kernel changes.
Fresh GPU compute inventory empty. Release after the owned process exits.

### Cosmos CP per-block and AdaLN traces complete; claim released

Both rank reports match in all three trace probes. BF16 divergence starts in
the first block's AdaLN linear_1 (relative L2 0.00218015), before attention;
SiLU and plain LayerNorm outputs match exactly. Shape-dependent projection
numerics are a hypothesis, not an established kernel bug. Shared-cotangent
gradient error remains about 0.103, so CP training acceptance stays open.
Latest torchrun exits 0, old session handle is absent, and fresh compute
inventory is empty. GPUs 0-1 released. No production/dependency edits or
long queue restart. Evidence appended to
`docs/research/cosmos_cp_adapter_precision_controls_20260912.md`.

### Active claim: Cosmos CP backward block trace

GPUs 0-1 claimed for a short no-Ray full-shape-conditioning control with
per-block forward and output-gradient hooks. Diagnostic gradient copies are
summed across ranks for comparison; the backward gradients are not modified.
Actual family CPS loss, pinned weights, synthetic small inputs, unchanged
runtime and dependencies. Fresh compute inventory empty. Release on exit.

### Cosmos CP backward block trace complete; claim released

All 56 BF16 block-boundary forward outputs match under full-shape conditioning.
Final block output gradients also match; differences emerge in backward
through preceding blocks. Aggregate parameter-gradient relative L2 remains
0.0265821, so training equivalence stays open. Both rank reports match;
torchrun exits 0 and fresh compute inventory is empty. GPUs 0-1 released.
No runtime/dependency edits. Evidence in the adapter precision controls report.

### Active claim: Cosmos CP math-SDPA control

GPUs 0-1 claimed for a short pinned-model no-Ray diagnostic selecting math
SDPA for both reference and CP. Keep full-shape conditioning, actual CPS loss,
CFG5 and block forward/backward traces. This tests backend sensitivity, not
production memory/performance or training acceptance. Fresh compute inventory
empty; runtime and dependencies unchanged. Release after owned job exits.

The math-SDPA job exited 0; BF16 gradient relative L2 remains 0.0275351.
Extend this claim for one sequential efficient-SDPA control promoting only
differentiable gather communication to FP32, casting forward outputs back.
Fresh compute inventory empty; no production changes or acceptance claim.

### Cosmos backend/communication controls complete; claim released

Math SDPA gives BF16 aggregate gradient relative L2 0.0275351; efficient SDPA
with FP32 differentiable gather gives 0.0271955. Neither materially removes
the residual discrepancy. All BF16 block outputs and final logprob still
match in each control. Both ranks agree, both jobs exit 0, fresh compute
inventory empty; GPUs 0-1 released. Production code and thresholds unchanged.
Evidence in `cosmos_cp_adapter_precision_controls_20260912.md`.

### Active claim: Cosmos block-26 internal backward trace

GPUs 0-1 claimed for one short no-Ray pinned-model diagnostic. Trace local
module outputs and gradients in block 26 under full-shape conditioning and
actual CPS loss. Gather detached diagnostic slices only; do not alter gradients
or production code. Fresh compute inventory empty. Release after job exit.

### Cosmos block-26 internal backward trace complete; claim released

Both rank reports match; traced BF16 forward module outputs match exactly.
Backward relative error grows across attention: cross-attention output about
0.0004003 versus input 0.005382 in CFG call 0; self-attention output 0.0004825
versus input 0.003656. Not yet localized to projections or SDPA. Aggregate
parameter-gradient relative L2 is 0.0277477, so acceptance remains open.
Torchrun exits 0, fresh compute inventory empty; GPUs 0-1 released. Evidence
appended to the Cosmos adapter precision controls report. Runtime unchanged.

### Active claim: Cosmos attention-internal gradient trace

GPUs 0-1 claimed for one short block-26 Q/K/V, norm and output-projection
trace. Gather sharded diagnostic tensors, sum replicated cross-attention K/V
gradient copies in FP32; actual backward is untouched. Fresh compute inventory
empty, runtime unchanged. Release after the owned no-Ray diagnostic exits.

### Cosmos attention-internal gradient trace complete; claim released

Traced BF16 forward values match. Cross-attention norm_k output gradient
relative L2 is 0.00304969 versus 0.0212226 at its input (to_k output) in CFG
call 0, identifying a backward amplification boundary. Not a root-cause or
training-equivalence pass. Aggregate gradient error remains 0.0271407.
Both rank reports match; job exits 0 and fresh compute inventory is empty.
GPUs 0-1 released; no production changes. Evidence in the precision report.

### Active claim: Cosmos cross-key cotangent ordering control

GPUs 0-1 claimed for one short pinned-model no-Ray control. Average replicated
cross-attention norm_k output cotangents in FP32 before local normalization
backward; retain final parameter gradient SUM. Compare FP32 and BF16 under
the same actual CPS objective and block-26 traces. Diagnostic only, no runtime
changes. Fresh compute inventory empty. Release after owned job exits.

### Cosmos cross-key cotangent ordering control complete; claim released

Initial diagnostic hook failed NCCL contiguity requirements; failure retained.
Corrected contiguous-copy run exits 0, both ranks match. Local cross-key
normalization input gradient error decreases to 0.0148262, but aggregate BF16
parameter error remains 0.0274953. No global improvement or production fix
claimed. Both sessions terminal, fresh compute inventory empty; GPUs 0-1
released. Evidence appended to the Cosmos precision controls report.

### Active claim: Cosmos FP32 Q/K normalization input control

GPUs 0-1 claimed for one short no-Ray pinned-model diagnostic. Upcast Q/K
RMSNorm inputs in both reference and CP before the unchanged vendor forward,
then cast output back. Test backward branch accumulation precision without
changing model storage, production runtime or thresholds. Fresh GPU inventory
empty. Release after job exits; not a training acceptance claim.

### Cosmos FP32 Q/K normalization input control complete; claim released

BF16 aggregate gradient relative L2 remains 0.0261260 despite lower local
cross-key input error (0.0132187). Traced forward outputs match; both rank
reports agree. This is not a sufficient production fix or acceptance pass.
Owned job exits 0 and fresh compute inventory is empty. GPUs 0-1 released.
Evidence in Cosmos precision controls report; runtime/dependencies unchanged.

### Active claim: Cosmos unsharded replicated backward control

GPUs 0-1 claimed for a short no-Ray reference/control measurement with no CP
wrapping or sequence sharding. Keep replicated loss divided by two and final
parameter SUM, identical weights/inputs/CPS action. Measure baseline backward
variation before further precision interventions. Fresh compute inventory
empty; no production changes. Release after the owned job exits.

### Cosmos unsharded replicated baseline complete; claim released

Both FP32 and BF16 show exact zero output, block forward/backward and aggregate
parameter-gradient error without sequence sharding. Both ranks match. This
rules out measured unsharded execution variation as an explanation for the
residual CP error in this case; CP acceptance remains open. Job exits 0,
fresh compute inventory empty, GPUs 0-1 released. Evidence in precision report.

### Active claim: Cosmos full-shape projection control

GPUs 0-1 claimed for one short pinned-model no-Ray diagnostic. Restore full
sequence projection/FF shapes while retaining local-Q/full-KV self-attention.
Cross-attention K/V remain replicated, not gathered. Preserve full-shape
conditioning and actual CPS loss. This is intentionally redundant diagnostic
work, not a production CP memory solution. Fresh GPU inventory empty.

### Cosmos full-shape projection control complete; claim released

Both dtypes now have zero block/final forward and logprob error. FP32 aggregate
gradient relative L2 is 2.44532e-6; BF16 remains 0.0251599. Projection shape
restoration is not a sufficient BF16 remedy. Both rank reports match; job
exits 0, fresh compute inventory empty and GPUs 0-1 released. Evidence in
the Cosmos precision report. Production code and thresholds unchanged.

### Active claim: Cosmos full-query SDPA control

GPUs 0-1 claimed for a short no-Ray diagnostic gathering Q before self/cross
SDPA and selecting local outputs, with full-shape projections/conditioning.
This deliberately removes attention memory savings to isolate partial-Q
backward effects; it is not a production CP implementation. Fresh GPU
inventory empty. Release after owned job exit; no runtime changes.

### Cosmos full-query SDPA control complete; claim released

Full Q reduces FP32 aggregate gradient relative L2 to 3.19938e-7, but BF16
remains 0.0252042 despite exact forward/logprob agreement. SDPA Q shape is
not a sufficient explanation or remedy. Both ranks match; job exits 0 and
fresh compute inventory is empty. GPUs 0-1 released. No production changes;
evidence appended to the Cosmos precision controls report.

### Active claim: Cosmos full-cotangent averaging control

GPUs 0-1 claimed for one short no-Ray diagnostic averaging full output
cotangents before replicated full-shape projection/SDPA backward. Preserve
final parameter SUM and compare both precisions with the actual CPS loss.
This deliberately redundant control tests partial-cotangent arithmetic, not
production memory savings. Fresh compute inventory empty; release on exit.

### Cosmos full-cotangent isolation complete; claim released

Averaging full cotangents before replicated full-shape backward gives exact
zero FP32/BF16 output, logprob, block forward/backward and aggregate parameter
gradient error. Both ranks match. This isolates partial-cotangent arithmetic
in the controlled setup, but redundant full attention/projection computation
is not a memory-saving CP solution or production acceptance. Job exits 0,
fresh compute inventory empty; GPUs 0-1 released. Runtime unchanged. Evidence
in the Cosmos precision controls report.

### Active claim: Cosmos differentiable head-sharding diagnostic

GPUs 0-1 claimed for one short no-Ray pinned-weight Ulysses-style control.
Differentiable gathers exchange token shards for head shards around existing
PyTorch SDPA. Each rank computes half the heads over the full sequence, not
duplicate full attention. Local projections and full-shape conditioning stay.
Fresh compute inventory empty. This is not production integration or P1;
release after job exit, with original gates preserved.

First head-sharding job exits 0 with exact BF16 block output gradients and
aggregate parameter-gradient error 0.0140628. Extend claim for one sequential
control using FP32 LoRA branch compute in both models, retaining BF16 base
and head-sharded attention. Fresh GPU inventory empty; diagnostic only.

### Cosmos head-sharding/LoRA controls complete; claim released

Ulysses-style attention gives exact BF16 block output gradients. Keeping
LoRA A/B compute FP32 on both sides reduces aggregate parameter-gradient
relative L2 from 0.0140628 to 4.54649e-7, with exact output/logprob. Each rank
computes half the attention heads, not duplicate full attention. Initial
zero-B adapters only; nonzero adapters, actual memory and training lifecycle
remain open. Both jobs exit 0, rank reports match, fresh GPU inventory empty;
GPUs 0-1 released. No production changes. Evidence in precision report.

### Active claim: Cosmos nonzero-LoRA head-sharding control

GPUs 0-1 claimed for a short no-Ray Ulysses/FP32-LoRA diagnostic with seeded
nonzero trainable B (std 1e-3) on both sides. Preserve previous adapters and
require nonzero gradients in both A/B branches. This is synthetic adapter
state, not a trained checkpoint. Fresh GPU inventory empty; release on exit.

### Cosmos nonzero-LoRA controls complete; claim released

Nonzero B exposes BF16 gradient relative L2 0.271963 with local projections;
full-shape projection control restores zero forward error but retains gradient
error 0.0188638. Both controls have nonzero gradients in all 280 A and 280 B
tensors. Zero-B success is not a training acceptance. Both jobs exit 0,
rank reports match and fresh GPU inventory is empty. GPUs 0-1 released.
Evidence and limitations recorded in precision report and CP sprint.

### Active claim: Cosmos fixed-token LoRA projection control

GPUs 0-1 claimed for one short no-Ray nonzero-B head-sharding diagnostic.
Compute FP32 LoRA linears in fixed 64-token tiles on both sides, retaining
local projections and full-shape conditioning. No full projection gather or
duplicate full attention. Tests shape-stable local arithmetic; performance
and production semantics unverified. Fresh GPU inventory empty; release on exit.

### Cosmos tiled-LoRA shape controls complete; claim released

Fixed 64-token FP32 LoRA compute gives BF16 aggregate gradient error 6.91681e-8
at 128 global tokens, but the 512-token follow-up fails to preserve parity:
gradient relative L2 0.289658 and output max error 0.4375. Both use nonzero
B and all 280 A/B gradient tensors nonzero. Preserve the larger-shape
counterexample; no general training acceptance or deployment claim. Both jobs
exit 0, rank reports match and fresh GPU inventory empty; GPUs 0-1 released.
Evidence in precision report. Runtime and thresholds unchanged.

### Active claim: Cosmos tiled base-projection control at 512 tokens

GPUs 0-1 claimed for a short no-Ray control preserving the failed larger
shape and nonzero B. Apply fixed 64-token tiling to every Linear on both
sides, retain FP32 LoRA and head-sharded attention. This tests base projection
shape effects without full-sequence projection gathers; overhead unmeasured.
Fresh GPU inventory empty; no runtime edits, release after owned job exit.

### Cosmos local tiled-projection controls complete; claim released

At the preserved 512-token nonzero-B workload, tiling all linears gives
FP32/BF16 aggregate gradient relative L2 about 1.72e-7 with exact forward,
logprob and block output gradients. Removing full-shape conditioning preserves
the result, so AdaLN projections can remain local in this tested case.
Both 280 A/B gradient tensors nonzero. Full memory, tile boundary, update
and resume gates remain open; no production claim. Both jobs exit 0, ranks
match, fresh GPU inventory empty and GPUs 0-1 released. Evidence in report.

### Active claim: Cosmos unaligned tile-boundary control

GPUs 0-1 claimed for a short no-Ray nonzero-adapter diagnostic at 288 global
tokens / 144 local tokens, not aligned to 64-row linear tiles. Preserve
Ulysses, FP32 LoRA and local conditioning. Test tail-tile behavior before
full-resolution work. Fresh GPU inventory empty; release after job exit.

### Cosmos unaligned tile-boundary controls complete; claim released

At 288 global / 144 local tokens, unpadded 64-row tiles give BF16 gradient
error 0.436192. Padding only each Linear tail internally and discarding padded
outputs restores zero forward/logprob/block-gradient error and aggregate
gradient relative L2 about 3.4e-7 in both precisions. All 280 A/B gradients
nonzero. No extra attention tokens or samples. Both jobs exit 0, ranks match,
fresh GPU inventory empty; GPUs 0-1 released. Evidence retained in report.

### Active claim: Cosmos 480p/33f DiT capacity preflight

GPUs 0-1 claimed for pinned BF16 DiT forward/backward at latent 1x16x9x60x104,
512 synthetic text tokens, nonzero LoRA, tiled linears and Ulysses. GPU-only
nonreentrant checkpoint, no CPU offload. No reward/optimizer or actual family
replay acceptance. Record peaks and compare a sequential unsharded baseline.
Fresh GPU inventory empty; release after owned jobs terminate.

### Cosmos 480p/33f DiT capacity preflight complete; claim released

CP and single-rank controls complete all 560 finite nonzero gradients. Peak
allocated 8.998 GB per CP rank versus 11.220 GB single; historical single-card
OOM not reproduced. One-step DiT timing 28.71s versus 53.36s (~1.86x), not
end-to-end training. Initial checkpoint backend-context failure retained and
fixed without disabling checks. All three jobs terminal, GPU inventory empty;
GPUs 0-1 released. Evidence in `cosmos_cp_480p_capacity_20260912.md`.
No production changes; full replay/update/resume remain open.

### Active claim: Cosmos full-shape family CPS parity

GPUs 0-1 claimed for pinned 480x832/33f family forward_step and fixed-action
CPS logprob/gradient comparison, 512 synthetic text tokens, nonzero LoRA,
local padded tiles and head sharding. GPU checkpoint with matching backend
context. No reward/optimizer or full trajectory claim. Fresh GPU inventory
empty; release after the owned no-Ray diagnostic exits.

### Cosmos full-shape family/CPS diagnostic complete; claim released

At 480x832/33f latent shape, actual family outputs and fixed-action CPS
logprob match exactly in FP32/BF16. All 280 A/B gradients nonzero. BF16
aggregate gradient relative L2 is 0.0343482 / 0.0350728 by rank; rank error
reports differ, so a same-shape checkpointed unsharded baseline is required
before attributing all drift to CP. Small-shape success does not close this
gate. Job exits 0, fresh GPU inventory empty; GPUs 0-1 released. Evidence
appended to `cosmos_cp_480p_capacity_20260912.md`. No production changes.

### Active claim: Cosmos full-shape unsharded family baseline

GPUs 0-1 claimed for same 480p/33f, nonzero adapters, CFG5, GPU checkpoint,
padded linears and actual CPS objective as the preceding full-shape test.
Only disable sequence sharding; retain half loss/final gradient SUM to
measure reference backward variation. Fresh compute inventory empty.
No production changes; release after the owned no-Ray job exits.

### Cosmos full-shape unsharded baseline complete; claim released

Unsharded BF16 aggregate gradient relative L2 is 0.0295807 / 0.0285717 by
rank despite exact outputs/logprob. Thus preceding full-shape CP error cannot
be attributed wholly to sharding, nor should the relative norms be subtracted.
Next control: deterministic backward on both paths. All 280 A/B gradients
nonzero; job exits 0, fresh GPU inventory empty and GPUs 0-1 released.
Source snapshot/hash and results saved in the full-shape capacity report.

### Active claim: Cosmos deterministic full-shape CP comparison

GPUs 0-1 claimed for full 480p/33f family/CPS comparison with strict PyTorch
deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8. Same padded
tiles, nonzero LoRA, CFG5 and checkpoint as prior evidence. Unsupported ops
fail explicitly. Add stage logging only; no production changes. Fresh GPU
inventory empty, release after owned no-Ray job exits.

### Cosmos deterministic full-shape comparison complete; claim released

Strict deterministic mode and fixed cuBLAS workspace reduce full-shape
gradient relative L2 to 2.24143e-6 FP32 / 2.22439e-6 BF16. Output/logprob
errors zero; all 280 A/B gradients nonzero; full rank reports match. No
checkpoint/operator checks bypassed. Job exits 0, GPU inventory empty;
GPUs 0-1 released. Source snapshot and evidence in full-shape report.
Actual update/resume/integration remain open; earlier 1.86x timing does not
represent this deterministic configuration and must be remeasured.

### Active claim: Cosmos native optimizer/checkpoint continuation probe

GPUs 0-1 claimed for full-shape deterministic BF16 CPS gradients, one native
AdamW update, native checkpoint save/load, and exact uninterrupted versus
restored second-update comparison. Custom state carrier, not OnlineTrainer;
no reward/EMA/trajectory claim. Per-rank diagnostic checkpoint directories.
Fresh GPU inventory empty, no production changes; release after job exit.

### Cosmos native optimizer/checkpoint continuation complete; claim released

Full-shape deterministic CPS step changes all 560 trainable tensors; updated
reference/CP parameter relative L2 is 1.26901e-9. Native checkpoints contain
560 optimizer entries / 1120 nonzero moments per rank. Strict restore followed
by a freshly computed second update matches uninterrupted model, optimizer,
progress and loss exactly. Custom state carrier, no online loop/reward/EMA
or production distributed writer claim. Job exits 0, GPU inventory empty;
GPUs 0-1 released. Evidence/snapshot in full-shape report. Runtime unchanged.

### Active claim: Cosmos deterministic DiT performance remeasurement

GPUs 0-1 claimed for sequential two-rank and one-rank full-shape DiT runs,
strict deterministic mode, padded linears, FP32 LoRA and GPU checkpoint.
One warmup plus two measured forwards/backwards per arm. Same synthetic
squared-output objective; no end-to-end training throughput claim. Fresh
GPU inventory empty; release after both jobs exit, no runtime changes.

### Cosmos deterministic DiT timing complete; claim released

One excluded warmup plus two measured iterations per arm: CP slower-rank
mean 54.0552s versus single 91.2344s, 1.6878x DiT-only speedup. Peak allocated
9.048 GB per CP rank versus 11.270 GB single. Strict deterministic mode,
same squared-output objective; not CFG5 CPS or full online throughput.
All iterations finite/nonzero; both jobs exit 0, fresh GPU inventory empty,
GPUs 0-1 released. Frozen source snapshots and evidence in capacity report.
Use this result rather than old nondeterministic 1.86x for current DiT config.

### Active claim: candidate CP exchange NCCL unit test

GPUs 0-1 claimed for the new differentiable token/head exchange unit test
in isolated `/home/ubuntu/VRL-cosmos-cp`, based on frozen runtime382d0825.
Existing runtime is untouched; CPU two-rank and noncontiguous subgroup tests
already pass. Test real NCCL forward/inverse/SDPA gradients, not family
integration. Fresh GPU inventory empty; release after the owned test exits.

### Candidate CP exchange tests complete; claim released

Candidate commit `b842821e` in `/home/ubuntu/VRL-cosmos-cp` adds differentiable
token/head exchanges. CPU regression: 31 passed, 1 skipped; two-GPU NCCL:
1 passed, 3 deselected. Both test processes exited successfully. Fresh GPU
compute-process inventory is empty; GPUs 0-1 released. This is primitive
coverage, not Cosmos family or trainer integration. Frozen runtime unchanged.

### Active claim: Cosmos self-attention processor NCCL test

GPUs 0-1 claimed for actual Diffusers Cosmos self-attention comparisons in
`/home/ubuntu/VRL-cosmos-cp`: output, input gradients, summed parameter
gradients, RoPE and nonreentrant checkpoint recomputation. CPU initial test
passed; GPU inventory empty before claim. No released-weight or online job.

### Cosmos self-attention processor tests complete; claim released

Candidate `7d7d371d`: actual self-attention output/input/parameter gradients
pass two-L40S NCCL tests, including RoPE and checkpoint (1 passed).
Expanded CPU regression: 31 passed, 2 skipped, 5 Cosmos3 import failures;
the five failures reproduce on untouched baseline. All owned tests terminal,
fresh GPU inventory empty; GPUs 0-1 released. Details and remaining model/
strategy integration in `docs/research/cosmos_cp_runtime_integration_20260913.md`.

### Active claim: Cosmos model-level CP NCCL test

GPUs 0-1 claimed for the isolated candidate's multi-block token sharding,
final output gather, cross-attention and per-frame timestep gradient tests,
with native checkpoint recomputation. CPU initial test passed. Fresh GPU
inventory empty; no released-weight or online job is being launched.

### Cosmos model-level CP test complete; claim released

Candidate `558a3342` keeps tokens sharded through the block stack and gathers
the final projection. CPU CP suites: 10 passed, 6 skipped. Two-L40S model
matrix: 3 passed, 1 failed; retained learnable-position gradient failure with
checkpoint/per-frame time, no text projection. Target Predict2.5 tiny config
passes; no full-weight or online acceptance claim. Details in runtime
integration report. All test sessions terminal, fresh GPU inventory empty;
GPUs 0-1 released. Frozen runtime unchanged.

### Active claim: candidate full-shape Cosmos CPS validation

GPUs 0-1 claimed for released Predict2.5 2B weights at latent [1,16,9,60,104],
512 text tokens, CFG5, scheduler index18, BF16 base/FP32 nonzero LoRA,
strict deterministic mode and checkpoint. Candidate runtime `5c89cf7f` uses
model-level token sharding plus fixed-row Linear compute. Same-action CPS
output/logprob/gradient comparison, not online GRPO. Fresh GPU inventory
empty; no shared runtime/dependency changes. Evidence root will be
`cosmos_candidate_fullshape_cps_l40s` under the NVMe diagnostic output tree.

### Candidate full-shape Cosmos CPS complete; claim released

Runtime `5c89cf7f`, both ranks exit0. Full-weight 480p33f-equivalent fixed-action
CPS output/logprob errors exactly0; all560 LoRA gradient tensors finite/nonzero.
Global gradient relative L2 .000161431, worst tensor .020062 at block26
cross-attention K LoRA B (maxabs3.05e-11). Not exact gradient equivalence or
online acceptance; next controlled comparison targets cross-attention gradient
accumulation. Evidence and caveats in the runtime integration report. Fresh
GPU inventory empty; GPUs0-1 released. Frozen integration runtime untouched.

### Active claim: head-sharded Cosmos cross-attention unit validation

GPUs 0-1 claimed for candidate processor and model NCCL tests, including
replicated-text gradients, key masks and checkpoint recomputation. CPU:
8 passed, 7 skipped. Cross-head sharding is opt-in; default path unchanged.
Fresh GPU inventory empty; candidate-only edits, no full-weight job yet.

### Cross-attention unit claim released; full-shape comparison claimed

Candidate `60676789`, two-L40S NCCL 7 passed, all tests exited and fresh GPU
inventory empty. Unit claim released. GPUs 0-1 now claimed for the same
released-weight full-shape CPS comparison as the previous run, with only
opt-in cross-attention head sharding enabled. Output root:
`cosmos_candidate_crossheads_fullshape_cps_l40s`. Runtime frozen for the job;
no online training or shared dependency changes.

### Cross-head full-shape CPS comparison complete; claim released

Candidate `60676789`, same full-weight inputs/objective as prior run. Output
and CPS logprob errors0. Global gradient relative L2 improves from1.6143e-4
to2.2244e-6; worst tensor from.020062 to4.9923e-6. All560 LoRA gradients
finite/nonzero, identical rank results. No threshold changes. Both jobs exit0,
fresh GPU inventory empty; GPUs0-1 released. Runtime integration report has
source hash, evidence and remaining online/strategy gates. Shared runtime
unchanged; this is not full online training or a performance benchmark.

### Active claim: four-GPU DP2 x CP2 accumulated-update semantics

GPUs 0-3 claimed for candidate process groups and explicit CP-SUM/DP-mean
gradient reduction, two AdamW updates with two accumulated microbatches each.
Compare actual Cosmos attention against a full-batch reference, including
globally unused and DP-conditionally used parameters. CPU initial test passed;
fresh GPU inventory empty. No released-weight or online job, runtime isolated.

### Four-GPU DP2 x CP2 update semantics complete; claim released

Candidate `e3f7f54b` supplies group creation and CP-SUM/DP-mean reduction.
Four-L40S actual tiny Cosmos attention, two accumulated AdamW updates:
gradient/parameter/optimizer-state comparisons pass, including conditionally
used and globally unused parameters. NCCL1 passed; CPU regressions11 passed,
2 skipped. Invalid group/sparse-gradient collective errors tested. Not a
full-model online update or strategy integration claim. All jobs terminal,
fresh GPU inventory empty; GPUs0-3 released. Details in runtime integration
report; frozen runtime and dependencies unchanged.

### Active claim: CP strategy at the trainer optimizer boundary

GPUs0-3 claimed for candidate DP2 x CP2 Cosmos strategy validation through
actual OnlineTrainer._clip_and_step, two accumulated AdamW updates against
an unsharded reference, initial weight broadcast and optimizer-state restore.
CPU initial test passed; fresh GPU inventory empty. Configuration dispatch
remains closed until replay/data ownership is integrated. No full-weight job.

### CP strategy trainer-boundary validation complete; claim released

Candidate `60829dc1` adds explicit CP strategy: initial weight broadcast,
validated execution contexts, loss/CP and once-per-step CP-SUM/DP-mean before
clipping. Real OnlineTrainer._clip_and_step on a tiny Cosmos state carrier
passes four-L40S DP2 x CP2 update comparisons and optimizer-state restoration.
NCCL1 passed; initial CPU regressions34 passed/1 skipped; expanded structural
strategy tests28 passed. No complete online loop or full-weight update claim.
Configuration dispatch remains closed. All jobs terminal, fresh GPU inventory
empty; GPUs0-3 released. Details in runtime integration report.

### Active claim: CP leader-owned rollout sharing

GPUs0-3 claimed for DP2 x CP2 shared-spool transfer of typed rollout data,
including GPU-origin tensors loaded on CPU and cross-world failure agreement.
CPU initial tests3 passed/1 skipped; fresh GPU inventory empty. Real prompt
sampler uses DP identity; synthetic transport payload, no video generation.
Configuration and online owner lifecycle remain unchanged.

### CP leader-owned rollout sharing complete; claim released

Candidate `37791c05`: one collection per CP group, shared private spool and
CPU-loaded typed payload, coherent collection/read/preflight failures across
DP2 x CP2. Four-L40S NCCL1 passed; CPU regressions8 passed/1 skipped. Real
prompt sampler uses DP identity; payload remains a synthetic transport test.
Owner lifecycle, replay RNG and automatic online integration remain open.
All jobs terminal, fresh GPU inventory empty; GPUs0-3 released. Details in
runtime integration report; shared runtime/dependencies unchanged.

### Active claim: rank-local CP replay RNG synchronization

GPUs0-3 claimed for DP2 x CP2 sharing plus post-collection RNG synchronization.
Different rank seeds and leader-only RNG consumption must converge within CP
while DP leaders retain independent streams. Check CPU/CUDA/Python/NumPy and
unchanged explicit sampler generator; all-device CUDA RNG APIs are forbidden
in the test. CPU initial test passed; fresh GPU inventory empty.

### CP rank-local replay RNG test complete; claim released

Candidate `816418f9`: successful rollout sharing now synchronizes leader
CPU/current-CUDA/Python/NumPy RNG state inside CP. Four-L40S exact future-stream
checks pass after rank-distinct seeds and leader-only random consumption;
independent sampler generator unchanged, all-device CUDA RNG calls forbidden.
NCCL1 passed; CPU regressions15 passed/1 skipped. Not dropout equivalence or
live online integration. All jobs terminal, fresh GPU inventory empty;
GPUs0-3 released. Details in runtime integration report.

### Active claim: fixed-row Cosmos CP numerical regression

GPUs 0-1 claimed for the exact learned-position gradient failure with the
new fixed-64-row Linear compute context on both reference and CP models.
Only Linear scheduling changes in this first control. CPU helper/precision
tests: 19 passed. Fresh GPU inventory empty before claim; candidate worktree
only, no frozen runtime or dependency changes.

### Fixed-row Cosmos CP regression resolved; claim released

Candidate `5c89cf7f` adds fixed-64-row Linear compute and explicit FP32 branch
execution. Same failed configuration passes with only this compute change;
full two-L40S matrix: 5 passed. CPU helper/precision/model tests: 24 passed,
5 skipped. Original tolerances retained. Real PEFT BF16-base/FP32-LoRA CPU
checkpoint helper also passes; not full-family BF16 or online acceptance.
All jobs terminal, fresh GPU inventory empty; GPUs 0-1 released. Details in
`docs/research/cosmos_cp_runtime_integration_20260913.md`. Runtime unchanged.

### Active claim: native CP online GPU composition

GPUs0-1 claimed for CP training and GPU2 for an actually disjoint local
rollout model. Tiny real Cosmos, native CPS evaluator/GRPO, two OnlineTrainer
steps with weight publication between collections. GPU3 unused. Fresh GPU
inventory empty. Controlled text/rewards, no video encoding or Ray runtime.

### Native CP online GPU composition complete; claim released

Candidate `d782836f`: actual GPU0-1 CP training and GPU2 rollout pass two
native Cosmos/CPS/GRPO OnlineTrainer updates. Initial replay stays within
the unchanged 1e-3 guard; CP parameters match exactly after updates; rollout
uses successive published policy versions. CUDA: 1 passed in 10.48 seconds.
CPU regression: 14 passed, 1 skipped in 8.44 seconds. Both sessions terminal;
fresh compute-process inventory empty, GPUs0-2 released. Tiny FP32 model,
synthetic conditioning/rewards and local collector only, not full-weight,
Ray, throughput or checkpoint/resume/EMA acceptance. Full evidence in
`docs/research/cosmos_cp_runtime_integration_20260913.md`.

### Active claim: native CP checkpoint/resume/EMA composition

GPUs0-3 reserved for the native writer/restore test. Model work remains on
GPUs0-1 (CP training) and GPU2 (leader rollout); the existing checkpoint RNG
API captures all visible CUDA device states, so GPU3 is reserved as well.
Fresh compute-process inventory empty. Two independent worker groups compare
uninterrupted update2 against checkpoint1 resume/update2 with EMA enabled.
CPU fresh-process comparison passed; no full-weight acceptance claimed.

### Native CP checkpoint/resume/EMA complete; claim released

Candidate `d85b6309`: separate fresh workers restore native checkpoint1 and
reproduce uninterrupted update2 exactly for model/optimizer/EMA/progress,
leader actions and future random draws. CUDA2 passed; CPU131 passed/3 skipped.
The first CUDA attempt exposed missing CPU coordination in the raw-NCCL test
harness; using the native initializer resolved it without runtime bypasses.
All sessions terminal and fresh compute inventory empty; GPUs0-3 released.
Tiny native recovery only; full-weight, real reward/video, EMA artifact export
and production configuration remain open. Details in runtime integration report.

### Active claim: released Cosmos native online composition and recovery

GPUs0-3 reserved. GPUS0-1 run CP training; GPU2 owns local unsharded rollout;
native checkpoint RNG capture may access GPU3. Pinned Cosmos2B transformer and
scheduler from local NVMe, BF16 base/FP32 rank32 LoRA, CFG5, 480x832/33f latent
shape, two samples and two recorded CPS transitions from a 20-step scheduler.
Controlled text and rewards, no real encoder/VAE/Ray. Bounded control updates1-2,
then separate resume from checkpoint1, with EMA enabled. No long queue.
Fresh compute inventory empty before claim. Output planned under
`/mnt/nvme/outputs/wan22_i2v_cache/cosmos_native_online_recovery_l40s`.

### Released Cosmos native online recovery complete; claim released

Candidate9ea0adff, pinned full2B weights, CP2 training + disjoint GPU2 rollout,
480x832/33f latent shape, BF16 base/FP32 LoRA, CFG5, checkpointing and EMA.
Control updates1-2 succeeded:461.343s/453.836s, replay5.96e-8/0, nonzero grads.
Fresh-process checkpoint1 resume/update2 succeeded in464.016s with replay0.
Both ranks exactly match uninterrupted model/optimizer/EMA/progress, actions
and future RNG. Step2 has1120/1120 finite nonzero optimizer moments. Training
peak allocated9.06GiB/rank; rollout8.14GiB. Both jobs exit0; fresh GPU compute
inventory empty, all four GPUs released. Evidence under the output above and
`docs/research/cosmos_cp_runtime_integration_20260913.md`.

Controlled text/rewards and two CPS transitions only. Not complete video
generation/rewards, Ray, standalone EMA export, production config, quality,
single-card OOM unlock or matched throughput acceptance. These gates remain
open; H3/VDN authorization/name prerequisites and other family work remain.

### Active claim: real Cosmos complete-sequence generation

GPU2 claimed for the production registry rollout builder, real pinned text
encoder/VAE/transformer, 512x512/93f,20 CPS steps,no-CFG. This is one generated
clip for integration, not the256x32x8 paper training run. Prompt encoder
offloaded after encoding, native VAE tiling/slicing, BF16/IEEE. Full video,
conditioning and JSON retained; replay gate explicitly1e-3 for noise/logprob.
Fresh compute inventory empty. Other GPUs available; no CP trainer or Ray job.

### Real Cosmos full sequence complete; claim released

Candidate435c8fa2 fixes generic probe LoRA-only preset handling and retains
real conditioning/full MP4/result artifacts. Real512x512/93f,20 CPS,no-CFG
generation succeeds twice; both step0 noise/logprob replay errors0 under
explicit1e-3 guards. Decoded MP4 has93frames/16fps; two outputs byte-identical.
Visual output is blurry and does not establish prompt/physics quality.

Root-cache load bottleneck measured at~11.5MiB/s and~83ms read latency.
Text encoder/tokenizer/VAE copied and byte-verified into NVMe; subsequent
same-workload total88.1s versus986.4s root-cache run, with cache-warming caveat,
not a multi-GPU speedup. GPU2 released; all runs/copy/checks terminal and fresh
GPU inventory empty. Real conditioning retained for CP integration. Evidence:
`docs/research/cosmos_real_sequence_20260913.md`. Full training/reward/quality
and paper-budget gates remain open.

### Active claim: real Cosmos native reward scoring

GPU3 claimed for pinned Kling/VideoReward through the production reward
artifact path. Retained93f/512x512 real Cosmos clip, actual prompt, existing
overall_reward/min_frame_pixels preset, repeated twice with cleanup/shutdown
checks. All model caches on NVMe, offline. No policy training or quality claim.
Fresh GPU compute inventory empty before claim.

### Real Cosmos native reward scoring complete; claim released

Pinned Kling/VideoReward scored the retained real93f clip through production
RewardSample/artifact/runtime twice. Overall=-3.4963098037512346, identical
VQ/MQ/TA across repeats; first call38.25s includes lazy model load, warm0.95s.
Artifact cleanup and shutdown passed. Full model.pth strict restoration
verified despite initial Qwen base key-layout warnings; CPU loader12 passed,
1 optional skipped, explicit strict key-layout test1 passed. All processes
terminal, fresh GPU inventory empty, GPU3 released. Raw diagnostics retained
under `cosmos_real_reward_l40s`; preflight timer label caveat documented in
`docs/research/cosmos_real_sequence_20260913.md`. This is scoring repeatability,
not reward improvement or end-to-end CP training acceptance.

### Active claim: H3 random-weight cross-device dispatch

GPUs 0 and 1 reserved for a bounded tiny H3 transformer forward/backward
comparison against an identical single-device model. Fresh compute inventory
empty. No released weights, model downloads, Ray fleet or long training queue.
This tests dispatch mechanics, not full-model capacity or model quality.

### H3 random-weight dispatch complete; claim released

Two-GPU tiny H3 tests: 2 passed in 6.59 seconds. Explicit non-overlapping
block maps pass FP32 full gradients and frozen mixed-base FP32 LoRA gradients
against single-device references. Overlapping root/block mapping fails device
alignment; full BF16 base gradients fail the original 1e-3 comparison. These
negative results remain open, not hidden by the passing LoRA path. See
`docs/research/h3_four_l40s_preflight_20260912.md` for scope and reproduction.
All jobs terminal, fresh compute inventory empty, GPUs 0 and 1 released.

### Active claim: H3 partitioned checkpoint loading

GPUs 0 and 1 reserved for local random-weight sharded-checkpoint loading,
native replay builder/LoRA preparation, and single-device forward/gradient
comparison. Fresh inventory empty. No released weights or production queue.

### H3 partitioned checkpoint loading complete; claim released

Explicit family replay loader now loads block-owned checkpoint shards directly
and defers the native LoRA whole-model device move. Local random-weight
sharded checkpoint passes output/gradient comparisons and an AdamW update
with changed parameters on both GPUs. Entire H3 family: 31 passed in 7.30
seconds. No public trainer strategy or full-size generation acceptance implied.
All jobs terminal, fresh compute inventory empty, GPUs 0 and 1 released.
Evidence: `docs/research/h3_four_l40s_preflight_20260912.md`.

### Active claim: H3 four-device conditioner composition

GPUs 0-3 reserved for a bounded random-weight test: local Qwen conditioner
checkpoint on GPUs 2-3, native H3 prompt encoding, and tiny H3 DiT on GPUs 0-1.
Fresh inventory empty. No released weights, long training or throughput claim.

### H3 four-device conditioner connection complete; claim released

Native H3 prompt encoding now verified with a local sharded random-weight
Qwen3-VL checkpoint on GPUs 2-3, consuming the selected remote intermediate
state in the tiny DiT on GPUs 0-1. Prompt/repeated encoding/native one-step
prediction comparisons pass. Complete H3 suite: 38 passed in 5.78 seconds.
No VAE decode or full-size generation/throughput claim. All jobs terminal,
fresh compute inventory empty, all four GPUs released. Evidence retained in
`docs/research/h3_four_l40s_preflight_20260912.md`.

### Active claim: staged H3 conditioner and VAE lifecycle

GPUs 0-3 reserved for tiny four-device composition with frozen encoder CPU
parking, video/audio VAE decode on vacated GPUs, and encoder restoration after
normal and injected-error paths. Fresh inventory empty. No released weights.

### Staged H3 conditioner lifecycle complete; claim released

Frozen encoder parking and restoration now tested around video/audio decode
on vacated GPUs 2-3. Normal and injected-error paths restore exact device maps,
parameters, buffers and subsequent prompt embeddings. Full H3 suite: 40 passed
in 6.41 seconds. Unified generation entry, released-weight capacity and timing
remain open. All jobs terminal, fresh compute inventory empty; all GPUs released.
Evidence: `docs/research/h3_four_l40s_preflight_20260912.md`.

### Active claim: unified staged H3 generation builder

GPUs 0-3 reserved for explicit generation bundle construction, local random
DiT/encoder shard loading, native LoRA preparation, and automatic staged
video/audio decode. Modular small-component metadata is substituted in the
test; real released weights and full geometry are not being run.

### Unified staged H3 generation builder complete; claim released

Explicit generation bundle now combines partitioned DiT/encoder loading,
native shared LoRA preparation and automatic staged video/audio VAE decoding.
H3 GPU suite 41 passed; CPU placement/shared LoRA/Wan/Cosmos regressions 32
passed. Integration uses local random large-component shards and substituted
small-component metadata, not released full generation or executor acceptance.
All jobs terminal, fresh compute inventory empty, all GPUs released. Details:
`docs/research/h3_four_l40s_preflight_20260912.md`.

### Active claim: H3 native modular checkpoint round trip

GPUs 0-3 reserved for all-component local random-weight modular save/load,
explicit staged generation builder and decode, without substituted loaders.
Fresh inventory empty. No released checkpoint download or long queue.

### H3 native modular and executor gate complete; claim released

Unified test now saves and reloads all eight real tiny components through native
modular metadata without loader substitutes, then executes an actual request
through production MiniMaxH3BatchExecutor. Finite video/observations/actions/
log-probs and audio replay payload verified; staged placement restores afterward.
H3 suite: 50 passed in 7.11 seconds. Released weights, numerical replay and
performance remain open. All processes terminal, fresh compute inventory empty;
all GPUs released. Evidence in H3 preflight report.

### Active claim: H3 native executor-to-replay agreement

GPUs 0-3 reserved for tiny native modular generation and independent partitioned
replay loading. Re-score recorded actions in reverse timestep order with nonzero
LoRA B; check original 1e-3 log-prob/ratio gates and backward. No released weights.

### H3 executor-to-replay agreement complete; claim released

Independently loaded partitioned replay re-scores all three native executor
actions in reverse timestep order with exact log-probs and ratio 1 (both max
errors 0 under absolute 1e-3 guards). Backward has 16 finite gradient tensors,
12 nonzero. H3 suite: 50 passed in 7.27 seconds; raw JSON retained. Tiny model
only, no reward or trainer update. All jobs terminal, fresh compute inventory
empty, all GPUs released. Details in H3 preflight report.

### Active claim: standalone partitioned H3 generation probe

GPUs 0-3 reserved for the standalone local-checkpoint CLI, native generation
executor and MP4/result artifacts. Uses retained tiny modular checkpoint,
rank-32 preset LoRA and native VAE tiling. No released weights or long queue.

### Standalone H3 probe complete; claim released

New CLI produced verified 8-frame/16x16/24fps MP4, trajectory and timing JSON
from a local tiny checkpoint. Native rank-32 LoRA and tiling active. Fixed
generation-memory role leakage exposed by real preset use. Regression suite:
56 passed in 7.28 seconds. Tiny load 3.19s/generation 0.87s are not released-model
performance; stage-reset peak caveat retained. All jobs terminal, fresh GPU
inventory empty, all GPUs released. See H3 preflight report for reproduction.

### Active claim: full-size random H3 DiT capacity

GPUs 0-1 reserved for the pinned-config 33B/50-block DiT with random weights,
768x1344/124-frame native geometry and one first-step forward. No checkpoint
weights, encoder/VAE, LoRA or backward. This is actual DiT capacity evidence,
not released-model generation, quality or a throughput comparison.

### Full-size random H3 DiT capacity complete; claim released

Actual 33.123B/50-block H3 DiT, native 768x1344/124f input and 38,222 tokens,
completed a finite first-step forward on GPUs 0-1 in 18.8691s. Final run keeps
17.222M FP32 parameters and exact original RoPE buffers. Peak allocated
38.204/35.786 GiB, reserved 41.283/39.170 GiB. Random weights only; no encoder,
VAE, LoRA or backward, not a full-video timing or quality claim. Executed source
and result JSON retained. All jobs terminal, fresh inventory empty, GPUs 0-1
released. Full scope and preliminary failure notes are in the H3 preflight report.

### Active claim: full-size random H3 LoRA training capacity

GPUs 0-3 reserved for the full 33B DiT, native 768x1344/124f geometry, rank-32
FP32 LoRA, native gradient checkpointing, backward and one AdamW update.
Uses random base/adapter weights and squared-prediction loss, not a reward/RL
training claim. Fresh compute inventory empty. No released weights or queue.

### Full-size H3 LoRA training capacity complete; claim released

Full random 33B H3, native 768x1344/124f geometry and 38,222 tokens, rank-32
FP32 LoRA and native checkpointing completed forward/backward/AdamW update on
all four L40S. Forward 21.24s, backward 63.30s, optimizer 0.22s; 416/416 finite
nonzero gradients, parameters changed on every GPU. Peak allocated 36.95,
33.26, 34.85, 33.26 GiB. Squared-prediction loss only, not reward/GRPO or learning
acceptance; encoder/VAE and accumulated trajectories excluded. Executed source
and JSON retained. Process terminal, fresh inventory empty, all GPUs released.
Full evidence and scope are in the H3 preflight report.

### Active claim: full-size random H3 video VAE capacity

GPU 2 reserved for pinned-config full video VAE, FP32 weights, native CUDA
FP16 autocast/tiling, and 768x1344/124f decode. Random parameters, no released
weights. DiT, text encoder, audio and training are excluded. Fresh inventory empty.

### Full-size H3 video VAE complete; claim transferred

Full random VAE decoded 768x1344/124f with native tiling and finite output in
17.58s, peak allocated 13.66 GiB. Process terminal, fresh inventory empty.
GPU 2 released from VAE work. GPUs 2-3 now reserved for full random Qwen3-VL
conditioner encoding of 512 tokens and native CPU parking/restoration. No
released weights, real prompt quality or other component co-residency.

### Full H3 conditioner capacity and transfers complete; claim released

Full random 33.357B Qwen3-VL conditioner encoded 512 synthetic tokens through
the native hidden-state-50 helper on GPUs 2-3. Encode 0.52s, CPU parking 34.65s,
restore 12.99s, subsequent encode 0.18s with exact restored embeddings. Full
VAE separately decoded native 768x1344/124f in 17.58s. These probes establish
individual component capacities and expensive transfer waits, not co-residency
or quality. Both sources/JSON retained. All jobs terminal, fresh compute
inventory empty; GPUs 2-3 released. Details and next placement gate in H3 report.

### Active claim: full-size H3 co-resident encoder and video VAE

GPUs 2-3 reserved for full random Qwen3-VL conditioner (32/32 layers) with the
full FP32 video VAE on GPU 3. Native 768x1344/124f tiled decode, then encoding
again without parking. Fresh inventory empty. No released weights, DiT or audio.

### Full H3 co-resident decode complete; claim released

Full random conditioner plus full FP32 video VAE completed native
768x1344/124f tiled decode in 17.742230s without CPU parking. Repeat encoding
with both components and video still resident reproduced embeddings exactly.
Peak allocated bytes on GPUs 2/3: 39,776,483,840 / 45,508,336,640. This avoids
the isolated transfer interval, not a measured end-to-end speedup. No released
weights, DiT/audio co-residency or quality claim. Executed source and result
saved in h3_fullsize_random_co_resident_capacity; details in H3 preflight.
Process terminal and fresh compute inventory empty; GPUs 2-3 released.

### Active claim: explicit H3 keep-encoder decode regression

All four GPUs reserved for native tiny partitioned generation/replay tests of
both parking and keep-encoder decode policies. Fresh compute inventory empty.
This validates runtime behavior, not full released-model co-residency.

### H3 keep-encoder decode regression complete; claim released

Candidate `17b6f460` adds explicit keep-encoder decode plus CLI video-VAE
placement. Default parking unchanged; VAEs retain per-decode CPU cleanup.
Both policies pass native four-device generation/executor/replay/backward;
video/audio failure cleanup tests pass. Final suite 64 passed in 7.72s.
All processes terminal and fresh compute inventory empty; all GPUs released.
Full-size runtime throughput and released-weight gates remain open.

### Active claim: full H3 runtime decode policy timing

GPUs 0-3 reserved for same full random encoder/VAE and fixed latent decode
comparison, keep/park/park/keep order. Encoder on 2/3, VAE on 3, output on 0;
no DiT or audio loaded. Fresh compute inventory empty. Runtime includes VAE
CPU transfers; exact video and restored embedding checks are outside timing.

### Full H3 runtime decode timing complete; claim released

Same full random weights and latents, keep/park/park/keep: 23.310518,
69.011640, 66.715116, 21.643072 seconds. Mean keep 22.476795s vs park
67.863378s: 45.386583s saved (66.879%) for video decode including VAE
transfers only. All videos bitwise equal; repeated encoder embeddings exact;
VAE returned to CPU after each call. No DiT/audio/real weights/quality or
end-to-end speedup claim. Evidence: h3_fullsize_random_decode_policy_timing.
Process exited 0, fresh compute inventory empty; all GPUs released.

### Active claim: full H3 composed capacity

All four GPUs reserved for full random DiT on 0/1 and full conditioner on 2/3,
with native full video VAE decode on 3. Same-process real encoder output feeds
one native DiT forward/scheduler step before decode. No released weights,
audio VAE, full denoising or training. Fresh compute inventory empty.

### Full H3 composed capacity complete; claim released

Full random DiT on 0/1 and conditioner on 2/3 remained resident while native
video VAE decoded on 3. Actual encoder embedding conditioned one native
first-of-40 DiT forward (18.898932s), scheduler update, then native video decode
(23.047333s including VAE transfers). Finite outputs, exact post-decode encoder
embedding; peaks allocated 38.249/35.786/33.346/42.384 GiB. No audio VAE,
released weights, LoRA/training, complete denoising or quality claim. Evidence
and auxiliary receipt caveat in H3 report, h3_composed_random_capacity.
Process exited 0 and fresh compute inventory empty; all four GPUs released.

### Active claim: native Cosmos collector, one real group

GPUs 0/2/3 reserved for the preflighted native collector run: CPU replay
initial policy with trainer reservation 0, Ray generation 2, Kling reward 3.
Eight samples, 512x512/93f/20 CPS steps, IEEE, CPU trajectory storage. Fresh
compute inventory empty; no prior collector probe process. No CP update or
long training queue is being launched.

### Native Cosmos real collector complete; claim released

One actual 8-sample group at 512x512/93f/20 CPS steps completed through native
Ray generation, policy-v1 sync, Kling scoring and RolloutCollector. Generation
492.601s, reward 37.171s, total collect 529.813s, no overlap. Saved ~2.89GB
typed training payload and initial LoRA state in cosmos_native_collector_real_group.
CPU audit proves 8 unique samples, full 20-step axis, finite tensors/rewards,
all serialized storage CPU. Independent model replay and CP update still open.
Root Ray temp warning retained; next launch should put RAY_TMPDIR on NVMe.
Both processes exited 0; owned Ray PIDs absent and fresh GPU inventory empty.
GPUs 0/2/3 released. Details in Cosmos real-sequence report.

### Active claim: independent real Cosmos replay

GPU 0 reserved for fresh released-model replay of the saved native group.
Initial LoRA readback verified; all 8x20 transitions planned, reverse timestep
order, fixed 1e-3 log-prob tolerance. Stops after a failing sample to preserve
the counterexample. No generation/reward queue, CP or optimizer update.
Fresh compute inventory empty and candidate worktree clean.

### Independent real Cosmos replay complete; claim released

Fresh released replay model and readback-verified original LoRA rescored all
8x20 real saved transitions in reverse timestep order. 160 unique sample/step
pairs verified independently; max log-prob and ratio errors both 0 at fixed
1e-3 tolerance. Replay/verification 405.980s, peak allocated 7,739,583,488 bytes.
No CP, gradients/update, new generation or quality claim. Evidence in
cosmos_real_independent_replay and Cosmos real-sequence report. Process exited
0, fresh GPU inventory empty; GPU 0 released. Real-conditioned CP gate next.

### Active claim: real Cosmos CP admission

GPUs 0-1 reserved for native CP2 strategy replay of real samples 0 and 7,
steps 19/10/0. Fixed 1e-3 log-prob compatibility and exact rank-output checks;
not full-group or optimizer acceptance. Original Ray rollout lacked the CP
fixed-row compute contract, so exact original noise parity is not claimed.
Fresh compute inventory empty; no generation queue launched.

### Real Cosmos CP admission complete; claim released

Native CP2 on GPUs 0/1 passed real samples 0/7 at steps 19/10/0: max saved
rollout log-prob error 1.847744e-6 at fixed 1e-3, rank prediction error 0.
Six transitions only, not full-group gradients/update. Total 71.117s, each
rank peak allocated 6,620,361,216 bytes. Different fixed-row compute contract
prevents fair speedup comparison with prior native replay. Matched-compute
single-device control and real CP update remain open. Evidence in
cosmos_real_cp_replay_admission and Cosmos real-sequence report. Torchrun
exited 0, probe PIDs absent, fresh GPU inventory empty; GPUs 0/1 released.

### Active claim: Cosmos matched-compute single-device control

GPU 0 reserved for the same six real sample/step inputs as CP admission,
fixed-row Linear, FP32 LoRA and efficient SDPA but no CP hooks. This measures
the unsharded compute control; CP run additionally included rank-consistency
collectives. Fixed 1e-3 saved-rollout log-prob tolerance. Fresh inventory empty.

### Cosmos fixed-compute control complete; claim released

Same six real inputs: fixed-row single-device 125.184s vs CP2 diagnostic
71.117s (observed 1.760x phase ratio, single run per arm; CP also includes
consistency collectives). Both max saved-rollout LP errors 1.847744e-6.
Single-device peak allocated 7,977,218,048 bytes. Ordinary native replay is
much faster than either fixed-row path, so do not promote CP as production
speedup at this geometry. Public CP dispatch remains disabled; update semantics
and compute-cost gates open. Evidence in cosmos_real_fixed_replay_control and
Cosmos report. Process exited 0, fresh inventory empty; GPU 0 released.

### Active claim: real Cosmos CP single-slice update

GPUs 0/1 reserved for all eight saved real samples at timestep 10: native
whole-group advantages, GRPO, eight accumulated backwards, CP reduction and
trainer clip/optimizer boundary. No full multi-timestep/PPO recipe or parity
claim. Fresh compute inventory empty; no generation or reward queue.

### Real Cosmos single-slice capture failure; claim released

All eight microbatch backwards completed (767.117s), max LP error 4.082918e-6,
rank prediction error 0. Probe failed after native clip/step: that boundary
clears gradients before returning, so the probe's late gradient dictionary was
empty. Do not mark update acceptance passed; no final state/rank proof saved.
Failure/source/transition receipts retained in cosmos_real_cp_single_slice_update.
Corrected reusable probe captures gradients via optimizer pre-hook and saves
per-rank raw gradients before stepping. Real CPU boundary regression: 1 passed
in 9.01s. Corrected GPU run not executed yet. Torchrun exited 1, regression 0;
fresh compute inventory empty, GPUs 0/1 released. Full details in Cosmos report.

### Active claim: corrected real CP gradient capture

GPUs 0/1 reserved for the unchanged real eight-sample single-time-slice update,
with validated optimizer pre-hook capture and per-rank pre-step snapshots.
Previous captured output directory absent and no old probe process; fresh GPU
inventory empty. Original failed artifact retained, no new generation queue.

### Corrected real CP capture passed; claim released

All eight real samples at timestep 10 completed native CP backwards and one
optimizer boundary. Norm 0.0008163046441; 560 gradients captured, 280 nonzero,
280 trainable tensors changed; updated rank parameters exact. CPU artifact
audit confirms raw CP rank-gradient sum to captured post-clip gradients has
max error 0, all optimizer moments finite, 560 entries at step 1. Scope is one
time slice, not full recipe or unsharded-gradient parity. Region 770.364s,
peak allocated 10,225,359,872 bytes per rank. Evidence in
cosmos_real_cp_single_slice_update_captured; original failure retained. Both
jobs exited 0, probe PIDs absent, fresh GPU inventory empty; GPUs 0/1 released.

### Active claim: unsharded real gradient reference via DP4

All four GPUs reserved for fixed-row/FP32-LoRA unsharded computation on the
same eight samples at timestep 10. Native DDP4 assigns two unique samples per
rank; local loss/2 plus DDP mean gives the global eight-sample mean. No CP
hooks and no single-device timing claim. Compare saved reference with prior
CP update, without regenerating data. Fresh compute inventory empty.

### Real unsharded reference and CP comparison passed; claim released

Native DP4 processed two unique samples per rank, same fixed compute contract,
whole-group advantages and timestep 10. Eight-sample mean update complete;
all final rank parameters exact. Independent CP2 comparison: gradient relative
L2 3.406e-7, update relative L2 7.099e-7, parameter max abs 1.119e-8; Adam
moment relative L2 3.424e-7/5.007e-7. All fixed tolerances passed. This is
single-slice unsharded-reference equivalence, not serial bitwise/full recipe.
DP4 region 288.514s, peak allocated 15,153,960,960 bytes/rank; no same-resource
speedup claim against CP2. Evidence in cosmos_real_dp4_unsharded_reference.
Both jobs exited 0, fresh GPU inventory empty; all four GPUs released.

### Active claim: ordinary native DP4 update control

All four GPUs reserved for the same real eight samples at timestep 10, now
ordinary Linear/autocast and default SDPA selection under strict IEEE.
No fixed-row override. Identical advantages, DP sample ownership, optimizer
and capture. Compare to fixed-compute CP with unchanged prior gradient/update
thresholds; no equivalence assumed. Fresh compute inventory empty.

### Native DP4 completes; fixed-contract equivalence fails; GPUs released

Native eight-sample single-slice DDP update ran in 36.435s, saved-rollout LP
errors all 0, final rank parameters exact. Both fixed CP2 and fixed DP4 differ
from native DP4 by ~0.525884 gradient relative L2 and ~0.561716 update relative
L2, failing unchanged limits. Same-DP4 comparison isolates combined compute
overrides, not CP partitioning; individual causes not yet isolated. Earlier
matched fixed-contract parity does NOT establish native-training equivalence.
Keep CP out of public/default dispatch. Evidence and failed comparisons in
cosmos_real_native_dp4_update, details in Cosmos report. GPU run exited 0,
comparisons exited 2; fresh inventory empty, all four GPUs released.

### Active claim: efficient-SDPA-only ablation

All four GPUs reserved for same real DP4 single-slice update, changing only
default attention selection to efficient SDPA. Native Linear and LoRA autocast
unchanged. Same fixed comparison limits; no new data generation. Fresh GPU
inventory empty before launch.

### Attention-only ablation complete; native parity fails; claim released

Efficient SDPA alone, with native Linear/LoRA unchanged, produces gradient
relative L2 0.522900 and update relative L2 0.547094 vs native DP4, failing
unchanged limits. Thus this override alone is sufficient for this workload's
parity failure; other override effects not excluded. Execution itself passed,
rank parameters exact, max LP error 4.321337e-6; region 100.509s vs native
36.435s. No general backend defect/quality conclusion. Evidence in
cosmos_real_efficient_only_dp4_update. GPU job exited 0, comparison 2;
fresh inventory empty, all four GPUs released. CP remains experimental.

### Active claim: H3 independent VAE placement

GPUs 0 and 1 reserved for native tiny video/audio VAE decoding with CUDA 0
latents and VAEs on CPU or CUDA 1. Fresh compute inventory empty. This checks
device-correct decode and return placement, not full H3 capacity or quality.

### H3 independent VAE placement complete; claim released

Native video/audio decode now follows each VAE's actual device and returns
outputs to the input device without moving/recasting the VAE. CUDA 0 inputs
with CPU or CUDA 1 VAEs match local-device controls exactly. H3 family suite:
39 passed, 1 four-GPU opt-in skipped in 10.12 seconds. Full VAE capacity and
staged conditioner lifecycle remain open. All processes terminal, fresh GPU
compute inventory empty; GPUs 0 and 1 released. Evidence in H3 preflight report.

### Active claim: fixed-row-only Cosmos DP4 ablation

GPUs 0-3 reserved after fresh empty compute inventory. Same saved real eight
samples and timestep 10, native optimizer and full-group advantages. Only
Linear row partitioning changes to 64; native LoRA autocast and default SDPA
remain. Unchanged native-parity tolerances; no rollout regeneration.

### Fixed-row-only ablation complete; native parity fails; claim released

Only 64-row Linear partitioning, retaining native SDPA and LoRA autocast,
failed unchanged native DP4 limits: gradient relative L2 0.571106, update
relative L2 0.575561. Max LP error 3.8146973e-6 still passed replay admission;
four final parameter replicas exact. Diagnostic region 244.958s versus native
36.435s, not a full-recipe benchmark. Removing efficient SDPA alone is therefore
insufficient; FP32-LoRA and interactions still unresolved. Artifacts in
cosmos_real_fixed_rows_only_dp4_update. GPU process 0, CPU comparison 2,
fresh compute inventory empty; all four GPUs released. CP remains experimental.

### Active claim: FP32-LoRA-only Cosmos DP4 ablation

GPUs 0-3 reserved after fresh empty compute inventory. Same real eight-sample
single-slice native DP4 harness; only LoRA A/B Linear autocast is disabled with
FP32 inputs. No fixed rows or attention override. Probe wrapper has CPU checks
for precision, finite gradients and restoration on normal/exception exits.
Native-parity tolerances unchanged; no new rollout or long training queue.

### FP32-LoRA ablation complete; claim transferred to native repeat

GPU process exited 0; comparison exited 2. FP32-LoRA-only gradient relative
L2 0.00298788, update 0.00668862, still above unchanged limits despite all LP
errors zero. Region 38.486s. CPU helper tests: 2 passed in 8.19s. Fresh GPU
inventory empty. GPUs 0-3 now reserved for an unchanged fresh-process native
DP4 repetition, to check whether the original reference is reproducible.

### Native reference repeat complete; all four GPU claims released

Fresh unchanged DP4 repetition matched all captured gradients, updates, final
parameters and both Adam moments exactly (all maximum/relative errors zero).
All LP errors zero; region 36.454606s versus original 36.435179s. GPU process
and comparator both exited 0; fresh compute inventory empty. Evidence in
cosmos_real_native_repeat_dp4_update. All three single-factor overrides fail
native parity; fixed CP remains experimental, not a production replacement.
Native single-device accumulation equivalence and full-recipe gates stay open.

### Active claim: native single-device accumulation control

GPU 0 reserved after fresh empty compute inventory. Same real eight samples,
timestep 10 and full-group advantages as native DP4; SingleProcessStrategy
accumulates loss/8 for all eight before the same native optimizer boundary.
No compute overrides, new rollouts or tolerance changes. GPUs 1-3 unclaimed.

### Native single-device versus DP4 parity passed; GPU 0 released

Same real eight-sample single-timestep update passed unchanged limits:
gradient relative L2 8.5849871e-11, update 3.3058375e-10, parameter max absolute
2.9103830e-11; Adam moments also passed. LP errors all zero. Single-device
140.464884s versus DP4 36.435179s (repeat 36.454606s), approximately 3.85x
for this diagnostic training phase only, not end-to-end throughput. Evidence
in cosmos_real_native_single_update. GPU job/comparator both exited 0, fresh
compute inventory empty, GPU 0 released. Full multi-timestep/PPO recipe,
production synchronization, streaming normalization, EMA/recovery and quality
still required. Fixed-compute CP remains experimental.

### Active claim: native OnlineTrainer cached-group full PPO cadence

GPUs 0-3 reserved after fresh empty inventory. CPU preflight passed for native
OnlineTrainer integration: saved real eight-sample group explicitly split two
samples/rank, full-group advantages supplied before splitting. Native selection
0,2,...,18 and four PPO epochs retained: 80 training evaluations/rank and four
optimizer updates, EMA enabled. Cached-batch injection does not test production
group ownership, live rollout/reward or weight synchronization. Candidate clean
435c8fa2; external probe cosmos_native_trainer_probe.py, output
cosmos_native_trainer_full_ppo. No precision override or tolerance relaxation.

### Native OnlineTrainer full cached-group cadence passed; claim released

All four PPO epochs over ten selected timesteps completed through native
OnlineTrainer. Four optimizer boundaries/rank, final trainable replicas exact,
initial global replay error zero, four Adam/EMA updates confirmed. Region
1397.785s; peak allocated 15,665,102,848 bytes/rank. All four GPUs repeatedly
observed at 100% utilization. CPU artifact audit and independent unfused
AdamW/EMA reconstruction passed: parameter max error 7.450581e-9, EMA max
1.490116e-8, first/second moment relative L2 2.153454e-7 / 1.292140e-5.
Existing CPU streaming regressions also passed 12 tests in 11.34s while GPU
work ran; timing is not an isolated benchmark. GPU and both audit processes
exited 0; fresh GPU inventory empty, all four GPUs released. Evidence in
cosmos_native_trainer_full_ppo. This cached one-group, explicitly sample-sharded
integration does not close production group ownership, live rollout/reward,
weight sync, real-GPU global_std streaming, recovery or quality gates.

### Active claim: fresh native four-rank trainer state restoration

GPUs 0-3 reserved after fresh empty compute inventory. Reconstruct real native
DDP trainer and strictly restore saved model, optimizer, EMA and counters from
cosmos_native_trainer_full_ppo/final_state.pt. Compare every exported tensor
and metadata field exactly, and verify load resets rollout/replay readiness.
This does not replay training or claim RNG/recipe/full-checkpoint recovery.

### Fresh native trainer restoration passed; all GPU claims released

All four new processes strictly restored 560 model tensors and 2240 trainer
state tensors exactly, including Adam/EMA and metadata. step=1/global_step=4,
EMA updates=4 retained; deliberately stale rollout/replay readiness flags reset
correctly. Restored replicas exact across ranks. Load/readback region ~0.742s
excludes startup/model loading. Evidence cosmos_native_trainer_fresh_restore.
GPU job exited 0, fresh compute inventory empty; all four GPUs released.
This is state rehydration only: native checkpoint file integration, saved
per-rank RNG/recipe progress and uninterrupted-versus-resumed next update remain
open, along with production rollout/weight sync and quality acceptance.

### Active claim: native checkpoint file and per-rank RNG roundtrip

GPUs 0-3 reserved after fresh empty inventory. Separate four-process writer
and reader launches use native save_training_checkpoint, TrainingCheckpoint
and strict restore_training_checkpoint on the real four-update snapshot.
Probe progress is explicitly marked, not a production iterator. Verify future
Python/NumPy/Torch CPU/all-visible-CUDA/named RNG draws exactly per rank.
No new training or live rollout; next-update equivalence remains separate.

### Native checkpoint and rank RNG roundtrip passed; claim released

Separate four-process writer/reader passed native schema-v2 checkpoint APIs.
checkpoint.pt 918,715,353 bytes; actual model identity and explicitly scoped
probe next_epoch=1/next_step=4 validated. Every rank restored 560 model and
2240 trainer tensors plus metadata exactly. Python/NumPy/Torch CPU/all-visible
CUDA/named generator future draws matched exactly; four rank CPU streams
independently verified distinct. Writer did not advance captured streams.
Both GPU jobs and CPU distinct-stream check exited 0; fresh GPU inventory
empty, all four GPUs released. Evidence cosmos_native_checkpoint_roundtrip.
Next training-update equivalence, production iterator/progress continuation,
live rollout synchronization and quality are not closed by this roundtrip.

### Active claim: actual next-update save/continue versus fresh resume

GPUs 0-3 reserved after fresh empty inventory. Writer reconstructs the real
four-update state, saves native checkpoint, then continues one native streaming
begin/backward/finish optimizer update over all ten selected timesteps. Fresh
reader restores that checkpoint and performs the same update. Same cached old
actions, full-group advantages and balanced explicit two-sample/rank ownership.
This checks the next optimizer boundary, not a fresh rollout or production data
cursor. No compute override or changed parity tolerances. Compare final gradients,
model/Adam/EMA state and future RNG, preserving failures if any.

### Actual next-update recovery equivalence passed; all GPU claims released

Save/continue and separate fresh-resume arms each completed the native
ten-timestep next update, global_step=5/EMA updates=5. Exact CPU comparison:
560 gradients, 560 model tensors, 2240 trainer tensors plus metadata, and all
four ranks' post-update future RNG streams. All 560 Adam counters are five;
trainer step=2. Final trainable replicas exact in each GPU arm. Rank-0 update
regions 346.828555s and 346.742082s, not restart latency or a speedup benchmark.
Evidence cosmos_native_next_update_resume. Both GPU jobs and comparator exited
0; fresh compute inventory empty, all four GPUs released. Real cached-group
next-update equivalence is proven at this boundary; production iterator,
fresh rollout/weight synchronization, continuous queue recovery and quality
remain open.

### Active claim: four real Ray workers, trained checkpoint content delivery

GPUs 0-3 reserved after fresh empty compute inventory and no live raylet/GCS.
Use existing vrl.scripts.perf.weight_delivery_probe with four isolated workers,
native strict restore from cosmos_native_checkpoint_roundtrip/checkpoint.
Receivers are intentionally poisoned, then two versioned in-place installs
require native parameter-content readback on every rank. This is delivery
acceptance, not video forward/quality or a production rollout scheduling test.
Private Ray temporary state is on /mnt/nvme/ray, not the near-full root disk.

### Four trained Ray receiver delivery passed; all claims released

Native isolated four-worker acceptance passed poison -> trained snapshot v1 ->
same snapshot v2, with exact content readback and version ACK from each receiver.
560 tensors / 183,500,800 bytes each. Snapshot fleet sync/readback 0.687883s and
0.503922s, excluding worker load; no generation throughput claim. Source is the
real four-update checkpoint. Native tool regressions: 10 passed in 62.99s.
GPU job exited 0, fresh compute and raylet/GCS/probe process queries empty,
all four GPUs released. Evidence cosmos_trained_four_worker_delivery. No video
generation, retained-version slots, live production scheduling or quality proof.

### Active claim: fresh trained two-worker generation and reward

GPUs 0-3 reserved for the native placement owner after empty compute inventory
and no live Ray cluster. CPU preflight verified two rollout engines on GPUs 1/2,
reward GPU 3 and trainer reservation GPU 0 (actual sender model on CPU).
Restore trained step-four checkpoint, native weight push, generate one real
eight-sample 512x512/93-frame/20-step group, then verify both workers' active
parameters against the sender snapshot. RAY_TMPDIR on NVMe. Independent replay
and quality comparison remain subsequent gates; no stale queue is restarted.

### Trained fresh collection passed; claim transferred to independent replay

Eight new samples collected in 286.722s: generation 248.959s, reward 37.723s,
overlap zero. Both rollout workers passed active trained-weight readback after
generation; 560 sender tensors differ from initial untrained adapters. CPU
trajectory audit passed all finite tensors/rewards, eight unique sample IDs,
20 steps and CPU-only serialized storage. Ray driver exited 0, fresh compute
and Ray process inventories empty. All four GPUs now reserved for four fresh
independent replay models, two samples each, covering all 160 sample/step pairs.

### Fresh trained multi-worker rollout/replay chain passed; claims released

Independent replay covered all 160 unique sample/step pairs with log-prob and
ratio errors both zero at unchanged 1e-3 tolerance. Four fresh models used
trained collection weights; no DDP/CP/update. Replay regions 101.396-101.748s,
peak allocated 7,739,583,488 bytes each. CPU coverage audit passed. Eight fresh
generated samples were verified as four per rollout worker on GPUs 1/2, all
version 1, with post-generation active trained-weight readback. Real reward
on GPU 3; generation/reward overlap zero. No controlled learning/speedup claim.
Evidence cosmos_trained_collector_real_group and cosmos_trained_independent_replay.
Ray and replay jobs/audits exited 0; fresh compute inventory empty, all four
GPUs released. Live trainer/worker orchestration, production data progress,
continuous recovery and quality remain open.

### Reward overlap preflight: CPU only, no GPU claim

Saved trained collector config has one prompt group and blocking local reward,
despite verified dedicated accelerator isolation. Native overlap capability is
false; forced streaming correctly fails before generation. Validated receipt:
cosmos_reward_overlap_preflight_validated/result.json. Focused collector,
orchestration and overlap-benchmark CPU tests: 68 passed in 1.29s. No models
loaded, no GPU benchmark or overlap fix claimed. Next: verified asynchronous
reward service plus at least two real groups; equal-work batched-serial,
per-group-serial and streaming controls with correctness and measured overlap.

### Active claim: native Cosmos HTTP reward capability

GPU 3 reserved after empty compute inventory for an isolated pinned Kling
service, CUDA_VISIBLE_DEVICES=3. Client CPU only; GPUs 0-2 unused. Verify real
HTTP scoring, historical same-artifact score equality, asynchronous runtime
capability and native collector admission. No concurrent generation or speedup
claim in this gate. Candidate 435c8fa2 unchanged.

### HTTP reward capability passed; GPU 3 released

Native collector overlap capability false before service preflight, true after;
HTTP scorer nonblocking, isolated service GPU mask 3, client CUDA uninitialized.
Two real pinned Kling calls matched historical same-video in-process component
score exactly (-0.6127294366809066). First call 36.455s includes lazy load; warm
call 1.018s. No generator or measured generation overlap. Evidence in
cosmos_http_reward_capability, including executed source and service config/log.
55 native service CPU tests passed in 0.88s. Client/service exited 0, fresh GPU
and service process inventories empty. GPU 3 released; no background service.
Next: equal-work multi-group A/B/C generation/reward comparison, not another
single-video reward compatibility check.

### Active claim: real Cosmos multi-group A/B/C collection pilot

GPUs 0-3 reserved for native placement after empty GPU/Ray/service inventories.
Unchanged candidate 435c8fa2. CPU preflight passed two PromptExample groups with
explicit seeds 12340/22340, eight samples/group, 512x512/93f/20 CPS steps.
Rollout GPUs 1/2, owned HTTP reward GPU 3, trainer role 0 reserved but CPU sender.
Warm both rollout workers and reward, then A batched serial, B per-group serial,
C streaming on the same fleet, pinned service and trained step-four weights.
Save every arm's actual trajectories, native timings and tensor fingerprints;
check equal work and output equality. One run/arm is a pilot, not confidence
or end-to-end training acceptance. Estimated generation about 25 minutes.

### Multi-group A/B/C pilot passed correctness; all claims released

Same trained policy, explicit seeds, two groups x eight real clips per arm:
A batched serial 520.154111s, B per-group serial 518.685593s, C streaming
508.315438s. Native overlap A/B zero, C 10.018006s. All stored trajectory tensor
hashes and all rewards match exactly; native counters confirm equal work and
one/two/two reward calls. Both workers passed final trained-weight readback.
GPU timeline captured generator GPUs 1/2 at 100% while reward GPU 3 was active.
CPU independent audit passed. Evidence cosmos_overlap_abc_real, with full
trajectories, executed sources, service config/log, receipts and GPU timeline.

Observed C vs A improvement 2.276%, 1.02329x, one fixed-order warmed run only.
No repeated confidence, 10% gate, optimizer or full training speedup claim.
Do not automatically expand this into a long reward-overlap campaign: generation
still takes about 497s, versus about 21-24s scoring including materialization.
Prioritize generation throughput and live production training orchestration.
Driver/service/monitor/audit exited 0; fresh GPU and Ray/service inventories
empty, all four GPUs released. H3 authorization and exact dqn model identity
were requested asynchronously again; no released H3 weights downloaded here.

### Active claim: checkpoint named-RNG CUDA regression

GPUs 0-3 reserved after empty compute inventory for two/four-rank NCCL unit
checkpoint RNG roundtrips and cross-device state equality. Candidate has a
scoped strict-resume fix: missing requested prompt_generator now fails before
RNG mutation rather than silently keeping the new seed. CPU regression 183
passed, 2 CUDA cases skipped; ten new tests reproduced failure before the fix.
Actual Cosmos step-four probe checkpoint contains probe, not prompt_generator,
on all ranks and is correctly rejected for production prompt RNG continuation.
No production data progress is invented or checkpoint content changed.

### Strict named-RNG fix verified; GPU claims released

Candidate d46766ec rejects missing requested generator names in strict restore
before RNG mutation; non-strict mode warns and retains missing streams. Ten new
tests reproduced the old silent omission. CPU broad regressions 183 passed/2
CUDA skipped; final focused checkpoint-RNG/sampler 39 passed/2 CUDA skipped.
Actual two/four-rank NCCL checkpoint RNG roundtrips passed, as did cross-device
tensor equality. Ruff and diff checks passed. Actual trained step-four checkpoint
probe proves all ranks have probe but lack production prompt_generator; original
probe streams remain exact while strict production RNG admission now fails.
Evidence cosmos_checkpoint_prompt_rng_admission with executed source and receipt.
No production data cursor fabricated, no checkpoint rewritten. All processes
exited and fresh compute inventory empty; all four GPUs released. Genuine online
recipe checkpoint/data continuation remains the next integration requirement.

### Production RNG preflight moved before model startup; CPU only

Candidate 30e11971 adds pure topology/named-stream validation reused by restore.
Online entry checks every saved rank before prompt/model/Ray construction.
Missing rank-3 stream is covered by rank-0 entry regression; valid preflight
does not mutate RNG. CPU suite 186 passed/3 CUDA skipped; final focused 3 passed.
Actual step-four checkpoint pure preflight passed its expected admit/reject
checks with CPU RNG unchanged and no CUDA initialization. Evidence
cosmos_checkpoint_rng_preflight. No GPU job/claim; all test/probe sessions exited,
Ray/reward process inventory empty. DDP production colocation remains gated on
collective parking; disjoint multi-rank rollout remains gated on owner design.
Do not represent standalone DDP tests as complete production orchestration.

### Active claim: four-GPU native FSDP parking update equivalence

GPUs 0-3 reserved after empty compute inventory. Extend existing real CUDA
adapter-only FSDP parking test with resident-versus-parked three-update control:
native BF16 frozen base/FP32 adapter, live gradients, Adam moments, native EMA
including temporary weights, frozen reference model and original device/identity
restoration. Unit policy, not Cosmos throughput or full production rollout.

### Claim transferred: real Cosmos FSDP update and parking

Both four-GPU unit parking tests passed (13.12s); candidate 260b7d19 commits the
new three-update exact control. All test processes exited. Retain GPUs 0-3 for
one actual Cosmos native FSDP slice update on the original eight real trajectories,
two samples/rank at step10, identical whole-group advantages/native optimizer.
Park/restore live gradients before update and Adam/EMA state after update, exact
local readback including frozen parameters. Compare reconstructed full gradients
and updates against the preserved native DP4 reference at unchanged limits.
No long training queue, no confidence/performance or full recipe claim.

### Real FSDP parking/update passed; all GPU claims released

Four-GPU unit resident-versus-parked three-update exact control committed as
candidate 260b7d19; both CUDA parking tests passed in 13.12s. Real Cosmos native
adapter-only FSDP4 then processed the original eight samples at step10: LP error
zero, whole-group advantages unchanged. Per rank, 2,809 tensor checks passed
with live gradients before update and 3,929 checks with Adam/EMA after update;
CPU parking/original device and parameter identity restoration exact.

Independent full-state CPU comparison to native DP4 passed unchanged thresholds:
gradient relative L2 7.895697e-11, update relative L2 3.328797e-10, parameter
maxabs 2.910383e-11; Adam moments also pass. Peak allocated 14,846,795,776 bytes
per rank. Diagnostic wall 53.537678s includes parking/readback/gather, not a fair
training speed comparison. Evidence cosmos_real_fsdp_parking_update with sources,
full gradients/update state and receipts. Tests/torchrun/comparator exited 0,
fresh compute inventory empty; all four GPUs released. Complete allocator
reclamation, live generation handoff, production data/resume and full recipe
remain open. Prefer investigating this supported native FSDP lifecycle over
bypassing DDP's collective parking guard or enabling failed fixed-compute CP.

### Active claim: real four-rank FSDP to Ray generation handoff

GPUs 0-3 reserved after empty compute inventory. Native per-rank colocated
resource preflight passed. Repeat the bounded real FSDP slice update, export
updated full weights, collectively park all trainers with Adam/EMA, then run
one independently owned Ray worker on each same physical GPU. Each generates
one 512p93f20-step clip and verifies active weights against the parent snapshot.
Native zero-reward batch construction only; no reward model or learning claim.
Each child caps Ray object store at 2GiB on NVMe and must exit/clean before all
parents restore. Verify exact trainer-state restoration after actual generation.
Candidate 260b7d19 unchanged; no production queue or full recipe claimed.

### Handoff probe activation omission rejected; retry after cleanup

First attempt failed before worker activation: deferred runtime requires the
schedule to await activate() before generate(). The standalone probe omitted
this call. All child drivers cleaned their owned Ray clusters; all FSDP ranks
agreed on failure, restored trainer state, and exited. Fresh GPU and Ray/child
inventories empty; failed parent/child sources and logs preserved in
cosmos_real_fsdp_generation_handoff. Add the native activate() step, retain
the same four-GPU claim, and retry into a distinct *_activated output directory.

### Generation fit observed; remove erroneous reward invocation from probe

Activated attempt generated all four clips and passed active-weight verification,
then failed because the probe invoked score_rollouts with no reward function in
a shared-reward topology. This is a probe error, not reward acceptance. Preserve
activated sources/logs/timeline. All ranks coordinated failure and exited; fresh
compute and Ray/child inventories empty. Observed each GPU retained a 1084MiB
trainer plus a 26530MiB generation worker. Correct generation-only scope: save
native GenerationOutput, validate its trajectory, explicitly offload runtime,
clean child workers, then restore parent state. Retry distinct *_unscored output;
same four-GPU claim, no reward gate bypass or claimed scoring success.

### Real four-card generation handoff passed; all claims released

Final *_unscored attempt completed real FSDP update -> parked live trainers ->
four rank-local native Ray workers generating one 512p93f20-step clip each ->
active updated-weight verification -> explicit offload/worker cleanup -> exact
parent restoration (3929 tensors/rank). Parent parked allocation 189665280 bytes;
saved nvidia-smi inventory shows 1084MiB trainer plus 26530MiB generator per GPU.
Generation 61.901-62.187s/clip, all four physical assignments verified. Native
uint8 videos [1,3,93,512,512], CPU storages/finite trajectories pass artifact audit.
Source generation weights equal restored trainer export. Native DDP gradient,
update and Adam comparisons pass unchanged limits. No reward invocation, next
post-handoff update, persistent-worker-cycle or full production-resume claim.
Evidence cosmos_real_fsdp_generation_handoff_unscored with full artifacts, sources,
logs, timeline and audits. All children/parent/monitor/audits exited 0, fresh GPU
and Ray/child inventories empty; all four GPUs released. Failed earlier probe
attempts remain preserved separately, not counted as passes.

### Claim: controlled real FSDP next update after generation handoff

Reserve GPUs 0-3 for sequential resident-control and real-generation-handoff
arms, each two identical cached-group single-slice native updates. Same initial
weights, samples, global advantages, optimizer and EMA. Only the experimental
arm parks after update one, generates four full clips through native rank-local
Ray workers, cleans them up and restores before update two. Compare full model
updates, gradients, Adam and local EMA exactly. This isolates handoff semantics;
not fresh-data training, production integration, quality or a speed benchmark.
Candidate remains 260b7d19; external probe only. Fresh compute inventory empty
before claim; no old continuous queue restarted.

### Controlled post-handoff next update passed; GPUs released

Resident and handoff arms each completed two identical real cached-group
single-slice native FSDP updates. Handoff included four actual full512p93f20-step
Ray-generated videos between updates and exact3929-tensor/rank restoration.
Both updates match resident control exactly in raw/clipped gradients, trainable
parameters, Adam moments/counters, advantages, local EMA and log-prob changes.
Second update changes all560 trainables; Adam/EMA counters2 on both arms.
Four clips, physical assignments, active weight version1, finite CPU trajectories
and source weights pass independent artifact audit. All two torchrun jobs,
children and CPU comparator exited0; fresh GPU and Ray/child inventories empty.
Release GPUs0-3. Evidence cosmos_fsdp_next_update_{control,handoff}, executed
sources, adjacent logs and handoff/comparison.json. Diagnostic79.695s versus
269.014s includes different work and is not a speed comparison. No fresh-data
update, reward, persistent-worker or complete production integration claim.

### Claim: native Cosmos four-rank online recipe integration

Reserve GPUs0-3 for a bounded actual CLI run, not a cached/proxy trainer. Add
four-rank colocated FSDP preset preserving global32x8 default workload. Hardware
launch overrides: two epochs, one prompt/rank, two samples/prompt, one PPO pass,
one of20 training timesteps, checkpoint each epoch; unchanged512p93f20 generation,
real pinned Cosmos and Kling weights, native strict lifecycle and data sampler.
This smaller run is an integration gate only, not full-recipe/quality acceptance.
Fresh compute inventory empty; no prior jobs or old queues restarted.

First native CLI attempt fff0e73f failed with OOM in training forward after all
four ranks generated/scored real videos and passed initial replay parity at0.
Cause identified in effective configuration: actor.gradient_checkpointing absent
means off, while prior accepted model probes explicitly enabled it. No optimizer
checkpoint/update completed. All ranks/torchrun exited1, fresh GPU/Ray inventories
empty. Preserve cosmos_native_fsdp_online with original launch/config/verdicts,
reward decoder/results receipts and adjacent log. Add explicit full activation checkpointing
to four-card preset and its budget/ownership tests; retry separate *_gc directory
with identical workload. Retain GPU0-3 claim; do not relax geometry or parity.

### Native two-epoch Cosmos integration passed; GPUs released

Candidatef55aca2a with explicit full activation checkpointing completed actual
four-rank CLI online recipe: two epochs, global8 fresh samples/epoch, real Kling
reward, native strict FSDP phase switching and persistent Ray workers. Both
updates finite/nonzero, pre-update LPdiff0, all560 trainables change on update2.
Checkpoint1/2 Adam+EMA counters1/2, full1120 owned tensors valid, final model and
trainer exactly equal checkpoint2. Native four-rank prompt RNG admission passes;
sampler transition from checkpoint1 to2 reproduces disjoint179,298,39,13 indices.
16 scored/decode-receipted videos; native temporary media cleanup verified.
Exactly four generation policy loads/PIDs for two epochs; warm activation~3s vs
cold~39s, warm reward wall~7.4s vs cold~58s. Different prompts/cold-vs-warm is not
a speed or quality comparison. Five preset tests and final independent CPU audit
pass; failed initial no-GC run and two mistaken audit assumptions preserved.
Evidence cosmos_native_fsdp_online_gc and adjacent log. All four success verdicts,
torchrun/audit exit0, fresh GPU/Ray inventories empty: releaseGPU0-3. Full recipe,
fresh-process resume, controlled throughput and quality remain open.

### Resume RNG gaps found before launching another video run

Inspection found unseeded collector requests rely on uncheckpointed Ray worker
Python/CUDA randomness. Add driver-owned random seed when unspecified; explicit
seeds unchanged. Three new cases fail old code and pass fix; collector/runtime/
schedule regression368passed1skipped. Actual native checkpoint1 replay of four
driver RNGs now reproduces three future request seeds/rank on fresh builders.
This cannot reconstruct historical unseeded trajectories; next controlled resume
comparison requires a fresh baseline under the fixed request semantics.

Real pinned Kling CPU cold-construction probe additionally changed Python and
TorchCPU RNG (NumPy unchanged). Preserve driver Python/NumPy/TorchCPU and already
initialized CUDA RNGs across synchronous reward import/build/preparation,
including construction failure. Five new CPU cases red->green;17 runtime tests
pass1skip. Real CPU cold construction repeats with all three RNG streams exact,
noCUDA. Reserve GPUs0-3 briefly for the explicit all-initialized-CUDA-stream
success/exception scope tests; fresh prior GPU inventory empty. No video queue.

Candidate7bf2b579 commits both RNG ownership fixes. Actual Kling CPU post-fix
construction leaves Python/NumPy/TorchCPU states exact with noCUDA; real native
checkpoint four-rank request-seed audit passes. All-four-visible-CUDA success and
exception tests2passed0.63s; final rewards/rollouts/checkpoint regression971passed,
11skipped36.33s. All probes/tests terminal and fresh GPU inventory empty; release
GPUs0-3. Evidence cosmos_reward_construction_rng_{before,after} with sources/logs
and cosmos_checkpoint_request_seed_audit. No full resume run performed yet:
fresh fixed-version baseline required because old remote unseeded RNGs were not
checkpointed. Do not claim old unseeded trajectories reproducible or quality.

### Claim: fixed-version uninterrupted versus fresh-process native resume

Reserve GPU0-3 for sequential native CLI runs at clean7bf2b579. Fresh baseline
two epochs under explicit request seeding and reward-construction RNG isolation;
then a new four-rank process resumes baseline/checkpoint-1 for epoch2 only, in a
separate output directory. Same bounded native integration workload as prior
run: global8 samples/epoch, real512p93f20-step Cosmos/Kling, one PPO/one timestep,
checkpoint every epoch. Compare uninterrupted/resumed checkpoint2 states, RNG,
metrics and real reward scores. This is resume correctness, not a full recipe
or speed/quality result. Fresh compute inventory empty before claim.

### Actual fixed-version four-card native resume passed; GPUs released

Clean7bf2b579 baseline two-epoch native CLI and fresh strict checkpoint1->epoch2
branch both complete, all rank verdicts success/torchrun exit0. New baseline
integration audit passes. Cross-arm comparison:1120 model+2240 trainer tensors,
12 Torch RNG tensors plus all Python/NumPy states, model identity/progress,
full metrics row and eight full reward-vector multiset all exactly equal.
Only representation delta is Adam betas tuple(0.9,0.999) versus list[0.9,0.999].
Preserve initial strict-container report exit2; final comparator records and
normalizes only that ordered pair, no numerical tolerance, mismatch_count0.
Evidence cosmos_native_resume_{baseline,resumed}, adjacent logs, executed sources,
baseline acceptance_audit and resumed resume_comparison reports. Historical
unseeded runs are not the control. Warm~171s versus restarted~258-259s is cold
restart overhead, not GPU speedup. All runtime/audit sessions terminal, fresh GPU
and Ray inventories empty: releaseGPU0-3. Full recipe/performance/quality and
continuous recovery remain open; no old queue restarted.

### Claim: controlled throughput four-card arm

Reserve GPU0-3 for the four-card arm of an identical-global-work pilot. New four
prompt JSONL selects actual VideoPhy train lines1-4 with seeds11000/22000/33000/
44000. Single arm has four groups/rank, four-card arm one group/rank; both eight
global samples/update, two updates, fixed sequential sampling, full512p93f20CPS
generation and the same one-PPO/one-timestep integration training workload.
CPU preflight of both authored launchers verifies identical global requests and
algorithm/actor/model/sampling/reward settings, topology-correct rank ownership
and exact partitioned advantages with the resolved algorithm config. Explicit
single-process override must clear inherited distributed.training.fsdp; caught
before GPU launch. Candidate remains clean7bf2b579. Four-card arm runs first;
no speedup conclusion until matching single-card measurement and state audit.

### Four-card throughput arm complete; GPUs released

Clean7bf2b579 torchrun exit0, two fixed-work updates, eight global samples each.
Audit exit0:1120 owned tensors,560 Adam/EMA counts1/2, all560 trainables changed,
both replay differences0,16 decode/score receipts,4 generation workers loaded
once, checkpoint-final exact and artifact seals verified. Native measured phase
totals cold260.025-261.092s/warm170.902-171.632s; excludes outer model load,
checkpoint and shutdown. Evidence cosmos_fair_throughput_four including executed
sources, acceptance_audit.json and timing_summary.json. Single-card arm prepared
and CPU-preflighted but NOT executed; no speedup/quality claim. Both runtime and
audit terminal; fresh GPU/Ray inventories empty, releaseGPUs0-3. No new queue.

### Claim: matching single-card throughput arm

Reserve GPU0 for the prepared single-process arm and keep GPUs1-3 idle to avoid
cross-job host/PCIe interference. Fresh inventory empty; candidate clean7bf2b579,
matching four-card arm. Same four fixed prompts/seeds, eight global samples and
two updates, full generation/reward with bounded one-PPO/one-timestep training.
The authored single launcher has not previously run; output root absent. Audit
both updates and compare numerical states before interpreting timing. Do not
restart the old queue or modify live candidate/scripts/dependencies.

### Matching single-card comparison passed; all GPU claims released

Single native CLI completed two updates at clean7bf2b579, exit0. Audit passes
1120 owned tensors,560 Adam/EMA counts1/2, all560 trainables changed,16 real
decode/score receipts and one worker loaded once. Single reward batching is
one request of8/update versus four requests of2 in the four-card arm; same
global8 samples. Both updates' full reward-vector multisets exact across arms,
both replay differences0. Cross-topology own-step update relative errors
1.633e-10/2.365e-10, parameter max3.638e-12/7.276e-12; initial frozen state exact,
Adam/EMA pass unchanged limits after verified integer-ID/FQN normalization.

Native measured phase totals single739.225/653.268s versus four261.092/171.632s:
cold2.83x and one warm observation3.81x. Native two-update loop including final
checkpoint:1403.335s versus449.977s,3.12x; excludes pre-loop trainer/backend
construction and post-loop shutdown. System throughput, not isolated GPU-only
acceleration or full-recipe/quality proof; fixed arm order, per-processOMP4,
different reward batch grouping and short overlapping CPU checkpoint audit are
recorded caveats. Evidence cosmos_fair_throughput_single and shared
cosmos_fair_throughput_comparison.json with executed sources. Comparator CPU
tests7passed; runtime/audit/both comparisons/summary terminal0. Fresh GPU and
Ray inventories empty; releaseGPU0 and idleGPU1-3. No additional job queued.

### Claim: released-weight Wan2.2 native dual-expert acceptance

Reserve GPUs0-1 for sequential current-candidate tiny CUDA prerequisites and a
new native two-rank dual-expert run; keep GPUs2-3 free of competing jobs. Candidate
clean7bf2b579 descends from frozen382d0825, includes verified checkpoint/RNG fixes.
Fresh GPU inventory empty. CPU preparation passed for pinned full14B+14B T2V
weights, original320p17f10-step lifecycle geometry, two samples/rank and two
updates. Nine replay steps cover both experts. New config replaces OCR entirely
with pinned Kling video reward, explicit full_cpu activation checkpoints, IEEE
math and EMA interval1; rank-local trainer/rollout/reward ownership verified.
This is a new bounded dual-expert run, not restarting the interrupted full480p
physics workload or the old SD3 queue. Evidence prefix wan22_native_dual in
/mnt/nvme/outputs/wan22_i2v_cache. Require actual nonzero moments/updates for
both experts, replay agreement, lifecycle memory and final checkpoint audit.
CPU preflight is not released-weight GPU fit, resume or quality acceptance.

Current-candidate tiny CUDA prerequisites2passed19.14s. First real launch used
an absolute snapshot path and spent~192.5s in local content identity before
trainer construction; live py-spy confirmed _hash_regular_file and /proc I/O
advanced through tens ofGB. Request graceful stop while still hashing; during
torchrun's grace period ranks entered weight loading, then terminated. No
generation/update/checkpoint completed; torchrun exit1, fresh GPU/process
inventories empty. Preserve sources/config/log/stack in
wan22_native_dual_local_hash_attempt. This was an intentional agent stop, not
an unexplained external Ray termination.

Prepared new output wan22_native_dual_kling_pinned with the original immutable
Hub repo ID/revision instead of absolute local source; HF_HOME remains NVMe and
HF_HUB_OFFLINE=1, so it uses the same cached snapshot without per-resolution
local-tree hashing. No shared runtime edit or identity bypass for local paths.
All workload settings unchanged; separate pinned preflight/config/launcher.
GPUs0-1 remain claimed across the sequential corrected launch.

Pinned-only attempt skipped identity hashing and reached trainer construction,
then exited1 because Diffusers shard discovery called Hub model_info despite
HF_HUB_OFFLINE=1. The supported model.local_files_only flag was still false.
Keep its config/log/verdicts at wan22_native_dual_kling_pinned and adjacent
files. New offline config explicitly sets model.local_files_only=true; CPU
preflight verifies both replay and rollout ModelBuild.pretrained_kwargs carry
the exact revision and local_files_only=True. No runtime edit or network
exception workaround. Fresh GPU/process inventories empty before this retry;
new output wan22_native_dual_kling_offline, same model/workload and claims.

Offline launch completed one real dual-expert update, but pre-update maxLP
0.0009684562683105469 and clip_fraction0.4722222222222222 (active0.19444444444444445)
do not establish strict rollout/replay semantics despite passing native0.01
guard. Agent requested stop of subsequent work; both controllers/monitor are
terminal, torchrun exit1 and monitor exit0; fresh GPU/Ray inventories empty.
Checkpoint1 audit passes1280 owned tensors/Adam states/FP32 masters and EMA1;
each expert640 tensors,320 nonzero first moments and320 nonzero LoRA B tensors.
All master->BF16 projections exactly match saved model. Preserve all attempts,
partial next-iteration artifacts and first_update_diagnostic.json, not success.

Investigate precision mismatch before another full run. Actor policy casts125
FP32 base parameters per expert toBF16; native none policy incorrectly rejects
frozen FP32 exceptions and can select a frozen leading dtype for LoRA validation.
Two new tests reproduce these admission failures. Scoped fix preserves floating
frozen dtypes under none, keeps actor casting and nonfloating/mixed-trainable
rejection unchanged. Reserve GPUs0-1 for expanded tiny-real CUDA/NCCL tests of
mixed BF16 trainables/FP32 frozen dual experts versus unsharded forward, not a
new released-weight run or proof that all observed drift has been fixed.

Released GPUs0-1 after native frozen-dtype validation. Candidate4c527cb1 is
committed and clean:207CPU passed13skipped;4tiny CUDA/NCCL cases passed34.17s,
covering native mixed frozen FP32/trainable BF16 forward equality, both expert
updates and unchanged frozen values on1/2ranks. Fresh GPU compute and Ray
process inventories are empty. No released-weight rerun followed this fix.
Saved the real first-update diagnostic, startup failures, replay clipping and
remaining controlled-test gates in
docs/research/wan22_dual_expert_l40s_preflight_20260912.md.
Next: same cached real trajectories, actor versus native precision and batch
shape controls, then gradient semantics; do not continue online updates merely
because the loose replay guard passed. All four GPUs are released.

Claim GPUs0-1 for Wan fixed replay-fixture capture on candidate4c527cb1.
Only one native two-sample/ten-step generation group and real Kling reward;
no training or optimizer update. CPU preflight resolved explicit trainer0,
dedicated rollout1/reward1 sharing after correcting inherited two-rank fields.
Capture initial dual-root LoRA and preserve CPU trajectory tensors for a later
same-input precision/batch-shape matrix. Existing online output has no saved
trajectory fixture. Script wan22_capture_replay_fixture.py; output
wan22_replay_fixture_real under /mnt/nvme/outputs/wan22_i2v_cache.

Release GPUs0-1 after capture diagnostics and reward pool fix. First capture
terminal1 before generation (missing harness runtime.activate); corrected
wan22_replay_fixture_active generated2videos69.035s but failed reward parking.
No rollout_batches.pt exists; do not claim fixture integrity or replay pass.
Both owned Ray sessions terminated and fresh GPU/Ray inventories are empty.
Small actual2GPU test reproduced missing CuMem capture on noncurrent reward
device, then exposed sleep also needs target-device context. Candidate96bcac9c
scopes full reward pool lifecycle to configured CUDA device, restoring caller.
GPU regression1passed4.76s;CPU rewards413passed8skipped8.56s;Ruff/diffchecks pass.
Research report records failed attempts and pending full-Kling verification.
Next capture uses new output and candidate96bcac9c, explicit activate, then
fixed cached-trajectory precision/batch-shape comparisons; no online update.

Claim GPUs0-1 for same bounded Wan capture on clean candidate96bcac9c.
Fresh GPU inventory empty. Unchanged capture script with explicit activate;
new output wan22_replay_fixture_devicefix preserves both previous failures.
Validate real Kling pool parking and save20transitions before any replay matrix.

Capture devicefix terminal0:2real videos,20finite transitions saved; initial
dual-root LoRA1280tensors and trajectory SHA256 audited. Kling sleep freed4.69GiB,
full owned Ray shutdown succeeded; fresh GPU/Ray inventories empty. Generation
69.588s, reward37.597s, collect107.186s. Release GPU1; retain GPU0 claim for native
one-rank FSDP fixed-trajectory matrix, actor and none precision sequentially,
batch2/1 in each, no optimizer. Script wan22_fixed_replay_matrix.py; outputs
wan22_fixed_replay_actor and wan22_fixed_replay_none. Candidate96bcac9c unchanged.

Release GPU0 after completed controlled matrix. Replay initially exposed
missing replay-only _pipeline_offload initialization before cached LoRA load;
candidate52a7cf44 fixes it,86Wan/MRO CPU tests passed. Both successful arms use
this clean candidate, same verified20transition fixture, no optimizer.
actor batch2:maxLP.0008442997932434082,11/20clipped;batch1:.0006373822689056396,10/20.
none batch2:LP tensor-exact0,0/20clipped;batch1:.0003195483877789229,9/20clipped.
Frozen250FP32tensors preserved under none. Original clip_ratio.0001 unchanged.
Independent saved-tensor comparison/fixture rehash passes;all sessions exit0,
fresh GPU compute inventory empty. Outputs actor_loaded/none and
wan22_fixed_replay_comparison.json; main research report updated.
Next require matched generation/replay batch shape with frozen FP32 preserved,
then actual gradient/accumulation/update single-vs-multirank comparison. Forward
equality does not prove native BF16 gradient reduction equivalent to actorFP32.
No long queue restart, full I2V/quality/resume claim or default policy flip.

Claim GPUs0-1 for four-sample real Wan fixture on clean candidate52a7cf44.
Keep generation batch2, initial policy/geometry/reward identical; capture4unique
samples in one group for matched single-rank2microbatch versus2rank1microbatch
gradient/update controls. Parameterized capture harness preserves executed
earlier sources; output wan22_four_sample_fixture. GPU inventory initially empty.

Four-sample capture terminal0,40finite transitions and distinct initial latents
audited; collection180.918s,gen138.164s,reward42.754s,realKling parking4.69GiB.
Fresh GPU/Ray inventories empty. Retain GPUs0-1 for sequential one/two-rank
FSDPnone gradient/update controls, matchedbatch2,global4samples,all9recipe replay
steps,full_cpuGC. One rank2chunks versus2ranks1chunk each; pre-updateLPmust0.
Outputs wan22_fixed_gradient_single / wan22_fixed_gradient_two; script
wan22_fixed_gradient_probe.py uses native advantages,loss,backward,clip,optimizer.

Release GPUs0-1. One/two-rank fixed gradient loops completed,36sample/step pairs
each withLPexact0; both GPU sessions terminal0 and fresh GPU/Ray inventoriesempty.
Initial single harness admission failed beforemodel because microbatch_size is
prompt-count; corrected samples_per_replay_batch=2, output single_matched.
Audit discovered direct build_optimizer omitted native FP32-master wrapper.
Executed ordinaryAdam updates remain diagnostics, notnativeoptimizerproof;
rawpreclipgradients unaffected. Preserved executedsources; adjacentprobe now
uses OnlineTrainer._ensure_optimizer but not yetGPUrerun. CPU-only native
master replay fromsavedgradients produced master_update.pt forbotharms.
Comparison terminal2 not_accepted:rawgradrel.00403357407,masterupdaterel.00789516337,
modelupdaterel.00797980957,Adamfirstrel.00403357238,secondrel.00764669540.
Fixedgates1e-4 (second2e-4) unchanged. All1280tensorstates finite;initialLoRA,
advantages,workcoverage andmasterprojections checked. Evidence/report saved.
Next isolate nativeBF16reduce versus accumulationorder; exactforward didnot
close trainingsemantics. No longtraining, qualityclaim ordefaultflip.

Claim GPU0 for one independent-contribution capture on clean52a7cf44.
Samefour-samplefixture,nine steps,chunk2,lossdivisor18,FSDPnone/full_cpuGC;
cleargradients between each of18backwards and save640active-expert tensors.
Nooptimizer. Reconstruct sequentialBF16versuspairedBF16 accumulation onCPU
against prior actualone/tworank rawgradients. Sourcewan22_gradient_contributions.py,
outputwan22_gradient_contributions. FreshGPUinventoryempty;GPUs1-3unclaimed.

ReleaseGPU0 after18independent real gradient contributions captured,allLP0,
capture254.645s(nooptimizer),terminal0. CPU rounding audit terminal0 exactly
reconstructs all1280 actualsingle gradients bysequentialBF16 additions and
all1280 actualtwo-rank gradients bypairedBF16 averaging/accumulation. Thus
observedcrossrankrel.00403357407 is fully reproducible fromBF16sumorder here.
FP32simulation orderdiffrel1.06272272368e-10,max1.81898940355e-12;notimplementedfix.
InstalledFSDPcastsreduceoutputtoorig_dtype beforegradientaccumulation;FP32master
preparationoccursafterthis,so changingreduce_dtypealoneisinsufficientguarantee.
Evidencewan22_gradient_contributions/rounding_audit.json andexecutedscripts;
researchreportupdated. Candidate52a7cf44unchanged/clean,GPUinventoryempty.
Nexttesthigher-precisiontrainable/gradientpathbothsingle/multirank,explicitly
newnumericalbaseline,thenactualnativeoptimizer,updatedrolloutsync,resume.
Noqualityregressionclaim,defaultflip,thresholdrelaxationorlongtrainingrestart.

ClaimGPUs0-1 for sequentialone/tworank realFP32-LoRA gradient/update experiment.
Clean52a7cf44,unchangedfour-samplefixture andfrozenparameterdtype/storage;
diagnosticpost-build castonly1280trainableLoRAtensors toFP32,identityoverride
explicitinreport(notpublicbuilderconfiguration). FSDPnone/batch2/full_cpuGC,
allnine steps,LPmust0,native_ensure_optimizer. Outputswan22_fp32_lora_single/two;
scriptwan22_fp32_lora_gradient_probe.py. FreshGPUinventoryempty. Compareboth
newprecisionarms,notbitwiseoldBF16baseline. NoRay/generation/reward inthisrun.

ReleaseGPUs0-1. RealFP32-LoRA single/tworank runs completedterminal0;all36sample/
step pairs LPexact0,1280FP32gradients/parameters andAdamstates checked. Native
_ensure_optimizer correctly selectsAdamW foralreadyFP32trainables, no separate
masterwrapper needed. Bothrootsupdate,640changedtensors. Saved-tensorcompare
terminal0passed:gradrel1.06272272368e-10,updaterel5.66592481312e-10,Adamfirstrel
1.10115087921e-10,secondrel4.380713043e-12;fixedgatesunchanged. EmptyGPUinventory.
Outputswan22_fp32_lora_single/two,summarywan22_fp32_lora_comparison.json;
researchreportupdated. Candidate52a7cf44unchanged/clean. Thisclosesboundedreal
newprecisiongradient/firstupdate,notoldBF16bitwiseequivalenceoronlineworkflow.
Next explicitfamily-ownedprecisionoption+identity/dtypechecks,actualupdated
rolloutdelivery andcheckpointcontinuation;no publicdefaultflip orqualityclaim.

Claim GPUs 0-1 for public FP32 Wan updated-weight rollout verification on
candidate 6ab984cf. Fresh GPU compute inventory empty. Load the saved two-rank
FP32 update through the public family option, verify all adapter tensors on the
actual Ray worker, then collect two real samples at 320x320/17f/10 steps with
Kling reward. Output: wan22_updated_fp32_rollout. No new optimizer update, full
I2V acceptance, quality claim or long queue restart in this run.

Updated rollout: initial run exited 1 at exact readback because sequential
offload leaves meta parameters. Candidate 21ae2051 adds hook-suspended actual
readback; 49 Wan tests passed. New output wan22_updated_fp32_rollout_readback
exited 0: 1280 FP32 tensors verified on worker, two clips and real Kling scores
saved. Release GPU 1 after fresh empty inventory; retain GPU 0 claim for native
one-rank FSDP updated-weight replay, all 20 transitions, matched batch 2 plus
batch 1 diagnostic. No new optimizer update or throughput claim.

Release GPU 0. Updated replay and independent tensor audit both exited 0;
fresh GPU inventory empty. Actual updated state has 320 changed tensors per
expert; worker readback verifies all 1280 FP32 tensors. Public-build one-rank
FSDP replay preserves frozen mixed dtypes: batch 2 all 20 log-probs exactly
match rollout; batch 1 max error 0.00039689987897872925, 8/20 ratio-clipped at
1e-4, so batch invariance remains unaccepted. Candidate 21ae2051; evidence in
wan22_updated_fp32_rollout_readback and wan22_updated_fp32_replay, with
acceptance_audit.json. Next checkpoint continuation, then broader native
recipe/performance gates. No quality claim or long training queue restarted.

Claim GPUs 0-1 for native public FP32 Wan checkpoint baseline, candidate
21ae2051, fresh compute inventory empty. Output wan22_fp32_native_baseline:
2 updates x 4 global samples, matched generation/replay batch 2, both experts,
9 replay steps, FSDP none/full_cpu GC, EMA interval 1 and checkpoint each update.
Parity gate tightened to 1e-8 from historical diagnostic 0.01. This prepares
controlled checkpoint-1 continuation, not a long quality or scaling run.

Release GPUs 0-1. Native public FP32 baseline completed both updates, exited 0;
fresh compute inventory empty. Each update has exact-zero replay difference
and clip fraction. checkpoint-1/-2/-final saved; independent checkpoint audit
exited 0, checking 1280 FP32 parameters/Adam/EMA entries, progress and both rank
RNG states. All 1280 parameters changed on second update. Candidate 21ae2051
unchanged. Evidence wan22_fp32_native_baseline/checkpoint_audit.json plus native
metrics/run evidence. Checkpoint/identity regression: 147 passed. Next fresh
process resume from checkpoint-1 to epoch 2 and compare uninterrupted state,
RNG, reward vectors and metrics. This baseline alone is not resume acceptance.

Claim GPUs 0-1 for fresh native FP32 Wan checkpoint-1 continuation to epoch 2,
candidate 21ae2051 unchanged, fresh GPU inventory empty. New output
wan22_fp32_native_resumed; same numerical recipe as baseline, only resume and
artifact output paths changed. Compare complete state and actual reward vectors
against uninterrupted checkpoint-2 with no numerical tolerance. One update only.

Release GPUs 0-1. Fresh two-rank native Wan FP32 resume exited 0; empty fresh
compute inventory. Comparator exited 0 with zero mismatches: 1280 model and
5120 trainer tensors, six Torch RNG tensors plus Python/NumPy/named RNG,
progress/identity, full epoch-1 metric row and all four complete reward vectors
exactly match uninterrupted checkpoint-2. Final checkpoint matches checkpoint-2
in each arm. Adam betas list/tuple representation is explicitly recorded and
compared as an ordered pair, with no numerical tolerance. Evidence:
wan22_fp32_native_resumed/resume_comparison.json. Candidate 21ae2051 unchanged.
Bounded T2V native resume accepted; full I2V, four-rank scaling and quality open.

Claim GPUs 0-3 for equal-work native Wan four-rank pilot, candidate 21ae2051.
Fresh compute inventory empty. Prepared four-prompt/seed plan equals single
arm, 2 updates x 8 global samples, batch 2, nine replay steps, FP32 LoRA/FSDP
none/full_cpu GC. Output wan22_fair_throughput_four, sampled memory and outer
process receipt alongside output. Four-rank host capacity unproven; no speedup
claim until both arms and numerical/workload audits complete.

Release GPUs 0-3. Four-rank pilot exited 1: eight initial samples generated,
but simultaneous Kling cold loading triggered Ray host-memory protection at
~95% usage. Agent sent SIGTERM to identified parent after low-memory sample;
elastic forced remaining children down. Fresh compute inventory empty. Sampled
minimum host available 5,881,606,144 bytes; no optimizer checkpoint or accepted
timing. Failure output/log/memory/process receipts retained without overwrite.
Candidate 9145b2af now memory-maps ZIP full Kling checkpoints, legacy compatible.
34 tests passed/1 skipped and all 1123 real checkpoint tensors equal ordinary
load. Four-rank capacity improvement not yet verified. Next distinct-output
retry with unchanged workload/thresholds, then matching single-card arm.

Claim GPUs 0-3 for four-rank equal-work retry on candidate 9145b2af (Kling mmap
loading). Fresh compute inventory empty, worktree clean. Output four_mmap is
distinct from failed four arm; only artifact destinations change in YAML.
Same two updates x eight global samples, numerical recipe and Ray protection.
Monitor host/GPU memory throughout; mmap value equivalence is already verified,
but capacity/performance success remains unproven.

Release GPUs 0-3: user paused experiments to rebase the review changes onto
main 6b723075e994255539b7604e249cf7efdb1f5547. The four_mmap supervisor
terminated with exit 1 after 198.739 seconds following the requested SIGTERM;
fresh compute inventory is empty. No completed update or accepted timing is
claimed for this interrupted retry. Existing output and monitoring receipts
are retained. Do not resume the experiment queue until requested by the user.

User explicitly resumed hardware work after the review-branch push. Claim GPUs
0-3 for the interrupted equal-work Wan mmap trial on clean candidate 9145b2af.
Fresh inventory has no compute processes; host available memory is 367 GiB.
New output four_mmap_resumed restarts from the original initialization (not a
checkpoint), with only artifact paths changed: 2 updates x 8 global samples.
Keep both previous failed/interrupted outputs. Acceptance still requires
completed updates, checkpoint audit and the matched single-card comparison.

Release GPUs 0-3. four_mmap_resumed exited 1 after 295.623 seconds. All eight
initial samples were generated and scored, but the trainer's host budget gate
rejected the collected batch at 97.1% used (95% limit, about 11 GiB available).
No optimizer update or accepted timing. Fresh compute inventory is empty;
retain complete logs/monitoring. mmap loading alone does not establish capacity.
Claim GPU 0 for a bounded real Kling load/score/park host-memory diagnostic,
including explicit allocator trim and identical-score validation after wake.

Release GPU 0. Production-contract Kling host and pinned-host probes exited 0,
with exact repeat scores across park/wake. libc trim recovered only ~110 MiB;
pinned cache clearing during live parking recovered only 15 bytes. After model
shutdown, pinned cache clearing recovered ~5.83 GiB. Two initial isolated
attempts omitted the registry residual allowance and failed zero-limit checks;
the corrected probes use the declared production allowance. Candidate unchanged.
Fresh compute inventory empty. Details and receipts are in the Wan report.
Next reduce cross-phase resident copies/owner lifetime, not memory thresholds
or workload. Four-rank update capacity and fair speedup remain unaccepted.

Post-integration continuation, 2026-09-13: review candidate 4e2c1163 passed
791 affected CPU tests and 12 explicit tiny GPU tests. Previous goal turn made
progress; no full experiment completion is claimed. GPU inventory was empty
before and after the next bounded real Kling owner-lifetime probe on GPU 0.
Review branch and its committed artifacts remain unchanged by this diagnostic.

Evidence: /mnt/nvme/outputs/wan22_i2v_cache/kling_reload_lifetime_rebased/result.json
and sibling kling_reload_lifetime_probe.py. Locked review environment uses
Torch 2.11.0+cu130; local pinned Kling weights, fixed original MP4 SHA256,
two independent runtime owners with two scores each. All four complete score
maps are exactly equal (overall -0.6317654154646988). Initial load 40.167s,
reload 37.685s; shutdown 0.274s/0.291s. GPU allocated memory after shutdown is
9,568,256 bytes, reserved 41,943,040 bytes. RSS after first shutdown is
2,008,236,032 bytes, after second 4,732,772,352 bytes. Thus ordinary runtime
shutdown/reload is numerically repeatable here but does not prove bounded host
residency or four-rank capacity. No production lifecycle option was changed.

This unpooled diagnostic is not a direct comparison to the older pooled,
re-encoded-media parking probe. Even the older original-MP4 direct probe used
Torch 2.12 and returned a different overall score (-0.6489913727634578).
Do not attribute this difference to a specific dependency without isolation;
do not mix these environments as a fair throughput/learning baseline. Future
single/four-rank arms must share the rebased commit, locked environment, exact
artifact and numerical recipe. Next distinguish live ownership from allocator
retention after repeated shutdown before integrating any reload policy.

Follow-up retention isolation on GPU 0 completed on unchanged review candidate
4e2c1163 and locked Torch 2.11. Evidence: kling_reload_retention_rebased/result.json
and kling_reload_retention_rebased.log under the same NVMe experiment root.
Each cycle tracked 1124 weak references to the reward wrapper, torch model and
parameters; none survived shutdown/collection. Second shutdown RSS was
4,726,161,408 bytes; gc did not change it, but malloc_trim reduced it to
1,921,343,488 bytes. This identifies ~2.61 GiB of reclaimable libc retention,
not surviving tracked model owners. The first-cycle trimmed RSS was
1,900,134,400 bytes. This Torch version has no empty_host_cache API, so the
host-cache-labelled snapshot is explicitly a no-op, not a successful clear.
All four scores exactly match the previous rebased fixed-MP4 result.

Then claimed GPUs 0-3 for a bounded four-process reward-only reproduction,
one physical GPU per process, two independent loads and two scores per load.
Supervisor kling_reload_four_probe.py completed exit 0 in 91.028s; result and
per-rank logs: kling_reload_retention_four_rebased/. All 16 complete score maps
are exactly equal. Both cycles on all ranks have zero surviving tracked model
owners. Final per-rank RSS after shutdown/trim: 1,906,061,312; 1,913,630,720;
1,919,094,784; 1,911,480,320 bytes. Reload times 37.76-37.97s, initial loads
40.25-40.50s. Minimum sampled host available memory 369,606,246,400 bytes.
Fresh GPU inventory empty after supervisor and all children exited.

This establishes an explicit owner-release/trim candidate with repeatable
reward scores on four GPUs, not a pooled-parking A/B, long-run leak bound, or
combined Wan trainer capacity/throughput result. Production defaults and
memory protection remain unchanged. Next implement an opt-in reward teardown
handoff with failure cleanup and RNG invariants, then run both equal-work
single/four-rank Wan arms on the same rebased runtime. Account for measured
reload overhead; never compare new timings against old-environment scores.

Implemented opt-in native reward reload parking on feat/reward-reload-handoff,
candidate a6fc7356 (worktree /home/ubuntu/VRL-review-all). The review/all-mgpu-main-
6b723075 ref stays at 4e2c1163. Default CuMem behavior unchanged. Explicit reload
mode requires sleep_offload and glibc, destroys the scorer model and trims
released CPU heap at handoff; next activation reloads under preserved RNG.
115 focused inference/disk-reward/online-lifecycle tests passed. Native four-GPU
Kling probe exited 0: all 16 score maps equal, reload 37.78-37.91s, final parked
RSS 1.909-1.919 GB/rank. No script-side trim. Evidence:
kling_native_reload_four_rebased/result.json and per-rank logs. GPUs released.
Repository report: docs/research/reward_reload_handoff_20260913.md on new branch.

Prepared fresh wan22_rebased_reload_single/four.yaml on the same locked runtime;
CPU preflight passed global request equality and exact partitioned advantages,
2 updates x 8 global samples, preserved prompt manifest and thresholds. The
preparation script adapts old batch fields and manifest-loading API, not work.
Next run full four-rank Wan capacity on this native candidate before the matched
single-card arm; reward-only success does not establish combined capacity.

Claim GPUs 0-3 for full Wan rebased native-reload capacity pilot, clean candidate
76714903 (code a6fc7356), locked review venv, empty starting compute inventory.
Supervisor wan22_rebased_reload_launch.py four, torchrun PID 846665, output
wan22_rebased_reload_four, exclusive log/memory/process receipts. Two updates
x eight global samples; no changed numerical gate or memory threshold. Both
training experts load from the pinned local Wan2.2 checkpoint. Monitor through
terminal status; do not launch another arm while this process remains live.

Release GPUs 0-3. Native reload four-rank Wan pilot exited 1 after 541.036s.
First optimizer update and checkpoint-1 completed: exact-zero replay difference
and pre-update clip fraction, gradient norm 0.030143787340297008. Independent
checkpoint_1_audit.json passed all 1280 FP32 parameters/Adam/EMA and four-rank RNG,
progress and eight global samples. No second update: next rollout wake surfaced
Ray's earlier worker kills for >95% host usage. Do not claim accepted timing or
speedup from the 335.68-335.76s first-update phase totals.

Memory: 523 samples, minimum available 7,638,007,808B, max GPU 11,983,126,528B.
Raylet threshold crossing starts 16:45:31-32, optimizer runs 16:45:31.384-32.575,
worker kills about 16:45:39, driver failure appears on next wake 16:45:52.
At kill, rollout workers use ~55.6-56.4GiB each, trainers ~29GiB each, object
store occupancy zero. Remaining pressure is at optimizer/update boundary;
precise live allocations versus allocator retention not yet established.
Fresh compute inventory empty, all handles terminal, failure artifacts retained.
Next isolate optimizer/export/checkpoint CPU lifetimes without lowering work or
raising thresholds. Single-card arm not yet run. Full report on follow-up branch:
docs/research/reward_reload_handoff_20260913.md.

Claim GPUs 0-3 for a diagnostic boundary-trim reproduction on unchanged code
ffbebe7e. Valid first checkpoint contains 838,860,800B model, 1,677,726,720B
optimizer and 838,860,800B EMA tensors (logical full-state sizes, not rank RSS).
New output wan22_rebased_boundary_trim_four, torchrun PID 853437, supervisor
wan22_boundary_trim_launch.py. Same two updates/eight global samples/config
except artifact paths; entry wrapper wan22_boundary_trim_entry.py records
RSS/PSS/USS before collection, after collection, after glibc trim, and after
the original optimizer method. No training math/threshold change. Compare
checkpoint-1 against the previous accepted first checkpoint if it completes.
No concurrent GPU arm; monitor this exact live handle to terminal status.

### GPU claim: prefetch-benefit profiling (vrl-74, 2026-09-13 22:30 PDT)

The Codex Wan I2V full-physics run at
`/mnt/nvme/outputs/wan_i2v_full_physics_batch_local_scheduler` was torn down
at 22:16 PDT (torchrun SIGTERM/SIGKILL, no optimizer update, metrics.csv
empty); GPU inventory is empty. vrl-74 now owns GPUs 0-3 for a ~2.5 h
sequential queue under `outputs/sd3_5_ocr_prefetch_profile/`: five 3-epoch arms
of the SD3.5 3x1 preset (strict / continuous, in-process vs HTTP OCR service,
eager vs replay-compiled) with py-spy on the driver and dmon on GPU 0. Do not
launch GPU work until `queue.log` says "queue done".

### Prefetch benefit confirmed once OCR leaves the driver (vrl-74, 2026-09-13 23:40 PDT)

`outputs/sd3_5_ocr_prefetch_profile/`, SD3.5 3x1 preset, 3 epochs per arm,
epochs 1-2 averaged, py-spy on the driver main thread, dmon on GPU 0:

| arm | epoch wall | evaluate | backward | GPU0 SM% in evaluate | PaddleOCR share of driver samples |
|---|---|---|---|---|---|
| strict, in-process OCR | 509 s | 253 s | 155 s | 44% | 29% (serial, in collect) |
| continuous, in-process OCR | 469 s | 282 s | 187 s | 45% | 28% (concurrent with training) |
| continuous, OCR via HTTP service (CPU process) | **398 s** | 248 s | 150 s | 52% | 0% |

- The replay is launch-bound (GPU0 idle >50% of evaluate; 12-15% of driver
  samples in nn.Linear.forward / diffusers norms / PEFT layers).
- In-process PaddleOCR competes with kernel launch when prefetch overlaps it:
  +61 s on evaluate+backward, halving the prefetch gain (509 -> 469 s).
- Moving OCR to `vrl-reward-service` (`+reward=ocr_http`, commit 59857f2b)
  restores evaluate/backward to the strict numbers and delivers the full
  overlap: 509 -> 398 s per epoch, **1.28x**, above the 25% ceiling estimated
  from the serial collect time because the driver also stops paying the
  in-process scoring cost. Rewards stay in the same range (0.35/0.46/0.29 vs
  0.36/0.51/0.22 on the same prompts).
- Rule for continuous scheduling on launch-bound recipes: keep every reward out
  of the driver process (HTTP service or a dedicated GPU), or the overlap is
  paid back in slower training phases. Compile arms (replay-scoped
  torch.compile) still running.

### Prefetch / reward placement / compile: final five-arm table (vrl-74, 2026-09-14 00:30 PDT)

`outputs/sd3_5_ocr_prefetch_profile/`, SD3.5 3x1 preset (batch 1), epochs 1-2
averaged; `*_diag` arms ran with `trainer.replay_parity.max_abs_logprob_diff=0.05`
because both compile scopes trip the 0.01 gate (replay-only 0.025, both roles
0.014).

| arm | epoch wall | evaluate | backward | GPU0 SM% (evaluate) | pre_update_clip_fraction | ratio_abs_dev_max |
|---|---|---|---|---|---|---|
| strict, in-process OCR (eager) | 509 s | 253 | 155 | 44% | 0.00 | 0 |
| continuous, in-process OCR | 469 s | 282 | 187 | 45% | 0.16-0.18 | 0.02-0.03 |
| continuous, OCR HTTP service | 398 s (1.28x) | 248 | 150 | 52% | 0.17 | 0.02-0.03 |
| strict, compile scope=all (diag) | 270 s (1.89x) | 138 | 61 | 63% | 0.54 | 0.010-0.014 |
| continuous, HTTP OCR, compile (diag) | **206 s (2.48x)** | 143 | 62 | 67% | 0.54-0.60 | 0.014-0.030 |

Conclusions:
1. The SD3.5 replay is launch-bound (GPU0 idle >50% of evaluate in eager).
2. Prefetch pays its full ~25% only when no reward runs inside the driver
   process; in-process CPU OCR competing with kernel launch cost 61 s/epoch.
   `+reward=ocr_http` (commit 59857f2b) fixes that: 509 -> 398 s.
3. torch.compile removes most of the launch overhead (1.89x strict, 2.48x
   stacked) but is NOT a usable training configuration as-is: with
   clip_ratio=1e-4 the compiled rollout/replay drift (max 0.014-0.030) clips
   54-60% of samples before any update. Same root cause as the batch-16
   parity failure: bf16 kernel paths differ between rollout and replay.
   Enabling it in production requires lifting clip_ratio and the parity gate
   above the measured drift (the Wan precedent, be6cbbe2), which is one
   decision for compile and batch 16 together.
4. The in-process-vs-HTTP rule is conditional: it matters only when reward is
   CPU-bound, scheduling is continuous, and training is launch-bound. Strict
   video recipes with GPU rewards (Wan + HPSv3) are unaffected.

## Coordination (session vrl-9941, 2026-09-14 12:30 PDT): Wan 2.2 T2V-A14B GRPO learning experiment

- Claim: GPUs 0-1 for the two-rank native FP32-LoRA Wan 2.2 T2V trainer
  (the accepted `wan22_fp32_native_baseline` configuration), GPU 2 for the
  fixed-prompt baseline/checkpoint evaluation and the Kling reward probes,
  starting now. GPU 3 is left free. Host RAM budget is the constraint: the
  two-rank trainer alone reached ~228 GiB; do not start another 14B pipeline.
- Code: isolated worktree `/home/ubuntu/VRL-wan22`, branch `exp/wan22-t2v-grpo`
  from `feat/cosmos-cp-runtime@9145b2af`. No edits to this checkout's Python.
- Outputs: `/mnt/nvme/outputs/wan22_t2v_grpo/` (NVMe, not root).
- Not touched: the paused Wan 2.1 heavy queue, the SD3.5 queue, the Codex I2V
  work. Message session vrl-9941 (socket 9941.sock) before claiming GPUs 0-2.

### GPU claim: GPU 3 only, reward-service park/wake acceptance (vrl-74, 2026-09-14)

Single-rank phase-cycled Wan 2.1 1.3B + HPSv3 (`online_grpo_hpsv3_fsdp_4x_l40s`
with `gpus_per_node=1`, `reward.inference.hpsv3.kind=service`), 2 updates on
GPU 3 while vrl-9941 holds GPUs 0-2. Output
`outputs/wan_hpsv3_flash_grpo/service_park_smoke_1gpu`. Validates P2 of
`planned/SPRINT_reward_service_isolation.md`: the HPSv3 model lives in a child
process that time-shares the card through POST /park and /wake.

Result (2026-09-14): status success, 2 updates, replay parity 0.00197 / 0.00204,
clip 0, no Xid, HPSv3 child parked/woke every phase. `reward_mean` was -4.5 /
-5.7 versus +3.0 / +3.5 in `fsdp_smoke_main`; scoring the smoke's own mp4s with
the in-process HPSv3 and the managed service gave identical numbers on every
sample (max |delta| 0.0), and both runs' per-sample scores span -10 .. +11, so
the level difference is prompt-sample variance of a 6-prompt single-rank
collection, not the transport. GPU 3 released 2026-09-14 21:40 PDT; vrl-74 holds
all GPU work until vrl-9941's Wan 2.2 four-rank stage 2 (GPUs 0-3, ~6 h) exits.

### GPU claim: all four, miles parity program gates (vrl-74, 2026-09-14 22:35 PDT)

vrl-9941's Wan 2.2 stage-2 trainer was stopped by the user at 22:33; all GPUs
free. vrl-74 runs, concurrently: GPUs 0-1 Wan 1.3B + HPSv3 cp=2 acceptance
(`experiment/wan_2_1/online_grpo_hpsv3_fsdp_2x_cp2`, 2 updates), GPU 2 the
single-rank baseline of the same recipe and seed (`cp2_baseline_1gpu`), GPU 3
the sglang-diffusion SD3.5 rollout server + trajectory probe (WS-B spike).
Next in the queue on whichever card frees first: WS-A py-spy gate (1 GPU),
then the SD3.5 recompute arm (trainer 0, rollout 1-3). Other sessions: claim
here before launching.
