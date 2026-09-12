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
