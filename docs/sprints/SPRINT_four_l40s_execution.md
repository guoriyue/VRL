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
   (`parked/SPRINT_wan_2_1_i2v_proof_run.md`).
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
  2 ranks): float32 rel_l2 2.5e-7, bfloat16 rel_l2 3.4e-3. Verdict: the
  the small FP32 forward difference supports precision amplification as the
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
