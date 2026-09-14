# Opt-in reward reload handoff

Candidate `a6fc7356`, branch `feat/reward-reload-handoff`, follows the unchanged
review branch `review/all-mgpu-main-6b723075` at `4e2c1163`.

## Contract

Set `reward.kwargs.kling_video_reward.worker_config.memory_parking_mode: reload`
on a topology that already requires reward parking. Resource resolution still
injects `sleep_offload: true`; standalone scorer construction must provide it.
Default mode remains `cumem`. Reload mode is explicit, requires glibc
`malloc_trim`, and is rejected when parking is not enabled.

At `park_memory`, reload mode drops the model, performs existing strict CUDA
cleanup, then trims released host heap pages. A later activation reloads and
prepares the model inside the existing RNG-preserving build boundary. No CuMem
backup is retained. Failure propagates; cleanup can be retried. This trades
checkpoint load latency for lower cross-phase host residency. Physical release
proof remains owned by the existing caller/lifecycle; no budget is relaxed.

## Evidence

- Reward inference tests: 66 passed, 4 GPU tests deselected.
- Disk reward and online lifecycle tests: 49 passed.
- Scoped Ruff passed.
- Native four-device probe: all 16 complete score maps exactly equal after two
  loads and two scores per load on each device, using one scorer per worker.
  No manual trim is called by the probe. Four workers completed in 91.028s.
- Reload latency: 37.78-37.91s per worker. Second parked RSS:
  1,913,270,272; 1,918,697,472; 1,915,133,952; 1,909,043,200 bytes.
- Lowest sampled host available memory: 368,779,689,984 bytes. All workers
  exited successfully and the final GPU compute inventory was empty.

Raw evidence under `/mnt/nvme/outputs/wan22_i2v_cache/`:
`kling_native_reload_four_rebased/result.json`, per-rank logs and result files,
`kling_native_reload_probe.py`, and `kling_native_reload_four_probe.py`.
Weights are cached pinned Kling VideoReward weights; source MP4 SHA256 and
runtime configuration are embedded in every worker report. Torch 2.11.0+cu130
comes from the lock-synced review venv, not the older experiment environment.

## Next Combined Gate

`wan22_rebased_reload_prepare.py` generated new single/four-rank YAMLs and
`wan22_rebased_reload_preflight.json` without rewriting the prompt manifest.
Both arms retain two updates, eight global samples per update, matched prompt
seeds, model/optimizer/precision, and two samples per generation/replay batch.
Global request plans and partitioned group advantages match exactly. Online
batch fields now use main's `prompts_per_collection` and
`training_microbatch_size`.

These results prove the reward-only handoff, not combined Wan capacity,
throughput, quality, or long-run memory stability. Run the complete four-rank
pilot next, then the matching single-card arm; retain existing failed runs and
the 95% host-memory protection. Do not mix old-environment timings or scores
with the rebased comparison.

## Combined Four-Rank Attempt

Clean candidate `76714903` ran `wan22_rebased_reload_four` on all four GPUs,
using the locked venv and unchanged workload/thresholds. Supervisor exited 1
after 541.036s. All initial eight samples were generated and scored; the first
optimizer update completed and `checkpoint-1` was saved. The next rollout wake
failed because Ray had killed the generation workers for host-memory pressure.
No second update or accepted throughput comparison exists.

First-update full-precision metrics: pre-update maximum log-probability difference
0, pre-update clip fraction 0, gradient norm 0.030143787340297008. Per-rank
phase totals are 335.68-335.76s, which must not be reported as a successful
two-update benchmark or speedup. Independent `checkpoint_1_audit.json` confirms
1280 FP32 adapter tensors, complete Adam and EMA state, four-rank RNG, progress,
and eight global collected samples. It does not compare against initialization
or certify the failed subsequent rollout synchronization.

523 memory samples reached 7,638,007,808 bytes available; sampled maximum GPU
usage was 11,983,126,528 bytes. Raylet logs first cross the 95% threshold around
16:45:31-32 and kill workers around 16:45:39. Recorded backward ends at
16:45:31.378 and optimizer runs 16:45:31.384-32.575; rank phase summaries finish
around 16:45:35. The driver receives the stored failure on second rollout wake
around 16:45:52. Thus pressure begins at the optimizer/update boundary, not
necessarily at the later wake where the error becomes visible. Specific tensor
or allocator ownership at that peak is not yet proven.

At Ray's kill decision, generation workers each used approximately 55.6-56.4 GiB
and trainer processes approximately 29 GiB. Object-store occupancy was zero.
All processes exited and the fresh GPU compute inventory was empty. Preserve
the failed output, adjacent log/memory/process receipts, and the valid first
checkpoint. Next isolate optimizer/export/checkpoint host lifetimes at this
boundary, retaining samples, replay gates, and memory protection. The matching
single-card arm remains unrun.

## Optimizer-Boundary Trim Diagnostic

Candidate `ffbebe7e` ran `wan22_rebased_boundary_trim_four` with the same
two-update workload and unchanged 95% host-memory protection. An external entry
wrapper measured each trainer immediately before `_clip_and_step`, after GC,
after glibc `malloc_trim(0)`, and after the optimizer. It did not clear PyTorch's
pinned host allocator cache or change training math. The supervisor exited 1
after 532.850s; checkpoint-1 exists, but the second update failed.

This attempt corrects the location inferred from the previous attempt: Raylet
first reported threshold crossing at 16:58:27.701, during the fifth replay
backward (epoch timestamp 1789343899.643-1789343908.254). Worker killing began
around 16:58:35. The optimizer phase only started at 16:59:34.224. Therefore
optimizer-boundary cleanup is too late for this run. The much higher available
host memory measured at that boundary follows worker termination and must not
be interpreted as successful memory reclamation by the wrapper.

Rank 0 GC left RSS unchanged at 31,968,821,248 bytes; glibc trimming lowered it
to 31,800,492,032 bytes, only 168,329,216 bytes reclaimed. This does not identify
the earlier live-tensor or pinned-cache ownership responsible for pressure.

The completed CPU comparison `first_state_comparison.json` establishes exact
first-checkpoint equality against `wan22_rebased_reload_four`: 6,412 Torch
tensors, four NumPy arrays, and 8,962 scalar leaves, including model, Adam, EMA,
RNG, identity and progress. It does not establish second-update synchronization
or throughput. All experiment processes exited and GPU compute inventory was
empty. No numerical-work reduction or threshold increase was applied.

Next inspect host allocation ownership during replay, before the first threshold
crossing, including the distinction between active and cached pinned memory.
Do not promote optimizer-boundary trimming as a capacity fix or rerun this same
failed intervention. Keep the matching single-card performance arm pending a
complete four-card run.

## Pinned Allocator Observation Gate

The locked Torch 2.11.0+cu130 environment exposes
`torch.cuda.memory.host_memory_stats()` and the private
`torch._C._host_emptyCache()` used by its own CUDA graph implementation.
`vrl/trainers/activation_checkpointing.py:cpu_checkpoint_func` uses
`save_on_cpu(pin_memory=True)`, making this allocator relevant to replay.

The bounded real-CUDA `pinned_host_allocator_probe.py` exited 0 and preserved its
raw report in `pinned_host_allocator_probe.json`. After an asynchronous transfer
from a 256 MiB pinned buffer, dropping that buffer and synchronizing, emptying
the host cache reduced allocator-owned bytes from 269,484,040 to 1,048,576.
A separately retained 1 MiB pinned tensor still contained the expected values.
After releasing that tensor and emptying again, allocator-owned bytes were zero.
This proves the local API can reclaim this synthetic unused allocation while
preserving that live tensor, not that it fixes Wan or is a production contract.

Crucially, the raw active counters were inconsistent: after emptying, reported
active bytes exceeded allocator-owned bytes; the final active-byte count was
269,484,041 with a cumulative freed count of -1 despite zero owned allocations.
The probe's `passed` status concerns release and retained tensor contents only,
not validity of every statistic. Do not subtract active bytes from owned bytes
to diagnose a leak in this environment. Future replay traces must retain raw
counters and pair them with process RSS/PSS and measured release deltas. Do not
patch the installed Torch or assume its accounting anomaly explains Wan's OOM.

## Per-Backward Release: Two Checkpoints, Failed Runtime Health

Candidate `9125fa42` ran `wan22_rebased_replay_trim_four` using an external
diagnostic entry wrapper. After each backward it synchronized CUDA, collected
garbage, emptied the pinned cache, then trimmed the glibc heap, with separate
RSS/PSS and raw allocator snapshots. Samples, replay steps, precision, thresholds,
and training math were unchanged. Each rank produced 72 snapshots across 18
backward calls. This is not a production implementation or timing baseline.

The first replay crossed the previous fifth-backward failure and the second
rollout successfully generated its samples. Rank 0 at backward sequence 3
dropped from approximately 29.45 GiB RSS to 23.82 GiB after pinned-cache clearing;
subsequent glibc trimming did not materially change it. This establishes a large
reclaimable allocation at that boundary, despite unreliable active counters.

However, Ray killed the generation workers during second-round reward loading.
One Raylet kill decision is timestamped 17:15:24.275, with host usage 95.3562%,
four generation workers around 55.71-57.15 GiB, and trainer processes around
28.38-29.71 GiB. The driver logged the kills around 17:15:41. Since generation
had already completed, training could still finish its second update. Cleanup
then logged `generation policy release wait failed; forcing actor cleanup`.
The supervisor exited **0** after 909.560s, which is explicitly **not** runtime
acceptance. `run_acceptance.json` records this failed health gate separately.

Both checkpoints passed `checkpoint_audit.json`: each contains 1280 FP32 adapter,
Adam, and EMA entries, correct progress and four-rank RNG. Every adapter tensor
changed between steps. Both updates have eight global samples, zero maximum
pre-update replay error, zero pre-update clipping and positive gradient norms.
`first_state_comparison.json` verifies exact first-checkpoint equality against
the earlier untrimmed run (6412 tensors, four arrays, 8962 scalar leaves).
These checks do not certify subsequent rollout-worker availability.

895 memory samples reached 16,869,380,096 bytes available; maximum sampled GPU
usage was 11,983,192,064 bytes. All processes subsequently exited and GPUs were
empty. No fair speedup is reported. The next intervention must address the
second reward-loading handoff's host footprint, accounting for post-update
optimizer/checkpoint allocations, while retaining the replay release evidence.
Do not rerun this same incomplete intervention or accept checkpoint-only/exit-code
checks as an end-to-end gate.

## Preload Heap Reclamation Candidate

The explicit reload mode now collects garbage and trims the host heap before
building a missing reward model, not only after destroying one. This shared
process may have performed training, weight export and checkpoint saving since
its previous reward shutdown. Live models are not trimmed on repeated activate
calls, default CuMem behavior is unchanged, and no private pinned allocator API
is added to production code. This candidate does not yet prove a Wan capacity fix.

Preload cleanup failures propagate before the factory runs; retry and driver RNG
preservation are covered. The locked reward inference suite passed 67 tests with
four GPU tests deselected; scoped Ruff check/format and diff checks passed.

`kling_preload_trim_four_rebased/result.json` records a real four-device,
two-cycle reward-only run: 93.031s outer elapsed, all 16 complete score maps
exact across workers/cycles and also exactly equal to the prior native reload
run. Final parked RSS ranged from 1,896,611,840 to 1,906,302,976 bytes. Minimum
sampled host availability was 367,623,798,784 bytes. All processes exited 0 and
the GPU compute inventory was empty. This unloaded-host test establishes score
and lifecycle regression coverage, not memory savings at Wan's loaded handoff.

Next combine this candidate with the external per-backward pinned release
diagnostic, recording immediately before and after reward preload reclamation.
Retain the exact numerical workload and require no Ray OOM or cleanup warnings
in addition to both checkpoint audits. Do not treat these reward-only timings
as a single/four-card training comparison.

## Combined Preload Candidate: First Replay Forward Still Exceeds RAM

Candidate `7b083103` ran `wan22_rebased_handoff_trim_four`, preserving the full
two-update/eight-global-sample workload and external per-backward pinned release.
The entry wrapper additionally measured the native reward heap-trim callback.
Rank 0's first preload trim reduced RSS from approximately 24.199 to 22.850 GiB;
second preload trim reduced it from 29.481 to 24.368 GiB. Thus preload reclamation
is real, but is not sufficient for this configuration.

Second-round reward scoring finished around 17:37:40; its shutdown trim completed
at epoch 1789346263.509 (17:37:43.509). Raylet first reported usage above threshold
at 17:37:51.289, during the subsequent first replay forward. The wrapper's first
second-update `before_backward` snapshot is 17:37:54.768, RSS 29.934 GiB. Raylet
decided to kill a generation worker at 17:37:58.939. The driver only received
the aggregated OOM messages around 17:38:24. Therefore absence of an immediate
driver OOM message did not establish healthy worker lifetime. Backward-boundary
release is too late for the preceding forward peak in this attempt.

Both updates nevertheless saved valid checkpoints. `checkpoint_audit.json`
passed for both steps (1280 model/Adam/EMA entries, four-rank RNG, eight samples,
zero replay error and clipping; all 1280 adapters changed at step 2).
`first_state_comparison.json` exactly matches the original first checkpoint.
The supervisor exited 0 after 922.955s, with policy-release cleanup warnings.
`run_acceptance.json` explicitly records failed runtime health and no accepted
performance result. 908 samples reached 15,775,535,104 bytes host available;
maximum sampled GPU usage was 11,983,126,528 bytes. All processes exited, GPU
compute inventory was empty, and the comparison/audit CPU commands exited 0.

Do not repeat this configuration as a presumed fix. For this 320x320/17-frame
comparison, next test existing `full` checkpointing instead of `full_cpu` to
move saved checkpoint inputs into the available GPU memory, keeping geometry,
samples, replay and precision unchanged. Verify real forward/backward capacity
and numerical results before adopting it in both comparison arms. The historical
full_cpu requirement came from the much larger 480x832/81-frame three-rank gate;
passing a smaller-video GPU-resident checkpoint test cannot close that separate
full-geometry requirement. No threshold increase or hidden workload reduction
is authorized by this diagnostic.

## Real Expert GPU Checkpoint Placement Gate

`wan22_checkpoint_placement_probe.py` completed with four ranks, exit 0, against
the pinned Wan 2.2 transformer and transformer_2 weights, tested sequentially.
Each used rank-32 FP32 LoRA, BF16 frozen weights/autocast, four-rank FSDP with
CPU offload and precision policy none. Synthetic conditioning used the current
comparison's CFG tensor geometry: latent [4,16,5,40,40], text [4,256,4096].
The squared-output loss is a capacity/numerical diagnostic, not GRPO replay.

All four `wan22_checkpoint_placement_four/rank-*.json` reports verify exact
outputs and all 640 local gradient tensors per expert between `full` and
`full_cpu`. Each expert/rank had 320 nonzero gradient tensors; all gradients were
finite. GPU peak allocated bytes were 8,147,722,752 and 8,146,674,176 for full,
versus 4,963,328,512 and 4,962,279,936 for full_cpu. Full fits with substantial
headroom at this geometry. The experts were not co-resident, and no rollout
workers, reward, optimizer update or larger-video acceptance is established.

RSS observations and elapsed times are retained in the reports but are not
controlled throughput results: full ran first, and its CPU reference output and
gradients remained alive during the subsequent comparison. All GPU processes
exited and the compute inventory was empty after the probe.

Prepared `wan22_rebased_gpu_checkpoint_{single,four}.yaml` using structured config
loading. Both resolve through current `resolve_online_run`. Deep comparison
against the original equal-work reload configs verifies only checkpoint mode
and output/artifact paths changed. The native `wan22_gpu_checkpoint_launch.py`
uses `torchrun -m vrl.scripts.train`, with no backward monkeypatch or pinned-cache
intervention; it retains the 95% threshold and refuses occupied GPUs/output paths.
It also returns a nonzero supervisor status on known Ray memory-kill or policy
release-failure log markers, even when torchrun exits 0. Full runtime/checkpoint
audits are still required. Neither newly prepared arm has run yet.

## Native GPU Checkpoint Four-Rank Pilot Passed

Candidate `29f3e5eb` completed `wan22_rebased_gpu_checkpoint_four` using the native
training entrypoint, full checkpointing, native reload reward lifecycle and no
per-backward diagnostic cleanup. Both updates collected eight global samples
at the unchanged 320x320/17-frame geometry and nine replay steps. Supervisor and
torchrun exited 0 after 727.229s, including startup, saving and shutdown.

Independent `checkpoint_audit.json` passed for checkpoint-1 and checkpoint-2:
1280 FP32 model/Adam/EMA entries, correct progress, four-rank RNG and all adapter
tensors changed on the second update. Each update's maximum pre-update logprob
difference and clip fraction were zero with positive gradient norm.
`first_state_comparison.json` exactly matches the prior full_cpu first checkpoint
across 6412 tensors, four arrays and 8962 scalar leaves. `final_state_comparison.json`
likewise matches checkpoint-final to checkpoint-2, including RNG and optimizer.

The supervisor's runtime log health check passed. All four Raylet logs from
sessions 17:50:38/39 (trainer PIDs 878802-878805) were independently checked:
no above-threshold memory report or worker memory-kill record. No policy-release
cleanup warning occurred. All GPU processes exited; all three CPU audit commands
exited 0. `run_acceptance.json` records this short four-rank pilot as passed.

713 memory samples reached a minimum 25,095,458,816 bytes host available and a
maximum 15,183,446,016 bytes GPU use. Per-rank phase totals were 311.337-311.569s
for update 1 and 258.369-259.171s for update 2; these phase totals exclude some
outer startup/checkpoint/shutdown work and are not an accepted speedup ratio.

Next run the already prepared matching native single-card full-checkpoint arm,
then compare exact workload, quality/correctness evidence and timing boundaries.
Do not repeat the passing four-card pilot merely to wait for more evidence.
This result does not establish long-run stability, single-card scaling, the
separate full-video geometry gate, or completion of the overall multi-GPU goal.
