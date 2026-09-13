# Wan 2.2 dual-expert L40S launch preflight

Status (2026-09-13): one bounded released-weight T2V dual-expert update
completed, but strict rollout/replay equivalence was not accepted. Subsequent
work was intentionally stopped. The original CPU preflight and tiny CUDA
prerequisites below are historical; the real diagnostic is recorded last.

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
