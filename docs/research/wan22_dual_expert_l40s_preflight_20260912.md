# Wan 2.2 dual-expert L40S launch preflight

Status: CPU-only preflight passed; released-weight GPU execution remains open.
No GPU claim or job was launched. The prior Wan full-size Ray termination
requires coordination before another expensive hardware run.

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
