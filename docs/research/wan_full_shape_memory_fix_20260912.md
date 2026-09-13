# Wan full-shape FSDP memory diagnosis

Status: **real-weight, full-shape forward/backward diagnostic passed;
real rollout training update remains open**.

The subsequent full-size real run passed generation, both reward services and
multiple replay/backward timesteps without the original OOM, but simultaneous
Ray SIGTERM interrupted it before the optimizer update. It is not an end-to-end
pass. See `wan_full_physics_cpu_checkpoint_interruption_20260912.md` for exact
scope, termination evidence and released hardware ownership.

Implementation candidate: `382d0825` in `/home/ubuntu/VRL-mgpu-integration`.
Shared dependencies unchanged. Torch 2.12.0+cu130, Diffusers 0.38.0 and isolated
Transformers 5.13.0. Three L40S policy GPUs; no reward service or rollout worker
in this diagnostic. All processes are terminal and GPU inventory is empty.

## What was measured

Use the real pinned Wan2.1 I2V 14B transformer, original 32/64 LoRA targets,
three-rank FSDP with CPU parameter offload and full block checkpointing.
Synthetic latent/conditioning tensors retain the complete production CFG
shape: latent `[2,36,21,60,104]`, text `[2,512,4096]`, image `[2,257,1280]`.
This corresponds to the full 480x832, 81-frame geometry, not a reduced model
or smaller video. Synthetic values and a squared-output loss isolate memory;
they do not establish rollout/replay parity or real-reward optimizer behavior.

| Mode | Forward | Backward | Result |
| --- | --- | --- | --- |
| Default allocator, full checkpoint | 31.95 s | OOM | 30.05 GiB allocated, 12.76 GiB reserved/unused at failure |
| Expandable segments, full checkpoint | 29.52 s | OOM | 42.83 GiB allocated, only 75.89 MiB reserved/unused |
| Expandable + whole-forward CPU saves | 55.07 s | 80.73 s | Passed, peak allocated 27,677,758,464 bytes |
| Expandable + per-block `full_cpu` | 33.93 s | 81.15 s | Passed, same peak allocated |

Times above are rank 0, one call each, not a statistical benchmark. All three
ranks passed both CPU-save variants with 400 finite nonzero gradient tensors
per rank. Per-block forward ranges 33.81-34.37 s, backward 80.87-81.21 s.
Peak allocated is about 25.8 GiB; per-block peak reserved 30,830,231,552 bytes.

Block-entry allocation under full checkpointing grows from 2.88 GB at block 0
to 9.58 GB at block 10, 16.29 GB at block 20, 23.00 GB at block 30 and 29.04 GB
at block 39. Full block recomputation still retains its inputs. Moving those
inputs to CPU addresses this cross-block GPU retention. The expandable-only
failure demonstrates that fragmentation is not the only problem.

## Implementation and verification

New explicit `actor.gradient_checkpointing=full_cpu` uses PyTorch non-reentrant
checkpointing inside `save_on_cpu(pin_memory=True)`. Existing modes and defaults
are unchanged. Unsupported custom checkpoint APIs fail rather than falling back;
NextStep's bool-only loader rejects the mode. Replay compilation remains rejected.

215 related config/model/checkpoint tests passed before the additional CUDA
case. The final checkpoint helper suite passes 11 tests with CUDA enabled,
including exact output, input-gradient and parameter-gradient equality with
dropout on both CPU and CUDA. These small numerical tests complement, but do
not replace, full-size hardware memory and finite-gradient checks.

Probe: `/mnt/nvme/outputs/wan22_i2v_cache/wan_full_shape_memory_probe.py`.
Reports: `/mnt/nvme/outputs/wan_full_shape_memory_{default,expandable,cpu_saved,block_cpu}`,
one JSON per rank with shape, block-memory events, status, errors and timing.
Matching logs are in the wan22_i2v_cache directory.

The full training supervisor now defaults to `full_cpu` with the default CUDA
allocator (see the follow-up below), retains generated reward artifacts
for inspection and refuses to overwrite a prior training directory. No complete
real rollout/update has yet been rerun with the fix. The original full-size OOM
evidence remains unchanged; the next gate must complete the real physics update.

## Default allocator follow-up

The real full-shape three-rank diagnostic also passes with `full_cpu` and both
allocator environment variables unset. Rank 0: forward 33.85 s, backward 81.14 s,
peak allocated 27,685,304,832 bytes, peak reserved 35,035,021,312 bytes, 400
finite nonzero gradient tensors. All three ranks succeeded; torchrun exited 0.
Reports: `/mnt/nvme/outputs/wan_full_shape_memory_block_cpu_default`.

The intervening real-training attempt at
`/mnt/nvme/outputs/wan_i2v_full_physics_cpu_ckpt_l40s` used expandable segments.
Codex stopped it during the first video's denoising after observing slow initial
progress and lower utilization. This suggests an allocator/offload interaction,
but it is NOT a completed throughput comparison or proof of its cause. No
complete sample group, reward batch, replay pass or optimizer update completed.
Torchrun needed forceful rank cleanup after SIGTERM; supervisor and services
are terminal and GPU inventory is empty. Logs and partial outputs are preserved.

Since expandable segments are not necessary for the successful full-shape
diagnostic, the supervisor now clears both allocator environment variables by
default. `--allocator expandable` explicitly opts back in. It records the chosen
allocator and device masks in `launch_environment.json`. The next full run must
use a fresh output directory and still prove real rollout/replay/update success.
