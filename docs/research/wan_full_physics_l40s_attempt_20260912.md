# Wan full-geometry physics training attempt

Status: **FAILED before optimizer update: replay forward CUDA OOM**.

Candidate `ff2b7857`, three FSDP policy ranks on physical GPUs 0-2 and two
real HTTP reward services on GPU 3. Shared environment unchanged; isolated
Transformers 5.13 overlay, Torch 2.12.0+cu130 and Diffusers 0.38.0.

## Workload

- Real pinned Wan2.1 I2V 14B, VideoPhy reference-image manifest.
- 480x832, 81 frames, 20 denoise steps, CFG 5.0, seed 7.
- Three prompts globally, two samples each: six videos per update.
- Recipe objective: 0.3 Kling motion quality + 0.7 VideoCon physical commonsense.
- BF16 actor FSDP precision, CPU offload, sequential rollout offload, gradient
  checkpointing, microbatch one, compilation off, strict on-policy schedule.
- One requested update, not a long queue or a learning-quality experiment.

## What actually completed

All three ranks generated their two full-size samples. Rank generation walls
were 1692.292, 1703.486 and 1704.172 seconds (about 28.2-28.4 minutes).
Individual sample execution was about 14 minutes. Logged peak generation
allocation was 9546 MB, including decode peaks around 2325-2326 MB.
Periodic utilization samples during denoising showed roughly 95-100% on policy
GPUs; the reward GPU waited for the generated groups. These samples do not
measure achieved FLOPS or constitute a scaling comparison.

Both real reward services completed six result receipts each. This exercises
three real training clients against dedicated HTTP scorers, not just the earlier
single-client reward probe. Each rank then parked its rollout worker; residual
rollout GPU use was about 674 MiB per card.

## Failure

All ranks failed in the replay forward called by
`OnlineTrainer.backward_on_training_batch` -> `_run_replay_pass` ->
`_compute_replay_loss` -> `model.replay_forward`. The exception occurs in Wan
image/text cross-attention at `hidden_states + hidden_states_img`, before any
successful optimizer update or usable replay-parity verdict.

Each rank reports its own logical cuda:0, corresponding to physical GPUs 0-2:

- Additional allocation requested: 1.25 GiB.
- Device total: 44.39 GiB; free: 1.11 GiB.
- Training process use: about 42.58 GiB.
- Torch allocated: 31.01 GiB.
- Torch reserved but unallocated: 10.90 GiB.

This is evidence of a replay-memory failure, not proof that total hardware
capacity is insufficient. Fragmentation is a candidate because substantial
memory was reserved but unused; expandable segments and the actual per-layer
activation peak still need controlled testing. Do not claim either fixes it.
No optimizer update, final checkpoint, nonzero-gradient acceptance or full-size
replay-parity acceptance was obtained. Metrics files contain headers only.

## Artifacts and cleanup

Root: `/mnt/nvme/outputs/wan_i2v_full_physics_l40s`.

- `launch_config.yaml`, `launch_overrides.json`, service YAMLs.
- `train.launch.log`, per-rank `torchrun/` logs and verdicts.
- `train/reward_debug/*_results.jsonl`: six receipts for each reward.
- `gpu_samples.csv`: intermittent samples starting during generation.
- `supervisor_result.json`: training exit 1.

The earlier pre-generation HTTP/local-config conflict is preserved separately
at `/mnt/nvme/outputs/wan_i2v_full_physics_l40s.failed_http_local_config`.
Removing inherited local worker_config from the HTTP client fixed that issue;
all model/device configuration remains in service YAMLs.

Supervisor script: `/mnt/nvme/outputs/wan22_i2v_cache/launch_wan_full_physics.py`.
It now rejects an existing training output directory; use a new `--output-dir`
for subsequent attempts. Prepared verifier `verify_wan_full_physics.py` was not
run because no final checkpoint exists. It is not passing evidence.

Supervisor and owned services are terminal, GPU process inventory is empty.
The old long queue remains disabled. Next action: isolate full-size replay
memory with the same model, conditioning, CFG and FSDP shape before repeating
the expensive generation. Do not shrink geometry and call that this gate.
