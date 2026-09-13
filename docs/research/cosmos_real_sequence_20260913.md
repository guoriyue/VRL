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
