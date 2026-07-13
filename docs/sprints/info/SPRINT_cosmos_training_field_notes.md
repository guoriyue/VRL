# Sprint: Cosmos Predict2.5 Training Field Notes (480p/49f NFT run)

状态：observations (field report from a real single-L40S run; some items are
actionable fixes, some are runbook knowledge)

> Context: brought up Cosmos Predict2.5 DiffusionNFT training at 832x480 / 49
> frames on a single L40S (46GB) box. This documents every non-obvious thing
> that bit us, in the order it bit, so the next run is a one-shot. Cross-refs:
> `SPRINT_env_bootstrap.md` (dependency declaration), memories
> `nvme-hf-cache-fast-model-load`, `env-bootstrap-undeclared-deps`.

## 1. 结论 / TL;DR runbook

To launch a correct, stable Cosmos Predict2.5 run on this box:

```bash
# prerequisites (one-time)
pip install qwen-vl-utils decord          # reward backend + video reader (UNDECLARED in pyproject)
# HF cache must live on local NVMe instance store, NOT the EBS root volume
# guidance_scale > 1 enables mandatory CFG.

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/mnt/nvme/hf \                      # NVMe cache: model load 15min -> 7s
FORCE_QWENVL_VIDEO_READER=decord \          # torchvision 0.27 dropped read_video
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_video_reward \
  /sampling/video=480p_49f \
  sampling.num_steps=10 sampling.guidance_scale=7.0 \
  rollout.n=3 rollout.sample_batch_size=2 \
  trainer.total_epochs=50 trainer.output_dir=outputs/<name> \
  trainer.save_freq=20 trainer.log_freq=1
```

The four things that each fully blocked the run, in order: missing
`qwen_vl_utils`; 15-minute model loads off EBS; **color-block rollouts from the
shipped no-CFG sampling**; reward-scoring crash from `torchvision.read_video`
removal. None are visible until you actually run end-to-end.

## 2. Rollout correctness: CFG is mandatory (the headline finding)

The shipped `cosmos_predict2_5/online_nft_video_reward` recipe uses
`/sampling/denoise/10_step_no_cfg` → `guidance_scale=1.0` (no classifier-free
guidance). **On the base Predict2.5 checkpoint this produces incoherent color
blocks, not video.** The reward model then scores garbage and RL has no real
signal.

Isolated by generating the same prompt three ways (probe = VRL rollout path,
plus the native diffusers pipeline as ground truth):

| steps | guidance | result |
|---|---|---|
| 10 | 1.0 (no CFG) — shipped | **color blocks** |
| 10 | 7.0 | **clear video** |
| 35 | 7.0 | clear video |
| native pipeline default (36 / 7.0) | | clear video |

→ The driver is **CFG, not step count**. 10 steps is fine *with* guidance.
Native `Cosmos2_5_PredictBasePipeline.__call__` defaults are `num_inference_steps=36,
guidance_scale=7.0, num_latent_conditional_frames=2` — i.e. the model is tuned
to be used with guidance.

Why no-CFG may be intentional for NFT (and why we overrode it anyway): no-CFG =
the model's true conditional distribution = a cleaner policy gradient. But the
base model's no-CFG samples are unusable, so the reward signal is meaningless
at start. We chose `guidance_scale=7.0` so rollouts are correct *now* (user
goal: "rollout correct videos"). The training replay carries `guidance_scale`
via `export_batch_context`, so rollout and replay stay CFG-consistent
(`vrl/models/diffusion/cosmos/predict2_5/model.py:399`). Note GRPO predict2
(`online_grpo_video_reward`) already ships with `35_step_cfg_7`, so CFG rollouts
are a supported path in this codebase.

**Action item:** either change the predict2.5 NFT default denoise to a CFG
preset, or add a loud note in the config that no-CFG rollouts are blocks on the
base model.

## 3. Model loading: HF cache MUST be on local NVMe, not EBS

First loads took **10–20 minutes each** (and the run loads the policy twice:
trainer replay bundle + Ray rollout worker). Diagnosed cold:

- process in `D` state (uninterruptible IO wait); `read_bytes` climbing while
  `rchar` flat → reads via **mmap page faults**, not `read()`.
- `iostat`: `r_await ≈ 84 ms`, ~24 IOPS, ~6 MB/s. The killer is mmap's
  serial, queue-depth≈2 fault pattern over high-latency **EBS** (network disk).
- It is NOT EBS throughput throttling: `dd bs=1M` sequential (after
  drop_caches) = **149 MB/s**. Only the low-QD mmap pattern is slow.

Fix (validated): the box has an unused **419GB EC2 NVMe instance store at
`/dev/nvme1n1`**. Format ext4, mount `/mnt/nvme`, copy the model dirs with
`cp -a` (NOT rsync — rsync ran at 6 MB/s for unclear reasons; cp got ~87 MB/s),
launch with `HF_HOME=/mnt/nvme/hf`. **Cosmos Predict2.5 load: ~15 min → 6.7 s.**

Gotchas:
- HF cache total here is **147 GB** (wan/sd3.5/nextstep/etc.) — copy only the
  models you need, not the whole cache.
- EBS has **burst credits that deplete**: early in the session sequential reads
  hit ~33–149 MB/s; after sustained reading they fell to ~6 MB/s baseline. So
  the one-time copy itself can be slow once burst is gone.
- Instance store is **ephemeral** (lost on stop/terminate) — re-copy after any
  instance stop. It's a cache, re-downloadable.

## 4. Undeclared dependencies (two separate runtime crashes)

`pyproject.toml` omits packages the reward path imports; each surfaced only at
runtime (see `SPRINT_env_bootstrap.md` for the full list):

1. `qwen_vl_utils` — driver preflight `preflight_kling_video_reward_backend()`.
   Fix: `pip install qwen-vl-utils` (pulls `av`).
2. **`torchvision 0.27.0+cu130` removed `torchvision.io.read_video`** (moved to
   torchcodec). `qwen_vl_utils` falls back to it to read the rollout mp4, fails,
   the RewardModelWorker dies, and **the dead Ray actor takes down GCS → the
   whole run terminates** (~`Failed to connect to GCS within 60 seconds`). Fix:
   `pip install decord` + `FORCE_QWENVL_VIDEO_READER=decord`. decord reads the
   832x480x49 mp4 fine.

The Kling VideoReward repo (`KlingTeam/VideoReward@main`) is a `checkpoint-*`
LoRA over `Qwen/Qwen2-VL-2B-Instruct`; the base is downloaded at runtime, so
pre-stage it on NVMe too (`hf download` writes ~400 MB/s, network-bound, fine).

## 5. Single-GPU memory model (the part I got wrong twice)

Everything time-shares one L40S: rollout / reward / train run **serially** with
`release_before_reward_model` + `release_after_score`, so at any instant only
one phase's model is resident. `nvidia-smi` shows whatever phase you caught.
Per-phase footprints (832x480x49, 10 steps):

- **generation** (RayGenerationWorker): ~28–29 GB at `sample_batch_size=2`.
- **training** (replay/backward, main proc): the higher phase.

Knob semantics (verified in code):

- `rollout.n` is the diffusion group-size field. **`n_samples_per_prompt` is
  AR-only** (`collector/core.py:204`: `1 if kind=="diffusion" else
  require("n_samples_per_prompt")`). Live proof: setting `rollout.n` made
  `metrics.csv` `group_size` track 2→4→3. `builders.py:55` prefers
  `n_samples_per_prompt` then falls back to `rollout.n`, so both *can* work, but
  diffusion configs use `rollout.n`.
- `rollout.sample_batch_size` = generation microbatch (`max_samples_per_chunk`,
  from `cfg.rollout.sample_batch_size`, launcher.py:325). `sample_batch_size=4`
  **OOMs** at this resolution (~36 GB allocated for 4 parallel videos);
  `sample_batch_size=2` is the safe ceiling.
- **`algorithm.mini_batch` is a DEAD parameter** — defined at
  `vrl/algorithms/diffusion_nft.py:26` and consumed *nowhere* (grep proves it).
  Setting it changes nothing.

Two wrong turns I made (recorded so the next person doesn't repeat them):

1. Assumed `mini_batch` would split the training microbatch and lower the peak.
   It's dead; peak was identical for mini_batch=1 and 2.
2. Assumed training peak ∝ `n`. It is NOT: `nvidia-smi` "used" was ~44.9 GB and
   **constant across n=3 and n=4**. That number is `expandable_segments`
   reserved/cached memory, not live allocation — the sb=4 OOM message showed the
   real split (`36 GiB allocated, 6.33 GiB reserved-but-unallocated`). **Do not
   tune off `nvidia-smi` "used" with expandable_segments on; it is cache-padded.**
   Use `torch.cuda.max_memory_allocated()` if you need the real peak.

**Real lever for higher `n` without OOM:** the NFT replay/backward processes all
`n` samples of a prompt at once, so the *real* allocated training peak grows with
`n`. To go past the ceiling, make `mini_batch` live — chunk the replay/backward
over samples with gradient accumulation in `diffusion_nft.py`. That reduces the
real allocated peak (not the cache). Not yet implemented.

## 6. Training dynamics / reward reading

- **"epoch" = one prompt-group step**, not a dataset pass (`online.py:221`
  `for epoch in range(total_epochs)` → sample `rollout_batch_size` prompts →
  `trainer.step`). So `total_epochs=50` = 50 optimizer steps.
- `rollout_batch_size=1` → each step's `reward_mean` is dominated by **which
  random prompt got sampled** (a hard "text" prompt scores far lower than a
  scene), not policy quality. Over the first ~8 steps reward bounced -1.2…-3.6
  with no trend — expected; RL on diffusion needs hundreds of steps, and the
  learning signal is the **within-group `reward_std`** (advantage), not the
  cross-step mean. To see a trend sooner, raise `rollout_batch_size` to smooth
  per-step prompt noise.
- Throughput on this box: ~7–8 min/epoch at n=3/sb=2 → 50 epochs ≈ 6 h.

## 7. Tools added this run

- `configs/sampling/video/480p_49f.yaml` — 832x480, 49 frames (49 satisfies the
  Cosmos temporal-VAE `(frames-1) % 4 == 0` constraint).
- `vrl/scripts/diffusion/cosmos/generate_video.py` — single-load generation
  probe (no Ray/reward/trainer) that replays the rollout denoise path
  (`forward_step` + `sde_step_with_logprob`) and writes an mp4. Use it to verify
  rollout correctness in ~1 min before committing to a multi-hour run. This is
  how the CFG finding was isolated.

## 8. Open / TODO

- Implement `mini_batch` chunking in `diffusion_nft.py` to unlock n≥8 safely.
- Decide on the no-CFG-vs-CFG default for the predict2.5 NFT recipe (§2).
- Fold `decord` into the `reward` extra in `SPRINT_env_bootstrap.md`.
- predict2 (video2world) rollout verification still pending (model is on NVMe;
  probe deferred to avoid GPU contention with the live training).
