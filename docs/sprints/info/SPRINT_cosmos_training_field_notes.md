# Sprint: Cosmos Predict2.5 Training Field Notes (480p/49f NFT run)

状态：**INFO field report（single-L40S historical run）**。本文按 KIND 归档当时的实跑观察，
不是当前 launch runbook，也不拥有 action；旧问题的当前结论与责任归属统一写在 §8。

> Context: this report brought up Cosmos Predict2.5 DiffusionNFT training at
> 832x480 / 49 frames on a single L40S (46GB). Current sources of truth are
> `pyproject.toml`, `docs/sprints/parked/SPRINT_cosmos_predict25_rl_paper_parity.md`
> and the rollout-preview scope recorded in
> `docs/sprints/done/SPRINT_rollout_preview.md`.

## 1. 历史 runbook（不要直接用于当前 HEAD）

以下命令记录当时机器与旧配置布局，只用于解释测量环境：

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
(`vrl/models/families/cosmos/predict2_5/model.py:458`). Note GRPO predict2
(`online_grpo_video_reward`) already ships with `35_step_cfg_7`, so CFG rollouts
are a supported path in this codebase.

**当前结案**：canonical NFT recipe 已明确选择 paper-shaped `20_step_no_cfg`，并在配置中解释
其论文口径；field run 的 CFG 7.0 只是为了让当时的单卡诊断先得到可读输出，不是默认值。
真实输出是否可接受由
`docs/sprints/parked/SPRINT_cosmos_predict25_rl_paper_parity.md` 负责；当前 rollout preview
只覆盖 image `t2i`，不能证明这条 Cosmos video 路径。本文不再持有“决定默认值”的 action，
也不把缺失的 video 验证冒充为已通过。

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

## 4. 历史 dependency crashes

当时环境缺少以下依赖，因此只在 runtime 暴露。当前 `pyproject.toml` 的 `reward` extra 已声明
`qwen-vl-utils`、Transformers、PEFT、Torchvision 与 OpenCV；`decord` 仍不是通用依赖。
目标环境究竟选择哪一个 video backend，必须由实际 Cosmos launch workflow 执行真实 reward
preflight；当前 image-only rollout preview 不加载 reward，也不证明 video backend。本文只记录历史故障，
不维护当前安装清单：

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

以下是当时运行暴露的 knob 语义；当前公开 grammar 已统一为
`rollout.n_samples_per_prompt`、`rollout.prompts_per_batch`、
`rollout.samples_per_chunk`、`rollout.microbatch_size` 与
`actor.replay_samples_per_chunk`，不要照抄旧名字：

- 历史 `rollout.n` 曾是 diffusion group size；当前统一使用
  `rollout.n_samples_per_prompt`，不再按 AR/diffusion 分叉字段名。
- 历史 `rollout.sample_batch_size` 当前名为 `rollout.samples_per_chunk`。
  当时的 `sample_batch_size=4`
  **OOMs** at this resolution (~36 GB allocated for 4 parallel videos);
  `sample_batch_size=2` is the safe ceiling.
- **`algorithm.mini_batch` 是历史 dead parameter，当前已经删除**；不要恢复它。当前样本轴
  replay 分块的唯一事实来源是 `actor.replay_samples_per_chunk`，prompt-group streaming 则由
  `rollout.microbatch_size` 表达。

Two wrong turns I made (recorded so the next person doesn't repeat them):

1. Assumed `mini_batch` would split the training microbatch and lower the peak.
   It's dead; peak was identical for mini_batch=1 and 2.
2. Assumed training peak ∝ `n`. It is NOT: `nvidia-smi` "used" was ~44.9 GB and
   **constant across n=3 and n=4**. That number is `expandable_segments`
   reserved/cached memory, not live allocation — the sb=4 OOM message showed the
   real split (`36 GiB allocated, 6.33 GiB reserved-but-unallocated`). **Do not
   tune off `nvidia-smi` "used" with expandable_segments on; it is cache-padded.**
   Use `torch.cuda.max_memory_allocated()` if you need the real peak.

**当前结案**：没有把 dead `mini_batch` 变活。共享 trainer 已落地
`actor.replay_samples_per_chunk`，两条 replay/backward 路径都消费它，并有分块前后梯度等价、
默认 1、非法值拒绝和 DDP/FSDP 合约测试。Predict2.5 的 paper-shape / 大组真实硬件验收归
`docs/sprints/parked/SPRINT_cosmos_predict25_rl_paper_parity.md`。

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

## 7. 一次性验证资产生命周期

- 当时的 `480p_49f` scratch config 与 single-load `generate_video.py` probe 已在回答问题后删除；
  它们不是当前 import graph 或长期 runbook 的一部分。
- 当前 test-owned rollout preview 的边界记录在
  `docs/sprints/done/SPRINT_rollout_preview.md`；它尚不支持 video。未来 video preview 必须复用
  registry/request/executor 路径，不能复活一个 family-specific 一次性脚本并让它重新成为 action owner。

## 8. 结案与责任归属（本 info 不持有 action）

- **mini_batch：CLOSED / REMOVE + replacement。** dead key 已删除；
  `actor.replay_samples_per_chunk` 是当前 sample-axis replay 分块契约。
- **no-CFG vs CFG：CLOSED / explicit decision。** canonical Predict2.5 NFT recipe 保留
  paper-shaped `20_step_no_cfg`；真实质量证明归 paper-parity，当前 image-only preview 不提供证明。
- **reward video dependency：TRANSFERRED。** `qwen-vl-utils` 已进入 `reward` extra；`decord`
  是否是目标环境必需 backend 由实际 launch 的 reward preflight 判定，不在 field report 留未结事项。
- **Predict2 V2W rollout verification：CLOSED。** 当前完成记录
  `docs/sprints/done/SPRINT_cosmos_robotic_data_factory_domain_rl.md` 已验证原生
  704p/93f 能生成连贯机器人视频，同时证明 240p/33f 是 OOD 垃圾；这不是仍待执行的 probe。
