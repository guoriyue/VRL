# SPRINT (info / measurement archive): Cosmos Predict2.5-2B + Kling GRPO/NFT reward run

状态：measurement archive（`info/`）。这是一次单卡训练观测记录，**不是 action item**；保留下来供以后复查。
日期：2026-06-16，单张 RTX 5090（32GB，host RAM 94GB），VRL @ `main`（含 `microbatch_size` streaming）。

## TL;DR

- 单卡 Cosmos Predict2.5-2B（LoRA）DiffusionNFT + Kling VideoReward 跑了 **14 epoch / ~6h50m**，然后被一个**瞬时网络超时**（HuggingFace ReadTimeout）打断，不是 OOM、不是代码 bug。
- **reward 在噪声内持平**（-5.12 ↔ -5.05 来回跳，无趋势），复现了 2026-06-13 的结论。`grad_norm ~0.05–0.12`（极小）= per-step 梯度太小这个根因。
- **度量本身无法显示 learning**：每个 epoch 训练在 309 prompt 集里**轮换的 16 个不同 prompt** 上，所以 `reward_mean` 主要反映"这轮抽到哪些 prompt"的难度，而非策略变化。要判断学习必须用**固定 eval prompt 集**逐 epoch 打分（prior 也提过）。
- **512p 视频 RL 装不进单张 32GB 卡**（policy + 同卡常驻的 ~5GB VideoReward）。OOM ladder 记录在下面。最终用 256p/49f 才跑起来。
- 被验证为**好用**的：`microbatch_size` streaming 在真实 cosmos run 里端到端正确（gas 由 microbatch_size 派生、host-RAM guard 生效、每 epoch 一次 optimizer step、reward 真打分）。唯一的墙是 GPU 容量，不是这套代码。

## 运行配置

Entrypoint `vrl.scripts.diffusion.train:train_diffusion_online`, config
`experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward`，override：

```
rollout.prompts_per_batch=16
actor.microbatch_size=1              # -> gradient_accumulation_steps 派生 = 16
rollout.n_samples_per_prompt=8
actor.host_memory_budget_fraction=0.95
sampling.width=256 sampling.height=256 sampling.num_frames=49 sampling.num_steps=20
trainer.total_epochs=50 trainer.save_freq=10
production.kling_video_reward.enabled=false      # 跳过 production-report preflight，仍作训练 reward
env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

每 epoch = 一次 optimizer update = 16 个 microbatch × 8 sample = 128 段视频；reward = Kling VideoReward（本地 HF 模型 `KlingTeam/VideoReward`，不需要 API key）。

## reward 曲线（14 epoch）

```
epoch  reward_mean  grad_norm
0      -5.1228      0.064
1      -5.1184      0.103
2      -5.1006      0.120
3      -5.1099      0.067
4      -5.1418      0.078
5      -5.0658      0.073
6      -5.1437      0.049
7      -5.0714      0.066
8      -5.0739      0.249
9      -5.1403      0.063
10     -5.1535      0.053
11     -5.1075      0.092
12     -5.0812      0.092
13     -5.0460      0.086
```

净变化 +0.077 / 13 epoch，但 epoch 间摆动 ±0.1（e5 −5.07 → e6 −5.14 → e10 −5.15 → e13 −5.05）。
末尾 3 连升约 1.5σ，**不是信号**。`grad_norm` 全程很小。结论：噪声内持平。

## 关键观测

1. **持平复现 prior。** 与 2026-06-13 单卡 cosmos+Kling GRPO（`ppo_epochs=1`）一致：reward 不动。杠杆是 per-step 梯度量级（更多 inner step / full-param 取代 LoRA / 更大 LR / diffusion-loss 正则），**不是 epoch 数**。
2. **轮换 prompt 集让训练 reward 不可解读为学习。** 需要固定 eval prompt 集 + block test。training `reward_mean` 的逐 epoch 抖动主要是 prompt 难度差异（每 epoch 抽不同的 16 个）。
3. **GPU 容量天花板（单卡 32GB）。** `sample_batch_size` 已是 1（peak = 单段视频前向），无更小的 chunk 杠杆。OOM ladder（已设 `expandable_segments`）：
   - 512p/93f：缺 ~3GB（峰值 ~28.4GB）
   - 512p/49f：仅缺 ~0.3GB（峰值 31.08/31.33GB）
   - 256p/49f：装得下，跑通
   要在单卡 32GB 上真跑 512p：降到 ≤384p、或 512p 约 41 帧（勉强）、或给 reward 第二张卡。
4. **VideoReward `min_frame_pixels=200704`（=448²）。** 生成分辨率低于 448² 时 reward 内部会 upscale——读趋势没问题，但不是 native-res 打分（本次 256p 即 upscaled 打分）。
5. **全分辨率 50 epoch 与内存无关也是多天级**；想要"一夜出 50 点曲线"必须 ≤256–320p。
6. **崩溃 = 瞬时网络超时，非代码问题。** `release_after_collect=true` 下 rollout worker 每个 cycle 被 kill+relaunch、用 diffusers 重新加载 Cosmos 模型，会 ping `huggingface.co` 的 `model_info`（未走 offline）。几千次 reload 后某次 ping 在 07:29 超时 → 整个 run 挂。**修复：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`**（三个模型都已本地缓存）。

## 复现 / 续跑（offline-robust）

resume 支持：`trainer.resume_from=<checkpoint dir>`（`vrl/trainers/checkpointing.py:load_training_checkpoint_from_config`）。本次 `checkpoint-10` 完整（`lora_weights/adapter_model.safetensors` + `checkpoint.pt`）。从 epoch 10 续跑 11→50：

```bash
cd ~/Desktop/VRL && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  rollout.prompts_per_batch=16 actor.microbatch_size=1 rollout.n_samples_per_prompt=8 \
  actor.host_memory_budget_fraction=0.95 \
  sampling.width=256 sampling.height=256 sampling.num_frames=49 sampling.num_steps=20 \
  trainer.total_epochs=50 trainer.save_freq=10 \
  trainer.resume_from=outputs/cosmos25_kling_50ep/checkpoint-10 \
  trainer.output_dir=outputs/cosmos25_kling_50ep production.kling_video_reward.enabled=false
```

## 下次该做的（不是这次的目标，记下来）

1. **加固定 eval prompt 集，逐 epoch 打分** —— 否则曲线无法判断学习。最高优先级。
2. **攻 per-step 梯度**（不是加 epoch）：更多 inner step、full-param 取代 LoRA、更大 LR、diffusion-loss 正则。
3. **基础设施**：训练默认带 `HF_HUB_OFFLINE=1`（weights 已缓存时）以免瞬时网络打断长跑；考虑让 model reload 走 `local_files_only`，避免 release-after-collect 每 cycle ping HF。
4. **想看 native-res reward**：≥448p 生成，但单卡 32GB 下 512p 装不下 → 需要第二张卡给 reward，或 448p 减帧。

## 关联

- 本 run 验证了同步 `microbatch_size` streaming（见 `done/SPRINT_streaming_rollout_accumulation.md`、
  `done/SPRINT_memory_budgeted_microbatch.md`）在真实 cosmos 上端到端可用；它不是 async overlap。
- 复现 2026-06-13 "first trustworthy curve" 的持平结论。
- 同类容量/配方坑点也记在个人 memory `project_cosmos_streaming_smoke.md`。

## 关键文件 / 产物

- config：`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- streaming + host-RAM guard：`vrl/scripts/common/online.py:_run_streaming_optimizer_update`
- size↔count 派生：`vrl/config/builders.py:build_online_batch_plan`、
  `vrl/trainers/online/config.py:OnlineBatchPlan`
- resume：`vrl/trainers/checkpointing.py`、`trainer.resume_from`
- 本次产物：`outputs/cosmos25_kling_50ep/metrics.csv`（14 行）、`outputs/cosmos25_kling_50ep/checkpoint-10/`
