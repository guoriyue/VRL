# SPRINT (info / runbook): Cosmos Predict2.5-2B + Kling — paper-aligned RL recipe & reward-confirmation loop

状态：runbook（`info/`，长期复用）。这是把 Cosmos-Predict2.5 论文 §4.2.2 的 RL 配方落到单卡可跑的步骤，
并用**固定 prompt+seed eval** 确认 reward 是否真的在涨。承接 `SPRINT_cosmos25_kling_reward_curve.md`
（2026-06-16 那次持平的观测记录）。配方改动已落在 VRL `main`（commit `5738914`）。

---

## TL;DR

- **病根不是代码、不是 pipeline overlap，是训练量。** 论文 §4.2.2：*"trained for 256 steps with a batch
  size of 32"*；之前的 run 只跑了 **14 步**（config 还写 `total_epochs: 10`），且 LoRA 用了论文给
  full-param 的 `3e-5`（grad_norm ~0.1，太小）→ reward 在噪声里持平。
- **配方已对齐论文 + LoRA 现实**（commit `5738914`）：`total_epochs 10→256`、`lr 3e-5→1e-4`、`save_freq→32`。
  diffusion-loss 正则已在 DiffusionNFT 的双 adapter 里、EMA 已开，无需另加。
- **论文规模单卡跑不动**（rbs=32 / 256p ≈ 58min/epoch × 256 ≈ ~10 天）。所以 config 是"论文参考"，
  单卡用 **override 跑一个可行探针**（rbs=16 / 64 步 ≈ ~1 天），先确认 reward 动不动，再决定是否上多卡跑满。
- **判据是固定 eval，不是训练曲线。** 训练期 `reward_mean` 每 epoch 轮换 16 个不同 prompt，读不出学习；
  用现成的 `vrl/scripts/eval/cosmos_predict25_kling_eval.py`（同 prompt 同 seed 跨 checkpoint 打分）才算数。

---

## 1. 论文 §4.2.2 → 配置映射

| 论文 | 配置（`online_nft_kling_video_reward.yaml`） | 状态 |
|---|---|---|
| trained for **256 steps** | `trainer.total_epochs: 256`（每 epoch = 一次 optimizer update） | ✅ 已改 |
| **batch size 32**、8 outputs/condition | `rollout_batch_size: 32`、`n_samples_per_prompt: 8` | ✅（streaming 让单卡也放得下） |
| 20 diffusion steps，训 10（½） | `sampling.num_steps: 20`、`timestep_fraction: 0.5` | ✅ |
| reward 组内归一（GRPO） | `algorithm.global_std: false` | ✅ |
| diffusion loss 正则（抗 reward hacking） | DiffusionNFT 双 adapter（trainable `default` + frozen `previous`，`kl_beta=1`） | ✅ 内建 |
| 释放 EMA 权重为最终 ckpt | `actor.ema.enable: true` | ✅ |
| full-param，lr 3e-5（2B） | **LoRA**（full-param 要多卡），`lr: 1e-4` | ⚠️ LoRA 偏离：见下 |

**为什么 LoRA 用 1e-4 而不是论文的 3e-5**：论文 post-train **full-param**；本地这条 recipe 是 LoRA
（`use_lora: true`，full-param 需要单独的 previous-policy 状态路径 + 多卡）。LoRA adapter 要更大的 LR 才推得动
策略——3e-5 在 2026-06-13 单卡 run 上给出 grad_norm ~0.05–0.12、reward 持平。若 `approx_kl` / `logprob_abs_diff`
开始 drift，再往 3e-5 方向回调。

---

## 2. 单卡可行性（为什么要 override）

来自 `SPRINT_cosmos25_kling_reward_curve.md` 实测：单张 5090（32GB）、256p/49f、20 步、`rbs=16`、`n=8`
≈ **29 min/epoch**。所以：

| 规模 | 每 epoch | 256 步 | 备注 |
|---|---|---|---|
| 论文 shape：rbs=32 / 256p | ~58 min | **~10 天** | config 默认，单卡不现实 |
| 探针：rbs=16 / 256p | ~29 min | 64 步 ≈ **~31 h** | 推荐：看 reward 动不动 |
| overnight：rbs=8 / 256p | ~15 min | 80 步 ≈ **~20 h** | 更快、advantage 估计更糙 |

- **512p 单卡装不下**（policy + 同卡常驻 ~5GB VideoReward；512p/93f 峰值 ~28–31GB 还差几 GB）。单卡只能 256p；
  想要 ≥448p 的 native-res reward 需要给 reward 第二张卡。
- `microbatch_size=1` streaming 让 host RAM 只持有 ~1 组，所以 `rbs` 大小不再撑爆 host；GPU 峰值由
  `sample_batch_size=1`（单段视频前向）决定，与 `rbs` 无关。

---

## 3. 训练命令（单卡探针，~1 天）

```bash
cd ~/Desktop/VRL && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  rollout.rollout_batch_size=16 trainer.total_epochs=64 trainer.save_freq=16 \
  sampling.width=256 sampling.height=256 sampling.num_frames=49 \
  rollout.host_memory_budget_fraction=0.95 production.kling_video_reward.enabled=false \
  trainer.output_dir=outputs/cosmos25_kling_probe_1e4_64ep
```

- `lr=1e-4`、`timestep_fraction=0.5`、`n=8`、20 步 来自 config，不用再写。
- 64 步 @ 1e-4 ≈ 持平那条 run（14 步 @3e-5）约 **15× 的累计权重位移**，够看出趋势。
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 必带：上次崩溃是 `release_after_collect` 每 cycle reload
  Cosmos 时 ping HF `model_info` 的瞬时超时，不是 OOM/代码 bug（三个模型都已本地缓存）。
- overnight 版：把 `rollout.rollout_batch_size=8 trainer.total_epochs=80 trainer.save_freq=20`。
- 断点续跑：`trainer.resume_from=<output_dir>/checkpoint-N`（`vrl/trainers/checkpointing.py`）。

---

## 4. Eval 命令（判据：固定 prompt+seed 逐 checkpoint）

用现成的 controlled eval（**不是**训练曲线）：同 prompt 同 seed 在各 checkpoint 上重生成 + Kling 打分。

```bash
cd ~/Desktop/VRL && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python -m vrl.scripts.eval.cosmos_predict25_kling_eval \
  --checkpoint e16=outputs/cosmos25_kling_probe_1e4_64ep/checkpoint-16 \
  --checkpoint e32=outputs/cosmos25_kling_probe_1e4_64ep/checkpoint-32 \
  --checkpoint e48=outputs/cosmos25_kling_probe_1e4_64ep/checkpoint-48 \
  --checkpoint e64=outputs/cosmos25_kling_probe_1e4_64ep/checkpoint-64 \
  --eval-manifest --samples-per-prompt 4 --seed 20260616 \
  sampling.width=256 sampling.height=256 sampling.num_frames=49 \
  --output-dir outputs/cosmos25_kling_probe_eval
```

- 看 `outputs/cosmos25_kling_probe_eval/summary.json` 里各 label 的 `mean`。
- **同 seed 跨 checkpoint**（脚本写死 `_seed_for` 忽略 checkpoint_index）→ 分差只反映权重，不是采样噪声。
- 要更稳的判断：`--samples-per-prompt` 调大（Kling 方差大），eval prompt 用固定的 `--manifest <file>`。
- **想要 pre-RL 基线**：跑训练时多存一个早期 ckpt（或单独 `trainer.total_epochs=0` 存初始 LoRA），
  当 `--checkpoint baseline=...` 加进来；否则用最早的 `e16` 作参考看相对趋势。

---

## 5. 怎么读 + 之后的分支

- **summary mean 随 epoch 单调上升**（e16 < e32 < e48 < e64）→ recipe 成立。再决定是否上多卡跑满 256 步 / 512p
  native-res / full-param。
- **还是平** → 按顺序加杠杆（都是"加大 per-step 信号"，不是加 epoch）：
  1. **lr 再加**：`approx_kl` 才 ~0.002、headroom 很大，可试 `actor.optim.lr=2e-4`。
  2. **full-param 取代 LoRA**：需要第 2 张卡（previous-policy 单独状态路径）。
  3. **reward 信噪比**：eval `--samples-per-prompt` 调大；native-res（≥448p）打分需要 reward 第二张卡。
  4. **吞吐优化另走 rollout/train async**：`SPRINT_microbatch_pipeline_overlap` 已退役为 scope guard；
     不再做 microbatch async。确认会涨之后，再看多卡 rollout/train overlap。

---

## 6. 关键文件 / 引用

- 配方：`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`（commit `5738914`）
- 训练入口：`vrl.scripts.diffusion.cosmos.train:train_cosmos_predict25_diffusion_nft` → `run_online_recipe`
- streaming + host-RAM guard：`vrl/scripts/common/online.py:_run_streaming_optimizer_update`
- size↔count 派生：`vrl/trainers/core/types.py:TrainerConfig.__post_init__`
- 固定 eval：`vrl/scripts/eval/cosmos_predict25_kling_eval.py`（同 prompt+seed、Kling 打分、per-ckpt summary）
- DiffusionNFT 双 adapter（= 论文的 diffusion-loss 正则）：`vrl/algorithms/diffusion_nft.py`、
  `configs/model/diffusion/cosmos/predict2_5_2b.yaml`
- 论文：`docs/papers/cosmos_predict2_5_world_simulation_with_video_foundation_models_for_physical_ai_2511.00062v2.pdf` §4.2.2
- 上一次观测（持平）：`docs/sprints/info/SPRINT_cosmos25_kling_reward_curve.md`
