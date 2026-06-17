# SPRINT (planned): 让 Cosmos+Kling RL 的 reward 真正动起来（且看得见）

状态：planned。针对 `info/SPRINT_cosmos25_kling_reward_curve.md` 记录的"reward 持平"问题的修复方案。
所有 RL 配方参数都对照过 **Cosmos Predict2.5 论文 §4.2.2**（`docs/papers/cosmos_predict2_5_..._2511.00062v2.pdf`，wm-infra clone）和 VRL 真实代码。

## 核心结论（被论文改写过的优先级）

**配方本身是对的、且论文证明它能涨 reward——我们的"平"主要是三件事:(1) 看不见、(2) 步数太少、(3) 单卡被迫降配。不是参数错。**

论文 §4.2.2 用**完全相同的配方**（VideoAlign reward、每 condition 8 个输出、20 denoise 步、GRPO 组内归一化、batch=32）把 2B 的 reward 从 1.08→1.69（Text2World sum, Table 6），人评 RL 胜率 41%（Fig 5）。所以方向没问题。差距在:

| 维度 | 论文 | 我们这次 | 影响 |
|---|---|---|---|
| 训练步数 | **256 步** | **14**（崩溃) | 只跑了 ~5%,reward 还没机会动 |
| 度量 | PAI-Bench / 固定 eval + 人评 | 轮换训练 prompt 的 `reward_mean` | **结构上看不见学习** |
| 分辨率/帧 | 多卡, 高分辨, 93f | 单卡 32GB → 256p/49f | 信号弱化 + VideoReward 要 upscale |
| 参数 | 2B（论文未明确 LoRA/全参,多卡 FSDP2） | 单卡 LoRA-only（全参在本卡崩, `model.py:239`） | 表达力天花板 |

## 论文确认 / 否定了哪些参数（用户要求核对的部分）

**论文 §4.2.2 (p13) 确认——这些当前配置就是对的,别改:**
- `rollout.n_samples_per_prompt: 8`（"generate eight outputs for each input condition"）✓
- `sampling.num_steps: 20`（"with 20 diffusion steps"）✓
- `rollout.rollout_batch_size: 32`（"batch size of 32"）✓
- GRPO 组内归一化（"normalizing the reward within its rollout group, following GRPO"）= `advantages.py:group_relative_advantages` ✓
- reward model = VideoAlign（Liu et al. 2025）= 本仓库的 `KlingTeam/VideoReward` ✓
- **diffusion loss 正则**（"we use diffusion loss on the fine-tuning dataset for regularization, which effectively alleviates the reward hacking"）→ 对应 NFT 的 `kl_beta`（`diffusion_nft.py:259-261`,对 base 的 L2）。**论文说它是用来防 reward-hacking 的,所以"降低它"有风险,不是免费的学习增益。**
- **每个 batch 一次参数更新**（"accumulate the gradient ... for one parameter update"）→ 论文就是 `ppo_epochs=1`,靠 **256 步** 而不是内层 epoch 拿到提升。
- EMA 权重（"release the EMA weight after post-training"）。
- 训练步数 = **256**（"trained for 256 steps"）。

**论文未确认 / 属于经验偏离的:**
- `advantage_high` / `adv_clip_max` 具体值:论文只说 GRPO 归一化,没给"优势乘子"。这是 DiffusionNFT (Ye et al. 2025) 特有,不在本论文。
- `lr`:论文给的 `3e-5` 是**预训练** lr（§4.1, p11, AdamW β1=0.9/β2=0.999/wd=0.001/2000 warmup);RL 段未单列 lr。当前配置 `actor.optim.lr=3e-5` = 预训练值。把它提到 `1e-4` 是**经验偏离**（兄弟 GRPO 配置注释 + memory 记录 LoRA 想要 1e-4),不是论文背书。
- `ppo_epochs>1`:**与论文相反**,论文是 1 次更新/batch × 256 步。不要优先动它。

## 修复计划（按论文重排优先级）

### P0 — 让学习"可见"（不做这个,下面全是盲调）

训练 `reward_mean` 用轮换 prompt,结构上看不见学习。**用固定 eval prompt 集逐周期打分**:
- 数据已就绪:`configs/dataset/videophy.yaml` 的 `eval_manifest`（35 eval, seed 42),NFT 配置已 import 但训练循环从不读。
- 打分器已存在:`vrl/scripts/eval/cosmos_predict25_kling_eval.py`（跨 checkpoint 同种子,生成+VideoReward 打分)。
- **零代码兜底(先做这个)**:对 `checkpoint-*` 离线跑该打分器对比(独立进程,避开显存叠加)。
- **逐 epoch(增量)**:给 `vrl/scripts/diffusion/cosmos/train.py:40` 的 NFT 入口补 `after_step` 钩子(`online.py:599` 每 epoch 已会调它,cosmos 现传 None),复用打分器,加稀疏 `eval_freq=5`;eval 前必须 `gc+empty_cache` 释放 rollout/Kling,否则 32GB OOM。
- ⚠️ 别信 `online_grpo_kling_*.yaml` 里 `eval_only=true` 的注释——`train.py` 没有该分发,是 stale 注释。

### P0 — 跑到论文的真实步数（我们只跑了 14/256）

论文 reward 是在 **256 步** 上涨起来的。我们崩在第 14 步。**先把这条做满**:
- offline-robust 续跑:`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`(防上次那种 HF 瞬时超时),`trainer.resume_from=outputs/cosmos25_kling_50ep/checkpoint-10`,把 `total_epochs` 拉到 ≥ 数十(单卡按时间预算定;论文 256 步 @ rbs=32,我们 rbs=16 则需要更多步才等价曝光)。
- 保持 streaming(`microbatch_size=1`)、256p/49f(单卡内存上限,见 info sprint 的 OOM ladder)。

> 在 P0 两条没做之前,**不要**下"配方不行"的结论——论文已证明配方能涨,我们只是没跑够、又没在看正确的指标。

### P1 — 仅当固定 eval 在足够步数后仍平,才动梯度（按性价比,全 config-only 零显存）

1. **`advantage_high` 5→8 + `adv_clip_max`≥8**:NFT loss 的全局梯度乘子(`diffusion_nft.py:258`);`adv_clip_max` 在更早 clamp(`advantages.py:35`),不同步抬就白加。经验偏离,论文无此项。
2. **`lr` 3e-5→1e-4**:经验偏离(LoRA);与 #1 同向,先单独试一个再叠加。
3. **`kl_beta`**:论文用它防 reward-hacking,**默认别降**;只有在确认没 hacking、且想要更大步长时,小幅降到 0.5 并盯 `adv_saturation`(`diffusion_nft.py:281`)。
4. **`n_samples_per_prompt`**:论文=8,已是对的,别改(改了还增显存)。

### 不要做（代码/论文已证伪）

- **`ppo_epochs>1`**:与论文相反(论文 1 更新/batch),且和 streaming 硬冲突(`types.py:325` raise),关 streaming 走 full-batch 会把 256 段视频一次性上 GPU(单卡必 OOM)。最后手段,且须把 `rollout_batch_size` 砍到 ~4。
- **全参微调(LoRA off)**:`predict2_5` 的 `enable_full_finetune()` 直接 raise(`model.py:239-243`);无 8-bit/paged 优化器(`trainer.py:51-72` 硬编码 fused AdamW)。单卡不可行,需多卡。
- **`global_std` true/false**:`microbatch_size=1` 下每 microbatch 一组,per-group 与 global std 数值相同(`online.py:127`),对本 run 无影响。
- **加 `init_kl_coef`**:GRPO-only 字段,塞进 NFT 配置会 raise "unknown DiffusionNFTConfig field"(`builders.py:96`)。NFT 的 KL 只有 `kl_beta`。

## 第一个该跑的实验

```bash
cd ~/Desktop/VRL && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  rollout.rollout_batch_size=16 rollout.microbatch_size=1 rollout.n_samples_per_prompt=8 \
  rollout.host_memory_budget_fraction=0.95 \
  sampling.width=256 sampling.height=256 sampling.num_frames=49 sampling.num_steps=20 \
  trainer.total_epochs=60 trainer.save_freq=5 trainer.eval_freq=5 \
  trainer.resume_from=outputs/cosmos25_kling_50ep/checkpoint-10 \
  trainer.output_dir=outputs/cosmos25_kling_50ep production.kling_video_reward.enabled=false
```
先**只**加 P0(固定 eval + 跑更多步),**保留所有论文确认的参数不变**。看固定 eval 曲线;若到几十步仍平,再叠加 P1 的 `advantage_high`/`lr`。

## 成功标准
- **主判据**:固定 35-prompt eval 集(同种子)的 VideoReward 随 step **上升**(论文 Table 6 是 sum 从 1.08→1.69 这种量级的提升)。训练 `reward_mean` 继续抖属正常,别看。
- **佐证**:`grad_norm` 抬升;`adv_saturation` 不接近 1。
- **判法**:block-test、多 prompt×多 sample 求均值,跨 checkpoint 差异 >2σ 才算(别把 ~1σ 抖动当学习——memory 教训)。

## 诚实兜底
单卡 256p/LoRA 是论文(多卡/高分辨)的削配版,即使做满 P0+P1,提升幅度大概率小于论文。真要复现论文量级,需要多卡 + 更高分辨 + 可能全参。本 sprint 的目标是:**在单卡可行范围内,先把"可见 + 跑够步数 + 参数对齐论文"做满,再判断单卡能到多少。**

## 引用

论文（`docs/papers/cosmos_predict2_5_..._2511.00062v2.pdf`）：
- §4.2.2 Reinforcement Learning (p13)：VideoAlign、8 输出、20 步、GRPO 组内归一化、256 步 batch 32、diffusion-loss 正则防 hacking、EMA、1 更新/batch。
- §4.1 (p11)：AdamW β1=0.9/β2=0.999、2B lr 3e-5、wd 0.001、2000 warmup。
- §3.2 (p9) + Table 3：2B 架构;93 帧/24 latent/16fps/~5.8s;WAN2.1 VAE 4×8×8。
- Table 6 / Fig 5：RL 前后 reward 提升（证明配方有效）。

代码（VRL）：
- `vrl/algorithms/diffusion_nft.py`（235-261 loss/乘子/KL）、`advantages.py`（27-35 组内归一+clamp）
- `vrl/trainers/core/types.py`（299-351 streaming/ppo_epochs/host_budget 约束）
- `vrl/trainers/online/trainer.py`（967 ppo_epochs 唯一消费点）、`vrl/scripts/common/online.py`（552-602 epoch loop + after_step 钩子）
- `vrl/scripts/diffusion/cosmos/train.py`（37-49 NFT 入口 = eval 钩子补丁点）、`vrl/scripts/eval/cosmos_predict25_kling_eval.py`（固定-prompt 打分器）
- `vrl/config/builders.py`（96-106 拒绝未知 algorithm 键）、`vrl/models/diffusion/cosmos/predict2_5/model.py`（239-243 全参 raise）
- 相关记录：`info/SPRINT_cosmos25_kling_reward_curve.md`
