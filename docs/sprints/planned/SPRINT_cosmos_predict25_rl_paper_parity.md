# SPRINT: Cosmos Predict2.5 RL — paper-faithful run（rollout 预算对齐论文）

状态：proposed / planned（2026-06-17）。**给另一台机器的要求**：复现 Cosmos-Predict2.5
RL 必须用 **paper-shaped 配置**（够显存才跑得起），**不要**沿用 field-notes 那次单卡
L40S 的缩水 override——那次只采了论文 **千分之二** 的 rollout 量，学不出东西。

> 来源：本篇核对了论文 PDF（`docs/papers/world-models/cosmos-predict2-5-world-foundation-models.pdf` 第 13 页
> §4.2.2 Reinforcement Learning）、仓库 config（`configs/experiment/diffusion/cosmos_predict2_5/`）、
> 以及 field-notes（`info/SPRINT_cosmos_training_field_notes.md`）。

## 0. 一句话

field-notes 那次单卡实跑**严重偏离论文**(每次更新只有 3 个 rollout vs 论文 256 个),
**但仓库的 `online_nft_kling_video_reward.yaml` 本身已经对齐论文**。"miss many rollout"
是**运行期被单卡 46GB 显存逼小**的妥协,不是配置 bug。要复现论文,**就跑那个已存在的
config,给够显存**(多卡或更大卡),别用 field-notes 的单卡 override。

## 1. 论文 §4.2.2 真实 RL 配方(已从 PDF 第 13 页核实)

原文要点:

> 每个 input condition 生成 **8 个 output**、每个 output **20 步去噪**,GRPO 组内归一化
> advantage;因显存约束,**每 2 个 conditional probability 算一次梯度、累积成 10 步**;
> **训练 256 步、batch size 32**;用 **fine-tuning 数据的 diffusion loss 做正则**(防 reward
> hacking);释放 **EMA** 权重。算法 **GRPO**,reward 用 **VideoAlign**。

→ 每次参数更新 = **8 × 32 = 256 个 rollout**;全程 256 次更新 = **65,536 个 rollout**。

## 2. 三方对比

| 项 | 论文 §4.2.2 | 仓库 config（`online_nft_kling_video_reward.yaml`） | field-notes 实跑（L40S 46GB） |
|---|---|---|---|
| 每 condition 输出 n | **8** | 8 ✅ | **3** ❌ |
| 每次更新 conditions（batch） | **32** | 32 ✅ | **1** ❌（§6: rbs=1） |
| **每次更新 rollout 数** | **256** | **256** ✅ | **3** ❌ |
| 去噪步数 | **20** | 20 ✅ | **10** ❌ |
| 梯度 timestep 切片 | **10**（每 2 步） | `tf=0.5`→10 ✅ | 未记 |
| 训练更新次数 | **256** | 256 ✅ | **50** ❌ |
| CFG | 未明示（GRPO 用 no-CFG 取干净 log-prob） | **no-CFG** ✅ | **CFG 7.0** ❌ |
| 分辨率/帧 | ≤720p/93f | 512p/93f | 480p/49f |
| 算法 | **GRPO** | DiffusionNFT ❌ | NFT |
| reward | **VideoAlign** | `kling_video_reward`（疑同源,待证） | kling |
| diffusion-loss 正则 | **有** | 没找到 ❓ | ❓ |
| lr（2B） | **3e-5 全参** | 1e-4 LoRA（注释:LoRA 要更高 lr、全参需多卡） | 未记 |

## 3. "miss many rollout" — 量化

- **每次更新**:实跑 3 个 rollout vs 论文 256 个 = **1.2%**。
- **全程**:实跑 50×1×3 = **150 个** vs 论文 256×32×8 = **65,536 个** = **~0.23%**。

实跑只采了论文 **千分之二** 的 rollout 量。GRPO 的学习信号是组内 advantage(`reward_std`),
rbs=1、n=3 时组内只有 3 个样本、每步只有 1 个 prompt——advantage 几乎是噪声。field-notes §6
观察到的 "reward bounced -1.2…-3.6 with no trend" 正是这个欠采样的后果。

## 4. 需要做什么(给另一台机器的 action)

**复现论文 = 跑仓库已对齐的 config,给够显存,不要单卡 override。**

```bash
# paper-shaped 配置已存在,直接用(NFT 变体):
#   configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml
#   → n_samples_per_prompt=8, rollout_batch_size=32, 512p_93f, 20_step_no_cfg,
#     total_epochs=256, timestep_fraction=0.5(→10 切片)
# 关键:这套 n=8/rbs=32/512p/93f 在单 L40S(46GB)塞不下,需要多卡或更大卡。
# 不要把它缩成 field-notes 的 n=3/rbs=1/480p/49f/10-step/CFG。
```

显存够之前,**不要**把 n/rbs/分辨率往下调来"先跑通"——那等于回到 field-notes 的欠采样,
得到的还是噪声曲线。

## 5. 仍未解决的偏差 / 待决策(即便用仓库 config 也离论文有差)

1. **算法**:论文是 **GRPO**,`online_nft_kling_video_reward.yaml` 用的是 **DiffusionNFT**。
   仓库有对应的 GRPO recipe `configs/experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward.yaml`——
   要 paper-faithful 应该用 GRPO 那个,不是 NFT。**决策:跑 NFT 还是 GRPO?**
2. **diffusion-loss 正则**:论文明确用它防 reward hacking;在 NFT recipe 里**没找到**对应实现。
   **行动:确认是否缺,缺则补**(否则容易 reward hack)。
3. **reward 模型**:论文 **VideoAlign** vs config `kling_video_reward`——很可能同源(Kling 的
   VideoReward),但**没 100% 确认**。**行动:确认是不是同一个模型。**
4. **lr / 微调方式**:论文 3e-5 全参,config 是 1e-4 LoRA(注释解释:全参需多卡)。这是**有意的
   资源妥协**,不是错——有多卡时可切回 3e-5 全参。

## 6. 参考

- 论文:`docs/papers/world-models/cosmos-predict2-5-world-foundation-models.pdf` §4.2.2(p13)
- paper-shaped config:`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- GRPO 变体:`configs/experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward.yaml`
- 单卡实跑记录:`docs/sprints/info/SPRINT_cosmos_training_field_notes.md`
- 单卡 runbook:`docs/sprints/info/SPRINT_cosmos25_kling_paper_recipe_runbook.md`
- 固定 eval 信号(判断有没有学到):`docs/sprints/done/SPRINT_cosmos_kling_fixed_eval_signal.md`
