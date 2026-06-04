# Sprint: Video RL Fine-Tune (TDM-R1 few-step)

状态:proposed(future-work roadmap,不是当前执行)

## 0. TL;DR / 一句话

把 video diffusion 的 **RL** fine-tune 这条路补上 —— **只做 RL,不做可微 reward fine-tune**。

- **TDM-R1** —— non-differentiable RL on **few-step**(1-4 step)video diffusion。低成本,几乎是把现有 DiffusionNFT recipe push 到 4 step。
- **(可选)本地 VLM video judge** —— 作为**非可微** reward 喂给同一条 RL,丰富 reward 信号,与 Kling reward 复合。不是可微 finetune,只是多一个 score function。

vrl 已经有 ~60% 基础设施(Cosmos V2W model + DiffusionNFT @ 10 step + Kling video reward + LeRobot dataset)。**抓手:Stage A(TDM-R1)几天就能跑,直接看 video 4-step RL 能不能立得住;Stage B(本地 VLM judge)可与 A 并行,纯 reward 增强。**

> 范围说明:可微 reward fine-tune(Diffusion-DRF:VLM critic 梯度反传 through denoise chain)**不在本 sprint**。本 sprint 全程走标准 RL(group-relative advantage / DiffusionNFT),所有 reward 保持 `@torch.no_grad()`。真要做可微 critic 是独立的 research sprint。

## 1. Context / 为什么做

旧的 image RL + Claude/Codex judge 路线已下线。视频侧有两个 image 没有的痛点,逼着要专门设计:

1. **每个 sample 巨贵**:1 个 video rollout = N 帧,GPU + reward 推理时间都贵 N 倍。**few-step**(1-4 step)distillation / RL 是让 video RL 跑得动的前提。
2. **VLM 评 video 比评 image 难**:多帧才能判断 motion,但每帧都过 VLM = 翻倍 cost。需要在 reward 设计上控制帧采样和耗时。

**对应两个抓手**:
- TDM-R1:**cost**(few-step)+ **通用 non-diff reward**(任何 score function 都能上)。
- 本地 VLM judge:更强的 **reward 信号质量**(motion / per-frame issue 结构化打分),仍然走 non-diff RL。

## 2. 已就绪 / What's already wired(高复用,不重建)

| 资源 | 路径 | 备注 |
|---|---|---|
| Cosmos Predict2/2.5 V2W model | `vrl/models/diffusion/cosmos/predict2{,_5}/model.py` | reference_image + init_latents + cond/uncond masks 全有 |
| V2W recipes | `configs/experiment/diffusion/cosmos_predict2{,_5}/online_{grpo,nft}_*.yaml` | 3 个:reference / Kling VideoReward / NFT |
| Kling VideoReward | `vrl/rewards/functions/kling_video_reward.py` + `models/kling_video_reward.py` | Ray actor pool,score_keys: `overall_reward` / `visual_quality` / `motion_quality` / `text_alignment` |
| Algorithms | `vrl/algorithms/diffusion_nft.py` + `algorithms/grpo/continuous.py` | DiffusionNFT 支持 video-only objective(无需 evaluator log-prob) |
| Video batch | `RolloutBatch.videos: [B, C, T, H, W]` (`vrl/rollouts/batch/`) | tensor stacking + 清理已通 |
| Dataset | `datasets/video_world/` + `vrl/scripts/data/video_world.py` | LeRobot v2.1 已跑通 |
| 训练入口 | `vrl/scripts/diffusion/cosmos/train.py:train_cosmos_predict25_diffusion_nft()` | 10-step NFT baseline ready |

## 3. Gap 诊断 / 为什么不开箱即用

### TDM-R1 gap(少,~1 周)
- **没有 few-step preset**——现有最低 10 step (`configs/sampling/denoise/10_step_cfg_4_5.yaml`);TDM-R1 paper 在 1-4 step 上跑。
- **可能崩**:把 step 数直接砍到 4 不蒸馏,video quality 大概率掉。两条路:
  - **a)** 直接 push 到 4 step,看实际崩多少(先跑实验)
  - **b)** 加 LCM / consistency distillation 阶段,先 10 → 4 step 蒸馏再做 RL

### 本地 VLM judge gap(可选,~3-5 天)
- 现在只有远端不可用的 Claude/Codex judge(已下线)和 Kling reward;**没有本地 VLM video judge**。
- 需要新增一个 **non-diff** local judge reward(结构化打分),注册进 reward registry,与 Kling 复合即可。无需改 RL 算法。

## 4. 分阶段方案 / Recommended approach

### Stage A — TDM-R1 lite(~3-5 天,先做这个)
- 加 `configs/sampling/denoise/4_step_cfg_4_5.yaml`(few-step preset)。
- 复制 `online_nft_kling_video_reward.yaml` → `online_nft_kling_video_reward_4step.yaml`,改 `denoise=4_step_cfg_4_5`。
- 跑 short run,看 video quality + reward。崩了再加 LCM distillation 阶段。
- **不写新 algorithm**:DiffusionNFT 已经接 non-diff reward,只是 step 数变。

### Stage B —(可选)本地 VLM video judge as RL reward(~3-5 天,可与 Stage A 并行)
- 加本地 VLM judge,**作为 RL 的非可微 reward**,不要重新引入 Claude/Codex 远端 judge:
  - 输入:N 帧(均匀采样,N=4-8 帧)+ prompt
  - 本地 Qwen2.5-VL / LLaVA-Next / InternVL 输出结构化 `{score, reasons, per_frame_issues}`(`@torch.no_grad()`)
  - 注册为 `local_video_judge` 在 `vrl/rewards/functions/registry.py`
- 复合:`reward.components.kling_video_reward=0.3 + local_video_judge=0.7`
- **不依赖 Stage A**:也可用在 10-step recipe 上。纯 reward 增强,RL 路径不变。

## 5. 关键文件 / Critical files

**复用(不重写)**:
- `vrl/models/diffusion/cosmos/predict2{,_5}/{model,runtime,runner}.py`
- `vrl/algorithms/diffusion_nft.py` + `algorithms/grpo/continuous.py`
- `vrl/rewards/functions/kling_video_reward.py` + `models/kling_video_reward.py`
- `vrl/rollouts/collector/{core,batch_builder,artifacts}.py`
- `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`

**新增 / 改**:
- `configs/sampling/denoise/4_step_cfg_4_5.yaml`(Stage A)
- `configs/experiment/.../online_nft_kling_video_reward_4step.yaml`(Stage A)
- `vrl/rewards/functions/local_video_judge.py`(Stage B,可选)

## 6. 验证矩阵 / Verification

| 阶段 | 验证 |
|---|---|
| A | 4-step run vs 10-step baseline:video quality / motion 不能崩超 X%;`overall_reward` 趋势上升;`metrics.csv` `reward.mean` 非 flat |
| B | Local VLM video judge 在 16 个 V2W sample 上与 Kling reward 排序一致性 ≥ 0.7(Spearman);耗时 < 30s / 16-sample batch |
| 全仓 lint + tests | `ruff check vrl tests` + `pytest tests/rewards/test_video_reward*.py` |

## 7. Open design decisions(等执行时再拍)

- **TDM-R1 step 数**:4 / 2 / 1?paper 在 1 step 上也跑,但需要 consistency 蒸馏。
- **(Stage B)Local VLM 模型**:Qwen2.5-VL 7B(快,中等能力)vs LLaVA-Next 13B(慢,强)vs InternVL3 8B(平衡)。
- **(Stage B)Local VLM memory ceiling**:先用 7B/8B 级别 judge,并用 Ray reward pool 控制显存占用。
- **KL 锚定**:few-step video RL 也容易让 cond 漂(模型忽略 reference_image),要不要加 reference reward(SSIM with `init_latents`)?

## 8. 非目标 / Non-goals

- **不**做可微 reward fine-tune(Diffusion-DRF:VLM critic 梯度反传 through denoise chain)——本 sprint 纯 RL,所有 reward 保持 non-diff。真要可微 critic 是独立 research sprint。
- **不**在本 sprint 实做 LCM / consistency distillation(Stage A 先 naive few-step 试;崩了再独立 sprint 蒸馏)。
- **不**建远端 judge/API 路线;Stage B 走本地 `transformers.AutoModelForVision2Seq` 或同级本地 VLM 加载。
- **不**支持 video → video editing(input video + output edited video)——本 sprint 范围是 I2V(reference image → video)+ T2V fine-tune 的提升。真要 V2V editing 是独立 sprint。
- **不**重写 Cosmos backbone(`predict2{,_5}` model/runtime/runner 已稳定)。

## 9. References

- **TDM-R1**:"Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward" — huggingface.co/papers/2603.07700
- **Awesome-RL-for-Video-Generation**(curated list):github.com/wendell0218/Awesome-RL-for-Video-Generation
- **Cosmos-Predict2.5** RL post-training reference: arxiv 2511.00062
