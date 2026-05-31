# Sprint: Video → Video Fine-Tune (Diffusion-DRF + TDM-R1)

状态:proposed(future-work roadmap,不是当前执行)

## 0. TL;DR / 一句话

把 video diffusion 的 RL/可微 fine-tune 这条路补上。**两篇 2026 论文各管一半**:

- **TDM-R1** —— non-differentiable RL on **few-step**(1-4 step)video diffusion。低成本,几乎是把现有 DiffusionNFT recipe push 到 4 step。
- **Diffusion-DRF** —— **VLM-as-differentiable-judge**,gradient backprop through denoise chain。高成本,需要扩 evaluator/reward 协议支持 grad-enabled path。

vrl 已经有 60% 基础设施(Cosmos V2W model + DiffusionNFT @ 10 step + Kling video reward + LeRobot dataset)。**抓手:Stage A(TDM-R1)几天就能跑,直接看 video 4-step RL 能不能立得住;Stage C(DRF)2-3 周 research-grade,真有差距再上。**

## 1. Context / 为什么做

`anima_anatomy` sprint 已经把 image RL + Claude judge 这条路捋清了。视频侧有两个 image 没有的痛点,逼着要专门设计:

1. **每个 sample 巨贵**:1 个 video rollout = N 帧,GPU + reward 推理时间都贵 N 倍。**few-step**(1-4 step)distillation / RL 是让 video RL 跑得动的前提。
2. **VLM 评 video 比评 image 难**:多帧才能判断 motion,但每帧都过 VLM = 翻倍 cost。还涉及 differentiability 选择。

**两篇论文各解一面**:
- TDM-R1:**cost**(few-step)+ **通用 non-diff reward**(任何 score function 都能上)。
- Diffusion-DRF:**sample efficiency**(differentiable VLM grad 直进 denoise,比 RL group-relative advantage 信号强一个量级)。

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

### Diffusion-DRF gap(多,~2-3 周)
- **所有 reward 都 `@torch.no_grad()`** —— VLM 梯度进不来,首先要 lifting。
- **Evaluator protocol 只支持 log-prob** —— 要扩 `vrl/rollouts/evaluators/base.py:Evaluator` 支持 `grad_enabled=True` 路径。
- **Reward artifact 评分后被释放** —— `vrl/rollouts/collector/artifacts.py` 会 `release_reward_artifact_if_needed`。DRF 要把 video tensor 留在 graph 里给 VLM 反传。
- **VLM 选择**:Claude(远端 API)**不可微**,跑不了 DRF。需要**本地可微 VLM**(Qwen2.5-VL 7B / LLaVA-Next 13B / InternVL3 8B),才能跑梯度。

## 4. 分阶段方案 / Recommended approach

### Stage A — TDM-R1 lite(~3-5 天,先做这个)
- 加 `configs/sampling/denoise/4_step_cfg_4_5.yaml`(few-step preset)。
- 复制 `online_nft_kling_video_reward.yaml` → `online_nft_kling_video_reward_4step.yaml`,改 `denoise=4_step_cfg_4_5`。
- 跑 short run,看 video quality + reward。崩了再加 LCM distillation 阶段。
- **不写新 algorithm**:DiffusionNFT 已经接 non-diff reward,只是 step 数变。

### Stage B — Claude-judge video reward(~3-5 天,可与 Stage A 并行)
- 仿 `anima_anatomy` sprint 的 `claude_anatomy_judge`,加 `claude_video_judge`:
  - 输入:N 帧(均匀采样,N=4-8 帧)+ prompt
  - 调 Claude vision API 一次性看全部帧(Claude 支持 image-list 输入),结构化输出 `{score, reasons, per_frame_issues}`
  - 注册为 `claude_video_judge` 在 `vrl/rewards/functions/registry.py`
- 复合:`reward.components.kling_video_reward=0.3 + claude_video_judge=0.7`
- **不依赖 Stage A**:也可用在 10-step recipe 上。

### Stage C — Diffusion-DRF(~2-3 周,research-grade)
1. **扩 Evaluator protocol**:`vrl/rollouts/evaluators/base.py` 加 `score_with_grad(rollout, model, vlm) -> Tensor`,启用 `torch.enable_grad()`。
2. **本地可微 VLM wrapper**:加 `vrl/rewards/models/vlm_judge.py`,wrap Qwen2.5-VL 7B / LLaVA-Next。`score_request` 返回带 grad 的 tensor。
3. **Reward artifact lifecycle 扩展**:`collector/artifacts.py` 加 `RewardArtifactPolicy(keep_for_grad=True)`,video tensor 留在 training_view。
4. **新 algorithm**:`vrl/algorithms/diffusion_drf.py`,VLM critic gradient 反传 through denoise chain。
5. **VRAM 概算先算账**:7B VLM + Cosmos transformer + activation cache → 单 H100/A100 能不能 fit。**先算 vram 再写代码**。

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
- `vrl/rewards/functions/claude_video_judge.py`(Stage B)
- `vrl/rewards/models/vlm_judge.py`(Stage C)
- `vrl/algorithms/diffusion_drf.py`(Stage C)
- `vrl/rollouts/evaluators/base.py`(Stage C — 扩 grad path)
- `vrl/rollouts/collector/artifacts.py`(Stage C — keep_for_grad policy)

## 6. 验证矩阵 / Verification

| 阶段 | 验证 |
|---|---|
| A | 4-step run vs 10-step baseline:video quality / motion 不能崩超 X%;`overall_reward` 趋势上升;`metrics.csv` `reward.mean` 非 flat |
| B | Claude video judge 在 16 个 V2W sample 上与 Kling reward 排序一致性 ≥ 0.7(Spearman);耗时 < 30s / 16-sample batch |
| C | Diffusion-DRF on toy task 收敛比同 step budget 的 GRPO 快 ≥ 2x(sample-efficiency benchmark);VLM 梯度 norm 健康(不 NaN / 不饱和) |
| 全仓 lint + tests | `ruff check vrl tests` + `pytest tests/rewards/test_video_reward*.py` |

## 7. Open design decisions(等执行时再拍)

- **TDM-R1 step 数**:4 / 2 / 1?paper 在 1 step 上也跑,但需要 consistency 蒸馏。
- **DRF VLM 模型**:Qwen2.5-VL 7B(快,中等能力)vs LLaVA-Next 13B(慢,强)vs InternVL3 8B(平衡)。
- **Claude judge cost ceiling**:per-frame-batch ≈ $0.01–0.02,1k step × 8 sample ≈ $80–160 / run。能接受。
- **KL 锚定**:DRF 在 video 上特别容易让 cond 漂(模型忽略 reference_image),要不要加 reference reward(SSIM with `init_latents`)?

## 8. 非目标 / Non-goals

- **不**在本 sprint 实做 LCM / consistency distillation(Stage A 先 naive few-step 试;崩了再独立 sprint 蒸馏)。
- **不**建 VLM serving 基础设施(Stage B 用 anthropic API;Stage C 用本地 `transformers.AutoModelForVision2Seq` 加载即可)。
- **不**支持 video → video editing(input video + output edited video)——本 sprint 范围是 I2V(reference image → video)+ T2V fine-tune 的提升。真要 V2V editing 是独立 sprint。
- **不**重写 Cosmos backbone(`predict2{,_5}` model/runtime/runner 已稳定)。

## 9. References

- **Diffusion-DRF**(Jan 2026):"Diffusion-DRF: Differentiable Reward Flow for Video Diffusion Fine-Tuning" — arxiv 2601.04153
- **TDM-R1**:"Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward" — huggingface.co/papers/2603.07700
- **Awesome-RL-for-Video-Generation**(curated list):github.com/wendell0218/Awesome-RL-for-Video-Generation
- **Cosmos-Predict2.5** RL post-training reference: arxiv 2511.00062
