# SPRINT: MoE 支持决策（cosmos-rl MoE / 图像模型是否需要 MoE / Wan 2.2 双专家）

状态：reading / 方向讨论（**not an action item**）。回答"要不要给 visual-RL infra 加 MoE 支持"，
给出研究结论 + 决策框架 + 条件化建议。本地证据用 file:line，外部事实用 URL，均在 §6 标注。

> 方法：本地读 cosmos-rl 源码笔记 + grep 现有模型族；外部用 3 个并行 web agent 核验
> （cosmos-rl MoE 现状 / 图像生成 MoE 全景 / Wan2.2 机制），再用 2 个对抗性 agent 试图
> 反驳两个关键论断——**均 confirmed**（见 §7）。

---

## 0. 核心结论 (TL;DR)

1. **cosmos-rl 支持 MoE，但只给 LLM/VLM policy trunk 用**（Qwen3-MoE / Qwen3-VL-MoE /
   DeepSeek-V3），有完整 expert parallelism。它是 **torch-native**（FSDP2+DTensor），**不是
   Megatron 引擎**——只从 Megatron 借了 MoE 的 **compute/dispatch kernel**。Cosmos 的扩散世界模型
   （Predict2/2.5 DiT）**本身是 dense，不用 MoE**。
2. **图像模型不需要 MoE**。所有生产级图像模型（SD3.5 / FLUX.1 / FLUX.2 / Janus-Pro /
   NextStep）都是 dense；图像 MoE（DiT-MoE / EC-DiT）只停留在研究，未进产品、非必需。
3. **本 infra 里唯一真实的 MoE 驱动 = Wan 2.2**，而且是**便宜的那种**（时间步双专家，不是
   token 路由）。它的拦路虎不是 MoE 系统工程，而是 **dual-stage 去噪的 replay/logprob 契约**
   ——你自己的代码已经把范围圈好了。

**一句话**：要"支持 MoE"先问是哪种——Wan 2.2 双专家是个有界的 rollout 接线活，值得；
token 路由 LLM-MoE 是大工程且当前无任何图像模型需要它，先别做。

---

## 1. 问题 A：cosmos-rl 支持 MoE 吗？

**支持，且是完整的 LLM/VLM MoE**，但三个限定直接修正"cosmos 用 megatron 的 moe"这个前提：

- **MoE 模型族**：`cosmos_rl/policy/model/{qwen3_moe, qwen3_vl_moe, deepseek_v3}`，配
  `_ExpertParallel(ParallelStyle)`（按 expert 维 dim-0 shard）、token dispatcher
  （`MoEFlexTokenDispatcher` / `MoEAlltoAllTokenDispatcher`）、grouped-GEMM、dual-mesh FSDP
  （`fsdp_no_moe` 管 attention/dense，`fsdp_moe` 单管 experts）。
- **不是 Megatron 引擎**：parallelism 是 torchtitan 风格（DTensor TP + FSDP2 `fully_shard`）。
  从 Megatron 只**借 kernel**——vendored 在 `cosmos_rl/policy/kernel/megatron_moe/`，其 README 原话：
  *"forked from megatron/core/transformer/moe with minor modifications to remove dependencies on
  megatron core"*。借的是 token dispatch / 路由 kernel，不是框架。
- **扩散世界模型是 dense**：`wfm/networks/minimal_v4_dit.py` 的 FFN 是普通 `GPT2FeedForward`
  （Linear→GELU→Linear），无 router/gate。Predict2.5（2B/14B）公开文档也只描述 dense flow-DiT。

> 你的本地笔记早已记准这点：`reading/cosmos-rl.md:699`（"no Megatron-LM engine … only borrowed
> MoE kernels"）、`:717`（qwen3_moe EP + dual-mesh FSDP）、`:1231-1232`。
>
> 备注：cosmos-rl 仓库已进入**有限维护**，开发转向 Cosmos 3。

## 2. 问题 B：图像模型需要 MoE 吗？—— 不需要

**所有生产级图像模型都是 dense**，恰好是本 infra 已覆盖的族：

| 类别 | 模型 | 架构 |
|---|---|---|
| 扩散 | SD3.5 Large（8B）、FLUX.1（12B）、FLUX.2（~32B） | dense MMDiT（FLUX.2 HF 明说用 parameter sharing，**非** MoE） |
| 自回归 | Janus-Pro、NextStep-1（14B）、Emu3、Chameleon、Lumina-mGPT | dense LLM trunk |

**图像 MoE 只在研究层面**：DiT-MoE（16B，token 路由）、EC-DiT（Apple，97B）、Switch-DiT——
未进产品；eDiff-I / RAPHAEL / ERNIE-ViLG2.0 用的是按 timestep/stage 的 expert ensemble（与 token
路由是两回事），也属研究/旧作。**纯图像生成不需要 MoE。**

## 3. 关键区分：你说的「MoE」有两种，成本差一个数量级

| | **token 路由 MoE**（LLM 式） | **时间步路由双专家**（Wan 2.2 式） |
|---|---|---|
| 机制 | 每 token 过 gate，top-k 选 N 专家 | 按去噪 SNR 阈值 `t_moe`（≈half of SNR_min）切：high-noise expert → low-noise expert，**整个 transformer 二选一** |
| 需要 | expert parallelism、all-to-all dispatch、load-balancing aux loss、grouped-GEMM kernel | **无** gate、**无** EP、**无** Megatron kernel —— 加载 2 个 checkpoint，按 sigma/timestep 派发 |
| 激活 | top-k experts / token | 任一时刻只 1 个 ~14B transformer 活跃（27B 总 / 14B 激活），FLOPs≈dense 14B |
| 谁用 | Qwen3-MoE / DeepSeek-V3（cosmos-rl 的 LLM trunk） | **Wan 2.2-T2V-A14B / I2V-A14B**（Wan2.2-TI2V-5B 是单塔 dense） |
| 在本 infra 的代价 | **大**（FSDP2+diffusers+LoRA 栈无 EP 机制） | **小**（diffusion runtime 的 rollout 改动） |

> 其它开源视频模型（HunyuanVideo、HunyuanVideo-1.5、CogVideoX、Mochi-1、Step-Video）**均 dense**。
> CogVideoX 的 "Expert AdaLN" 是按模态（视觉/文本）的自适应 LayerNorm，**不是**路由 MoE。

## 4. 对本 infra 的含义

唯一真实的 MoE 驱动 = **Wan 2.2**，而且是便宜的双专家型。你的代码已经摸清并圈好范围：

- `vrl/scripts/eval/wan_i2v_base_sample.py:14`：已识别 A14B = high-noise `transformer` +
  low-noise `transformer_2` 双塔；`:102`：已处理 `guidance_scale_2` / `boundary_ratio`。
- **base 推理已能跑**（带 offload，~28GB peak / CUDA OOM 自动 fallback sequential）。
- **训练/rollout 还没接**：
  - `vrl/models/diffusion/wan_2_1/runner.py:82-86`：*"currently only supports Wan 2.1 …
    Wan 2.2 … intentionally left out as a future 2.2 upgrade path"*。
  - `vrl/models/diffusion/wan_2_1/model.py:699-702`：`boundary_ratio is not None` 时直接拒绝，
    注明 *"Wan 2.2 dual-stage I2V needs a separate replay contract"*。

**拦路虎是 dual-stage 去噪的 replay/logprob 契约**（跨 `boundary_ratio` 哪步用了哪个专家、
两段的 logprob 怎么记账），是 rollout/trajectory 接线活，**不是** expert-parallelism。

## 5. 建议（按意图分两路）

- **若「支持 MoE」= 想跑 Wan 2.2** → **值得做，范围有界**：补上 `model.py:702` 说的
  "separate replay contract"（dual-stage 去噪轨迹 + 双专家 logprob 记账），dispatch 按 sigma。
  不碰 Megatron、不碰 EP。建议：等确定要 Wan 2.2 质量时，作为一个独立 diffusion-rollout sprint 做。
- **若「支持 MoE」= 想支持 token 路由 LLM-MoE backbone**（像 cosmos-rl 那样）→ **先别做**。
  没有任何图像模型需要它；在 FSDP2+LoRA 栈上加 EP/dispatch 是大工程。你自己的
  `SPRINT_multi_gpu_training.md:20` 已下结论："megatron 不进 schema，只有大 MoE 才值得"。
  等真有一个 MoE-LLM-backed AR 图像模型落到 roadmap，再做不迟。

**非目标**：不为"将来可能用"而预先搭 expert-parallelism / Megatron MoE 接入——当前零需求。

---

## 6. 来源标注

**本地证据（本仓 file:line）：**
- `docs/sprints/reading/cosmos-rl.md:699`（torch-native，仅借 MoE kernel）、`:717`（qwen3_moe EP +
  dual-mesh FSDP）、`:1231-1232`（EP ParallelStyle；唯一 megatron import）、`:331`（MoE EP mesh）
- `vrl/models/diffusion/wan_2_1/runner.py:82-86`（2.2 是 future upgrade path）
- `vrl/models/diffusion/wan_2_1/model.py:699-702`（2.2 dual-stage 需 separate replay contract）
- `vrl/scripts/eval/wan_i2v_base_sample.py:14`（A14B = 双 ~14B transformer MoE）、`:102`
  （`boundary_ratio` / `guidance_scale_2` 处理）
- `vrl/models/diffusion/wan_2_1/runtime.py:53`（`_MODEL_BY_TASK` t2v/i2v 派发）
- `docs/sprints/parked/SPRINT_multi_gpu_training.md:11,20,390`（megatron 边界 / 只借一个 MoE kernel）

**外部来源（URL）：**
- cosmos-rl 仓库：<https://github.com/nvidia-cosmos/cosmos-rl>
  - 模型族：<https://github.com/nvidia-cosmos/cosmos-rl/tree/main/cosmos_rl/policy/model>
  - qwen3_moe EP：<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/cosmos_rl/policy/model/qwen3_moe/parallelize.py>
  - megatron_moe kernel（README "forked from megatron/core/transformer/moe"）：<https://github.com/nvidia-cosmos/cosmos-rl/tree/main/cosmos_rl/policy/kernel/megatron_moe>
  - MoE 模块：<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/cosmos_rl/policy/kernel/moe/moe.py>
  - 世界模型 dense DiT：<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/cosmos_rl/policy/model/wfm/networks/minimal_v4_dit.py>
- Cosmos Predict2.5（dense flow-DiT，2B/14B）：<https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5>
- FLUX.2（dense MM-DiT，非 MoE）：<https://huggingface.co/blog/flux-2>
- Wan 2.2 双专家（high/low-noise，SNR `t_moe` 路由，27B/14B）：HF model card
  `Wan-AI/Wan2.2-T2V-A14B` / `Wan-AI/Wan2.2-I2V-A14B` + Wan2.2 GitHub README
  （注：官方有 arXiv 技术报告，但本轮以 model card / README 为一手来源，未抓取 arXiv 编号）
- 图像 MoE 研究：DiT-MoE <https://arxiv.org/abs/2407.11633>；RAPHAEL（space/time-MoE）
  <https://arxiv.org/abs/2305.18295>；EC-DiT（Apple，研究）

## 7. 验证记录

- 2 个对抗性 agent 分别试图反驳两个关键论断，均 **confirmed**：
  1. "Wan 2.2 A14B 是时间步路由双专家、非 token 路由，故支持便宜、无需 EP/Megatron kernel" → confirmed。
  2. "无生产级纯图像模型需要 MoE；图像 MoE 属研究方向" → confirmed（对照本仓 sd3_5/wan/cosmos +
     janus_pro/nextstep_1 全 dense）。
- 已知不确定项：cosmos-rl 各模型族端到端可训性未逐一核（深读 qwen3_moe）；Wan 2.2 未抓 arXiv 原文
  （以 model card 为准）；Mochi-1/Step-Video 的 dense 结论来自二手综述。
