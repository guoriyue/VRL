# SPRINT: 模型家族覆盖度路线图（对照 cosmos-rl / VeRL-Omni）

状态：**DONE（2026-07-09 终局对账；原 planned 2026-06-21）**。性质：**竞品模型覆盖对照 +
优先级路线图**（index 文档，不直接落地）。
Tier-1 的两个 T2I 家族已拆为独立落地 sprint：[[SPRINT_flux_t2i]]、[[SPRINT_qwen_image_t2i]]。

> **终局对账（2026-07-09，对 registry 实况）**：本路线图的可落地项已全部落地。
> Tier-1 两家（FLUX LANDED 2026-06-21、Qwen-Image code-landed）之外，
> [[SPRINT_thin_model_seam_and_ten_model_expansion]] Phase 1（2026-07-08，10/10 真权重验证）
> 又接入：SANA / Lumina-Image-2 / PixArt-Σ / HunyuanImage-2.1 / **HunyuanVideo**（即本文 Tier-2 项）/
> Mochi-1 / CogVideoX / Emu3 / LlamaGen / GLM-Image。`vrl/rollouts/families/registry.py` 现有
> **23 个 family**（§1 的 7 行盘点是 2026-06 旧况，作历史记录保留）。剩余 gap 只剩 Tier-3
> 统一理解+生成 / omni（BAGEL、HunyuanImage-3.0、Qwen3-Omni）——按 §5 仍是非目标：需要新
> rollout/logprob seam，各自独立大 sprint。本 index 使命完成，归档 done/。

> 来源：
> - VeRL-Omni 发布博客（vLLM，2026-05-14）+ vllm-omni supported-models 文档；
> - cosmos-rl README / docs（nvidia-cosmos）；
> - 本仓库现有家族盘点（`vrl/models/{ar,diffusion}/`）与接入面盘点
>   （`vrl/rollouts/families/registry.py`、`vrl/models/diffusion/base.py`、`vrl/models/loader.py`）。
> 相关：[[SPRINT_flow_grpo_recipe_parity]]（flow_grpo 是 SD3/**Flux**/Wan 的母体，Flux 配方已被验证能学）、
> 记忆 `project_first_trustworthy_curve`（diffusion GRPO 真能不能学的判据）。

## 0. 一句话

**我们的 diffusion seam 已经能装下竞品几乎所有"纯生成"模型——缺的是家族实现文件，不是架构。**
对照下来，真正值得补、且能直接套现有四文件模式（`model/runner/runtime` + registry 一条目 + yaml）的，
是两个 T2I DiT：**FLUX.1** 和 **Qwen-Image**。这两个是社区当下用得最多、竞品都已支持、而我们没有的。
其余竞品独有模型分两类：(a) 视频扩散（HunyuanVideo 等）——能套 seam 但更重；(b) 统一理解+生成 / omni
（BAGEL、HunyuanImage-3.0、Qwen3-Omni）和 VLA（OpenVLA、PI0.5）——**架构上不属于纯 diffusion 或纯 AR
seam**，是独立大 sprint，不在本 sprint 落地范围，只列入路线图。

## 1. 现状盘点：我们已支持的家族

| family | 变体 | 模态 | 架构 | 训练 | 路径 |
|---|---|---|---|---|---|
| `janus_pro` | Janus-Pro-1B | T2I | AR 离散 token | GRPO(LoRA) | `vrl/models/ar/janus_pro/` |
| `nextstep_1` | NextStep-1.1 14B | T2I | AR 连续+flow head | GRPO(LoRA) | `vrl/models/ar/nextstep_1/` |
| `sd3_5` | SD3.5-Medium | T2I | DiT flow-matching | GRPO(LoRA/FT) | `vrl/models/diffusion/sd3_5/` |
| `wan`/`wan_2_1_i2v` | Wan2.1/2.2 T2V/I2V | T2V/I2V | DiT | GRPO(LoRA) | `vrl/models/diffusion/wan_2_1/` |
| `cosmos-predict2` | Predict2 2B V2W | V2W | DiT | GRPO(FT) | `vrl/models/diffusion/cosmos/predict2/` |
| `cosmos-predict2.5` | Predict2.5 2B | T2W | DiT(DiffusionNFT) | GRPO(LoRA) | `vrl/models/diffusion/cosmos/predict2_5/` |
| `cosmos-predict2-anima` | Anima Preview3 | T2I | DiT | GRPO(LoRA) | `vrl/models/diffusion/cosmos/anima/` |

**结论**：我们在 video / world-model 扩散这一侧覆盖很厚（Wan + 三个 Cosmos），但 **T2I 这一侧只有
SD3.5 + Anima**——恰恰是竞品最卷、社区基线最常用的那两个（FLUX、Qwen-Image）我们都没有。

## 2. 竞品支持、我们没有的（gap 表）

### 2.1 VeRL-Omni（与我们同域：diffusion / omni 生成的 RL 后训练）

| 模型 | 模态/架构 | VeRL-Omni 状态 | 我们 | 套现有 seam？ |
|---|---|---|---|---|
| **Qwen-Image** | T2I / MMDiT ~20B | **已发布**（FlowGRPO/MixGRPO/GRPO-Guard）| ❌ | ✅ 纯 diffusion seam |
| **FLUX.1** | T2I / DiT 12B（rectified flow）| 支持（diffusion 栈）| ❌ | ✅ 纯 diffusion seam |
| SD3.5 | T2I / DiT | WIP(DPO) | ✅ 已有 | — |
| Wan2.2 | T2V / DiT | WIP(DanceGRPO) | ✅ 已有 | — |
| BAGEL | 统一理解+生成 / MoT 7B(14B) | 预发布(FlowGRPO) | ❌ | ⚠️ 新 seam |
| Qwen3-Omni-Thinker | omni（文/图/视频/音频）/ AR | 预发布(GSPO) | ❌ | ⚠️ 新 seam（含音频）|
| HunyuanImage-3.0 | 统一理解+生成 / MoE 大模型 | 规划(MixGRPO/SRPO) | ❌ | ⚠️ 新 seam |

### 2.2 cosmos-rl（NVIDIA，偏 Physical-AI 推理/动作，**与我们不同域**）

| 模型 | 类型 | 我们是否该追 |
|---|---|---|
| OpenVLA / OpenVLA-OFT / PI0.5 | VLA（视觉-语言-动作，具身）| **否**——动作 RL 范式，非生成 RL |
| Cosmos-Reason1 | VLM 推理（Qwen2.5-VL 架构，长 CoT）| **否**——推理 RL，非生成 RL |
| HF LLM/VLM（LLaMA/Qwen）| 文本 LLM RL | 否 |
| 基于 diffusers 的世界基础模型 | 视频扩散 | 已被我们 Wan/Cosmos 覆盖 |

> **诚实结论**：cosmos-rl 的"独有"模型几乎都是 VLA / VLM 推理，那是另一条范式（reward 打在动作/答案上，
> 不是打在生成像素上），我们的 rollout/algorithms 层（flow-matching SDE、`sde_step_with_logprob`）不适配。
> 所以**可落地的 gap 基本由 VeRL-Omni 的生成模型主导**，cosmos-rl 这边只确认了"我们视频侧已对齐"。

## 3. 优先级与理由

**Tier 1（已拆为独立落地 sprint）— T2I DiT，直接套四文件模式：**

1. **FLUX.1 [dev]**（12B，rectified-flow）→ [[SPRINT_flux_t2i]]
   - 为什么：社区 T2I 事实基线；用户点名；**flow_grpo 母体已含 Flux 配方**（见
     [[SPRINT_flow_grpo_recipe_parity]] 标题"SD3/Flux/Wan"）。技术点：guidance-distilled 需单分支 runner。
2. **Qwen-Image**（~20B MMDiT，Apache-2.0，强文字渲染）→ [[SPRINT_qwen_image_t2i]]
   - 为什么：VeRL-Omni 旗舰已发布模型；开源 T2I SOTA 之一；有真 CFG，seam 完全吻合。约束：20B → LoRA-only。

**Tier 2（后续 sprint）— 视频扩散，套 Wan/Cosmos seam：**
- **HunyuanVideo**（T2V DiT）——能套 5D 潜变量 seam，但 VAE/调度器要单独对齐，单列。

**Tier 3（独立大 sprint，本 sprint 不做）— 统一/omni，需新 seam：**
- **BAGEL / HunyuanImage-3.0**（统一理解+生成，AR+diffusion 混合）——既不是纯 diffusion 也不是纯 AR，
  rollout 与 logprob 定义要重做。
- **Qwen3-Omni-Thinker**（含**音频**模态）——超出当前图像/视频 reward 与 tokenizer 范围。

## 4. 落地：接一个 T2I 家族的通用形状（已盘点接入面）

**一个新 T2I 家族 ≈ 1 条 registry（`vrl/rollouts/families/registry.py`，照 sd3_5 第 132-143 行）+ 4 个文件
（`vrl/models/diffusion/<family>/{model,runner,runtime,__init__}.py`，实现 `base.py:41-96` 的抽象方法）+ 1 个 yaml
（照 `sd3_5/medium.yaml`）**，不碰算法/rollout/loader 共享层（`common/*`、`flow_matching.py`、`loader.py` 家族无关）。
这正是 FLUX/Qwen-Image 性价比最高的原因。逐文件清单与各自家族的技术点见落地 sprint：

- **FLUX**：单分支 + guidance runner → [[SPRINT_flux_t2i]]
- **Qwen-Image**：Qwen2.5-VL 文本编码器（无 pooled）、20B LoRA-only → [[SPRINT_qwen_image_t2i]]

## 5. 路线图非目标（index 级）

- **不做 VLA / VLM 推理模型**（cosmos-rl 的 OpenVLA/PI0.5/Cosmos-Reason1）——不同 RL 范式，超出生成 RL 基础设施。
- **不做统一理解+生成 / omni**（BAGEL、HunyuanImage-3.0、Qwen3-Omni）——需新 rollout/logprob seam 与音频模态，
  各自独立 sprint，本 index 只登记不落地。
- Tier-1 各自的家族级非目标（不接 schnell、不做 edit 变体等）见两个落地 sprint。

## 参考

- VeRL-Omni 发布：https://vllm.ai/blog/2026-05-14-verl-omni
- vllm-omni supported models：https://github.com/vllm-project/vllm-omni
- cosmos-rl：https://github.com/nvidia-cosmos/cosmos-rl
- 接入面证据：`vrl/rollouts/families/registry.py:92-143`、`vrl/models/diffusion/base.py:41-96`、
  `vrl/models/loader.py`、`vrl/models/diffusion/sd3_5/{model,runner,runtime}.py`、
  `configs/model/diffusion/sd3_5/medium.yaml`
