# SPRINT：图像生成模型后训练——大厂 tech report 读书笔记（reading）

状态：**reading / 存档（2026-09-16）**。KIND：**reading**——不是待办清单。每个来源
一节：先记原文说了什么（只记能从原文抽出的、可核对的内容），再记"对 VRL 的含义"。
由这些含义汇总出的实施项在 [[SPRINT_industry_post_training_gap_backlog]]。
视频侧见 [[SPRINT_video_post_training_reports]]，方法论文见
[[SPRINT_diffusion_rl_method_notes]]。

调研方式：网络检索 + 逐篇抓取原文抽取。结论：**"大规模图像后训练工程博客"公开
的几乎没有**，各家把这部分放在 tech report 里（arXiv PDF，但写法是工程报告）。
所有 report 都不给 RLHF 的超参、loss 公式和 reward model 架构细节；能拿到的是
**流程结构、reward 设计思路、数据闭环、稳定性手段** 这四类信息。

---

## 0. 一页总结：各家后训练流水线长什么样

| 家 | 阶段 | RL 目标形式 | Reward 模型 | 稳定性手段 | 数据闭环 |
|---|---|---|---|---|---|
| Seedream 2.0 | CT → SFT → RLHF → PE | **直接最大化多 RM 输出**（REFL 式），试过 DPO/DDPO 认为不如 | 中英双语 CLIP，**CLIP 相似度直接当 reward**，ranking loss 训练；三个 RM：图文对齐 / 美学 / 文字渲染 | lr 调整、**选 denoising timestep**、**权重 EMA**、DiT 与 LLM 文本编码器联合微调 | **迭代**：优化 → 在新模型上标 bad case → 训 bad-case-aware RM → 再优化 |
| Seedream 3.0 | CT → SFT → RLHF → PE（去掉 Refiner） | 同上 | **VLM 做 RM**：指令作 query，reward = 归一化的 "Yes" token 概率；**RM 从 1B scale 到 >20B，观察到 RM scaling** | 同上 | 同上 |
| Seedream 4.0 | CT → SFT → RLHF → PE（T2I + 编辑联合） | 未披露 | 未披露 | 未披露 | 未披露；PE 模块用 Seed1.5-VL 微调，带 auto-thinking / 自适应比例 / 动态思考预算 |
| HunyuanImage 3.0 | SFT → DPO → MixGRPO → SRPO → ReDA | 三种并用：DPO（结构缺陷）、MixGRPO（美学/畸变/伪影）、SRPO（单步、可微 reward、正负文本引导） | "开源 + 自研" | ReDA：让生成分布对齐"高 reward 图像分布" | DPO 数据来自 SFT 模型自采样 + 标注 |
| Qwen-Image | SFT → DPO → Flow-GRPO | DPO（flow-matching 形式，大规模离线）+ Flow-GRPO（小规模在线精修） | 未披露 | KL（SDE 下闭式）、clip | 标注员在多张生成里选最好/最差；有参考图的先和参考图比 |
| Kandinsky 5.0 | SFT（SFT-soup） → RL | "对抗式 RLHF"：把生成图和 SFT 数据集比 | Relative Reward Training（细节少） | 模型汤（等权或按根号比例） | 9 个 VLM 域分类，分域 SFT 再合并 |

三个共同点：
1. **RLHF 前面都有 CT/SFT**，且 SFT 数据是人工筛过的、按域/风格分桶的；多家用**模型汤**合并分域 SFT。
2. **RM 在往 VLM 走**（Seedream 3.0、HunyuanImage 的 SRPO/ReDA），reward 是判别式的 "Yes" 概率或对齐分布，不是 CLIP head。
3. **RM 和 policy 迭代**：没有人一次训完；bad case 回流进 RM 数据是标准做法。

---

## 1. Seedream 2.0（arXiv 2503.07703）

### 1.1 原文要点

**CT（Continuing Training）**：用 IQA 模型筛的高质量预训练数据 + 艺术/摄影/设计的
人工数据（百万级）联合训练；VMix：把色彩/光照/质感/构图的美学标签作为去噪条件。

**SFT**：多轮人工精修 caption，带风格与细粒度美学标签。**引入负样本**：模型生成图
标为 fake 与真图一起训练，让模型能区分真/假。SFT 会**损害图文对齐**，用数据重采样
算法在美学与对齐间平衡。

**RLHF**
- RM：中英双语 CLIP，"forgo additional head output reward methods like ImageReward,
  opting to utilize the output of CLIP model as the reward itself"，主损失 ranking loss。
  三个方向的 RM：图文对齐、美学、文字渲染。
- 数据：100 万条多维 prompt（训练 caption + 用户输入），跨版本、跨模型标注，多维融合
  标注（图文匹配 / 文字渲染 / 美学）。
- 学习算法：**直接优化多个 RM 的输出分数**（REFL 范式）；调研过 DPO 和 DDPO，认为
  直接法更有效。
- 稳定性：仔细调 lr、**选合适的去噪 timestep**、**权重 EMA**、DiT 与 LLM 文本编码器
  联合微调。
- 迭代：(1) 用现有 RM 优化 diffusion → (2) 在优化后的模型上标偏好数据，训
  bad-case-aware RM → (3) 用新 RM 再优化 → 重复。

**PE（Prompt Engineering）**：u→r（人工改写直到出好图）和 r→u（把训练 caption 退化成
用户 prompt）两路造 pair，SFT 一个 LLM；再用 SimPO 做 PE 的 RLHF（每个用户 prompt
生成多个改写，按美学/对齐选图对）。报告收益：美学 +30%、对齐 +5%、多样性上升。

**加速**：Hyper-SD 的 TSCD（k=16→8→4→2→1 分段一致性蒸馏）；guidance scale embedding
省掉 CFG 的两次前向；算子融合 5–20%；自适应混合量化。

**评测**：Bench-240（240 条双语 prompt）；专家 Likert 1–5（对齐/结构/美学）；
ELO（>50 万次对比，每模型 ~3 万）；自动指标 EvalMuse / VQAScore / HPSv2 / MPS；
文字渲染 availability rate、accuracy rate、hit rate。

### 1.2 对 VRL 的含义
- **直接 reward 反传**是 ByteDance 图像/视频线的主力目标形式，VRL 没有这条路径
  （只有 GRPO 族、NFT、DPO）。→ backlog C。
- 稳定性三件套：timestep 选择（VRL 有 `timestep_selection`）、EMA（VRL 有
  `trainers/online/ema.py`）、文本编码器联合微调（VRL 冻结，暂不做）。
- bad-case 回流：VRL 没有"从训练中导出低分/高增益样本供标注"的机制。→ backlog G。

---

## 2. Seedream 3.0（arXiv 2504.11346）

### 2.1 原文要点
- 四阶段 CT → SFT → RLHF → PE，去掉 2.0 的 Refiner（直接出 512²–2048²）。
- 多个 caption 模型变体服务 CT/SFT（美学、风格、版式专业描述）。
- **RM 改成 VLM**："formulates instructions as queries and derives rewards from the
  normalized probability of the 'Yes' response token"。
- **RM scaling**："We systematically scale the reward model from 1B to >20B parameters.
  Empirical results reveal the emergence of reward model scaling."
- SFT 做分辨率平衡采样。
- 加速：consistent noise expectation（预训练模型估的统一噪声期望向量做全局参考）；
  importance-aware timestep sampling（SSD + 学一个数据相关的 timestep 分布）；4–8× 提速，
  1K 图 3.0s。
- 评测：EvalMuse 0.694、HPSv2 0.3011、MPS 13.93；Bench-377；Artificial Analysis ELO 1158。

### 2.2 对 VRL 的含义
- VRL 已有 P(Yes) 形式的 reward（`rewards/models/videocon_physics.py`）和生成式
  Qwen-VL judge（`rewards/models/qwen_vl_judge.py`），但没有**通用的** "rubric →
  P(Yes)" 图像 RM。→ backlog F。
- RM scaling 意味着 RM 会大到必须分卡；VRL reward service 的远程路径 + `device_map`
  已能承载 20B 级 judge（2026-07 已 ship），这条不缺。

---

## 3. Seedream 4.0（arXiv 2509.20427）

### 3.1 原文要点
- CT → SFT → RLHF → PE，T2I 与图像编辑**联合**后训练（DiT 内 causal diffusion）。
- 编辑数据：参考图 + 目标图 + 指令 + 两边 caption（三档细节做增强），术语一致。
- PE：Seed1.5-VL 微调，任务路由 + auto-thinking 改写 + 自适应宽高比 + 动态思考预算
  （AdaCoT）。
- RLHF 细节**未披露**。
- 加速：ADP（对抗蒸馏后训练，混合判别器）+ ADM（对抗分布匹配，可学习的
  diffusion 判别器）；4/8 bit 混合量化；PE 的投机解码（Hyper-Bagel）；2K 图 1.4s。
- 评测：MagicBench 4.0（325 T2I + 300 单图编辑 + 100 多图编辑）；DreamEval（4 场景、
  128 子任务、1600 prompt，VQA 式打分，三档难度）。**发现**：Seedream 4.0 单次均分低但
  **best-of-4 最强**，"greater variability"。

### 3.2 对 VRL 的含义
- **best-of-k 评测口径**：VRL 的 held-out eval 只报均值；应同时报 best-of-k 和方差，
  否则会把"多样性高"误判成"平均差"。→ backlog K。
- 编辑任务的 RL 不在 VRL 当前范围。

---

## 4. HunyuanImage 3.0（arXiv 2509.23951）

### 4.1 原文要点
- 后训练五段：**SFT → DPO → MixGRPO → SRPO → ReDA**。
- SFT：风景/人像/动物/OCR 等分类高质量图，多阶段逐步提高质量门槛。
- DPO：SFT 模型自采样 → 标注高/低质量对 → 抑制结构畸变。
- MixGRPO：滑动窗口 SDE + 其余 ODE；目标是美学（风格/构图/光照）、畸变、伪影；
  "refine advantage estimation to accelerate convergence"，声称可大规模训练。
- SRPO：噪声先验注入 latent → **单步**去噪到干净图 → 可微 reward；选早期去噪区间；
  reward 带正负文本引导。
- ReDA：最小化生成分布与"高 reward 分布"（多流派高质量图集合）的散度。
- RM："both open-source and proprietary"，无细节。
- 推理 CoT：T2T 与 T2TI 推理数据，思考 token 也走 next-token 预测。
- 评测：SSAE（500 prompt → 3500 语义要点，12 字段，MLLM 0/1 匹配）；GSB（1000 prompt，
  100+ 专业评审）。

### 4.2 对 VRL 的含义
- 这是**唯一把三种目标串起来用**的 report：DPO 修结构 → GRPO 提美学 → 单步可微
  reward 精修。VRL 有 DPO（offline）和 MixGRPO 形态（`sde_window`），缺第三段。
  → backlog C（Direct-Align 形式）。
- SRPO 的正负文本 reward 差（见方法笔记 §4）是对 CLIP 系 RM 的零成本改造。
  → backlog C 的子项。

---

## 5. Qwen-Image（arXiv 2508.02324）

### 5.1 原文要点
- SFT：按语义类别分层组织，人工标注；数据要求清晰、细节丰富、明亮、写实。
- **DPO（flow-matching 形式）**：
  `L_DPO = -E[log σ(-β(Diff_policy − Diff_ref))]`，Diff 是 policy/ref 在 chosen 与
  rejected 上的速度预测平方误差之差。数据：每 prompt 多张生成里选最好/最差；有参考
  图的先和参考图比，差得多的直接当 rejected。定位："flow-matching (one step) online
  preference modeling"，效率高，**用于大规模离线偏好学习**。
- **Flow-GRPO**：`A_i = (R − mean)/std`；`L = E[Σ min(r A, clip(r,1±ε) A) − β KL]`，
  KL 在 SDE 下闭式；SDE：`dx_t = [v_t + σ_t²/(2t)(x_t + (1−t)v_t)]dt + σ_t dw`。
  定位："small fine-grained RL refinement" 在 DPO 之后。
- 基建：producer-consumer（VAE 编码与 IO 在 producer，GPU 只训练）；Megatron 4-way TP
  + head-wise 并行；bf16 all-gather / fp32 reduce-scatter；**关闭 activation
  checkpointing**（3.75× 慢）。
- 无 β、G、RM 架构。

### 5.2 对 VRL 的含义
- 序列 **大规模 DPO → 小规模 GRPO** 与 HunyuanImage 一致。VRL 两者都有，但没有"先
  offline DPO 出 checkpoint，再 online GRPO"的 recipe 串联。→ backlog H（recipe 级）。
- Flow-GRPO 公式与 VRL `GRPO` 一致，无缺口。

---

## 6. Kandinsky 5.0（arXiv 2511.14993）图像部分

### 6.1 原文要点
- SFT 数据：v1 严格 4.5 万张 / v2 放宽 15.3 万张，有摄影背景的专家评审；batch 64
  （预训练 4096），lr 1e-5（预训练 1e-4）；Qwen2.5-VL-32B + SkyCaptioner 多长度 caption。
- 9 个 VLM 域（Qwen2.5-VL-32B 分类），图像域下 2–9 个子域；**SFT-soup**：分域训练后
  权重平均，等权或按根号比例最好。
- RL："RLHF post-training adversarial method based on comparing generated images with
  those from the SFT dataset"，Relative Reward Training，细节不在可见章节。
- 评测：FVD / VBench / CLIP-score / 专家 SBS 问卷（构图、光照、曝光、伪影、艺术性）。

### 6.2 对 VRL 的含义
- SFT-soup 是纯离线工具，VRL 没有 SFT trainer 也没有权重平均脚本。→ backlog H。

---

## 7. 没找到的

Midjourney、Ideogram、Recraft、BFL 均无后训练技术博客；BFL FLUX.2 博客为产品页。
