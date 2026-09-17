# SPRINT：视频生成模型后训练——大厂 tech report / 访谈读书笔记（reading）

状态：**reading / 存档（2026-09-16）**。KIND：**reading**。同
[[SPRINT_image_post_training_reports]] 的写法：每源一节，先原文再含义；实施项汇总在
[[SPRINT_industry_post_training_gap_backlog]]。

结论先行：视频侧公开材料里**只有 Seedance 1.0 把 RLHF 写具体**。Movie Gen 只有 SFT，
Wan 报告几乎没有后训练内容，Kandinsky 5.0 视频段主要讲蒸馏，Kling/Runway/Luma 没有
公开材料。唯一的一线工程视角是 Latent Space 对 xAI Grok Imagine 前负责人的访谈。

---

## 0. 一页总结

| 家 | 后训练结构 | RL | Reward | 关键做法 |
|---|---|---|---|---|
| Seedance 1.0 | SFT（分桶训练 + **模型合并**）→ 视频 RLHF → 超分 refiner RLHF → 蒸馏 | **直接最大化多 RM 的 composite reward**，比较过 DPO/PPO/GRPO 认为直接法最有效 | 三个 RM：基础（VLM，图文对齐 + 结构稳定）/ 运动（伪影、幅度、生动）/ 美学（关键帧图像空间） | 多轮 policy↔RM 迭代；标注"best 不劣于 worst"约束 |
| Movie Gen | 预训练 → SFT（四段数据清洗）→ 个性化 / 编辑后训练；**多次 SFT 模型平均** | 无 | 无 | 600 动词分类 k-NN 概念平衡；人工电影级筛选；512 H100 SFT |
| Wan | 后训练同预训练配置，480p/720p，数据按 12 类平衡 | 未描述 | 未描述 | prompt 改写（Qwen2.5-Plus）是推理侧 |
| Kandinsky 5.0（视频） | SFT（2.8k–12.5k 场景，Q-Align>4、美学>2）→ 蒸馏 | 未详 | 未详 | CFG 蒸馏 + TSCD + 对抗后训练，NFE 100→16 |
| Grok Imagine（访谈） | 图像模型先行 → 视频；后训练做 reference-to-video、extension | 未细说 | 未细说 | 全合成 caption；prompt 改写用更大的 LLM；"视觉智能大多来自语言" |

---

## 1. Seedance 1.0（arXiv 2506.09113）

### 1.1 原文要点

**SFT**
- 人工核验 caption 的高质量视频文本对；按视觉风格、运动类型等定义**几百个类别**，
  每类定向采集。
- **不训单模型**：在为风格/运动/场景设计的子集上分别训练，再**合并**；每个子模型
  在有效点 early stopping 防过拟合。

**RLHF**
- 三个 RM：
  1. 基础 RM：VLM 架构，图文对齐与结构稳定；
  2. 运动 RM：视频伪影 + 运动幅度/生动性；
  3. 美学 RM：从视频关键帧、图像空间输入。
- 反馈数据：多源素材扩域；多维标注规则——"在某一维度下选 best 和 worst，同时保证
  best 在其他维度不劣于 worst"；prompt 来自训练集与线上用户，做平衡与信息过滤。
- 算法：**直接最大化 composite reward**——"directly predict x₀ (generated clean video)
  when the RM adequately assesses video quality"。对比结论：比 DPO/PPO/GRPO
  "most efficient and effective"，同时提升文视对齐、运动质量、美学。
- **多轮迭代**："multi-round iterative learning between the diffusion model and RMs...
  more stable and controllable than dynamic update of the RM"。
- 超分 refiner 单独 RLHF：低 NFE 下直接最大化多 RM 的线性组合。

**蒸馏与加速**：TSCD（4×）+ RayFlow score distillation + 多步对抗训练（带人类偏好
数据）；thin VAE decoder 2×；端到端 >10×，1080p 5s 视频 41.4s。

**未披露**：样本数、迭代数、任何超参。

### 1.2 对 VRL 的含义
- 与 Seedream 一样，**直接 reward 反传 + x₀ 预测**是主力。视频上这意味着 reward 走
  VAE 解码后的帧，梯度经解码器回传——显存是主要工程问题。→ backlog C（视频子项）。
- 三 RM 分工在 VRL 已有对应件：`kling_video_reward`（多维）、`motion_dynamics`、
  `videoscore2`、`unified_reward_video`、`aesthetic`（关键帧）。缺的是**多 RM 的
  advantage 级合成**（现在是分数加权和）。→ backlog A。
- "best 不劣于 worst"是 RM 训练数据规则，属于 RM 侧，VRL 不训 RM。
- 分桶 SFT + 合并 → backlog H。

---

## 2. Movie Gen（arXiv 2410.13720）

### 2.1 原文要点
- SFT 数据四段：自动过滤（美学、运动量、场景切换、小主体）→ 概念平衡（600 个人类
  动词/表情分类，视频文本联合 embedding 做 k-NN）→ 人工电影级筛选（自然/棚拍光、
  不过饱和、无杂乱、非平凡运动、无手抖、无后期特效）→ 人工精修 caption（相机控制、
  表情、主体/背景、运动、光照；额外标 6 类相机运动）。
- 时长 10.6–16s，一半 16s；16 FPS（16s）或 24 FPS。
- SFT 用预训练架构 + 预训练 checkpoint 初始化，小 batch，64 节点 512 H100，cosine lr。
- **模型平均**："different sets of finetune data, hyperparameters as well as pre-train
  checkpoints significantly affect key aspects"，多次 SFT 平均成最终模型。
- 个性化：人像视频子集自动构造 (图, 文) → 视频 pair。
- 编辑：三段（单帧编辑 → 多帧编辑 → 回译），**无监督视频编辑数据**。
- Movie Gen Video Bench：1003 prompt，按运动量分级；人工 A/B 三轴（对齐、视觉质量、
  真实感/美学）。
- **无偏好学习或 RL**。

### 2.2 对 VRL 的含义
- 只有 SFT，对 RL 库的直接启示是评测：按运动量分层的固定 bench + 成对 A/B。VRL 的
  held-out 评测没有按 prompt 属性分层报告。→ backlog K。

---

## 3. Wan（arXiv 2503.20314）

### 3.1 原文要点
- 后训练数据：百万级图像 + 百万级简单运动视频 + 百万级复杂运动视频，12 大类，强调
  类别平衡与多样性。
- 后训练沿用预训练架构与优化器配置，480p/720p。
- **无 SFT/RLHF/RM 描述**；prompt 对齐一节是推理侧 LLM 改写（Qwen2.5-Plus）。
- 基建（并行、显存、蒸馏）按整体流水线讲，不按阶段分。

### 3.2 对 VRL 的含义
- 无新增。Wan 家族在 VRL 里已是 RL 基座（`wan_2_1` presets），Wan 官方自己没公开 RL。

---

## 4. Kandinsky 5.0（arXiv 2511.14993）视频部分

### 4.1 原文要点
- 视频 SFT：2,833–12,461 个场景，多段过滤（Q-Align >4、美学 >2）；batch 64、lr 1e-5。
- 蒸馏：CFG 蒸馏 + TSCD + 对抗后训练，NFE 100→16，SBS 无损。
- 评测：FVD / VBench / CLIP-score / 专家 SBS，对手 Sora、Veo 3、Wan 2.2 A14B、
  CogVideoX-1.5，用 MovieGen prompt 集。

### 4.2 对 VRL 的含义
- 无新增。

---

## 5. Latent Space 访谈：Ethan He（前 xAI Grok Imagine，前 NVIDIA Cosmos）

来源：https://www.latent.space/p/video-agents（2026-06）。这是访谈转述，不是 report，
以下是原话或接近原话的断言。

### 5.1 要点
- 团队与节奏："as a few engineers, we built it in three months and released the first
  model"（Grok Imagine 0.9）；每天一次 sync，靠迭代次数取胜。
- 数据：**100% 合成视频 caption**；冷启动需要人工按严格协议标注（"让盲人听完能
  重建画面"）；**先做图像模型**，因为便宜且语言-图像连接更密。
- 架构：VAE + DiT；时间压缩 8×8×4 省上下文但有延迟，逐帧压缩可实时但费上下文；
  音视频对齐是最难的模态对齐。
- 后训练特性：reference-to-video（最多 7 张参考图）、video extension（保留全部历史
  上下文）、Grok Imagine Agent（计划 → 生成 → 编辑 → 迭代）。
- 推理：步数蒸馏（Cosmos 有 4 步 / 8 步变体）+ 一致性模型 + GAN loss 组合。
- 中心论点："Visual intelligence are actually mostly coming from language. Every
  improvement came from language model, not from the video model itself"；prompt 改写
  用比 7B 视频模型更大的 LLM（LLaMA/Mixtral）。
- 成本：视频与特征存储"tens of petabytes"，AWS 出口流量比存储还贵；19B 参数、
  数十万亿视觉 token；训练是 IO-bound 而非 compute-bound。
- 世界模型定义：实时（游戏毫秒级、语音 ~200ms）+ 长时程（分钟/小时）+ 交互输入。

### 5.2 对 VRL 的含义
- **prompt 改写在 RL 之外**：VRL 的 rollout 直接吃数据集 prompt。Seedream 的 PE 模块
  和这里的说法一致——线上效果的一大块来自改写。VRL 若要复现线上级效果，需要在
  collector 前加可替换的 prompt rewriter（离线预处理即可）。→ backlog G 的备注项。
- "IO-bound" 与 VRL 在 RTX 5090 上 rollout compute-bound 的实测不矛盾：他说的是
  预训练数据流，不是 RL rollout。
