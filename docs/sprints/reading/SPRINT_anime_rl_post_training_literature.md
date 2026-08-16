# SPRINT：动漫域 RL 后训练文献调研（reading）

状态：**reading / 决策依据（2026-08-15）**。KIND：**reading**——这不是待办清单，
是一次深度检索的结论存档，供动漫/插画方向的 recipe 设计与留出评测设计引用。

调研方式：五路并行检索 → 去重抓取 → **每条断言三票对抗性验证**（需 2/3 反驳才淘汰）
→ 带引用综合（105 个 agent / 941 次工具调用）。下文只保留通过验证的断言，并标注
置信度与验证票型；被反驳的断言已剔除（见 §6 的一条记录）。

---

## 0. 一句话结论

**动漫静图的 RL 后训练在公开文献里基本是空白**：没有任何论文对 Animagine /
Illustrious / Pony / NovelAI 系做 RLHF / GRPO / DPO。最接近的在域工作是**动漫视频**
（AnimeReward + GAPO，基座恰好是 CogVideoX-5B）。其余可用的全是通用域方法学，其
奖励栈与 reward-hacking 结论只能**外推**到动漫域。而 reward hacking 的证据非常硬，
且与本仓库 SANA 那次后验（训练曲线 +0.46、留出全在噪声内、结构崩坏）完全吻合。

---

## 1. 动漫域直接相关工作

### 1.1 空白是系统性的（高置信，6 次独立验证一致）

**没有已发表工作**把 RL/偏好后训练用在动漫/插画**静图**模型上。验证者对原文做了
显式 grep：

- Flow-GRPO：全文 `anime|illustration|manga|cartoon` **仅 1 命中**，是一条 DrawBench
  prompt "Young man with an orange beard, cartoon style"；
- Pref-GRPO：`anime|manga|Animagine|Illustrious|Pony` **零命中**（"Illustration" 只作为
  UniGenBench 五个 prompt 主题之一出现）；
- VisionReward：零动漫提及，数据是 ImageRewardDB / HPDv2 / Pick-a-Pic；
- DanceGRPO：无动漫实验、奖励模型或评测；
- Adv-GRPO 是唯一部分例外，且仅在 §4.5 风格定制（动漫风格参考集，Fig. 11/16）——
  那是风格迁移演示，不是动漫域 RL 后训练。

全语料的实验基座是 SD3.5-M / FLUX.1-dev / SDXL / SD v1.4 / CogVideoX / HunyuanVideo，
**没有任何动漫 checkpoint**。

来源：[Flow-GRPO 2505.05470](https://arxiv.org/abs/2505.05470)、
[DanceGRPO 2505.07818](https://arxiv.org/abs/2505.07818)、
[Pref-GRPO 2508.20751](https://arxiv.org/abs/2508.20751)、
[Adv-GRPO 2511.20256](https://arxiv.org/abs/2511.20256)、
[VisionReward 2412.21059](https://arxiv.org/html/2412.21059)、
[Illustrious 2409.19946](https://arxiv.org/html/2409.19946v1)

### 1.2 最接近的在域工作：动漫视频（高置信，3-0）

**《Aligning Anime Video Generation with Human Feedback》**（Bilibili + 复旦，2025-04，
[arXiv 2504.10044](https://arxiv.org/abs/2504.10044)）——**基座正是 CogVideoX-5B**，
与本仓库 `cogvideox` 家族直接对应。

**AnimeReward**：首个 30k 人工标注动漫偏好数据集训出的六维奖励模型，分两组——

| 组 | 维度 | 实现 |
|---|---|---|
| 视觉表现 | Visual Smoothness / Visual Motion / **Visual Appeal** | 观感 = SigLIP + 回归头；运动 = ActionCLIP |
| 视觉一致 | Text-Video / Image-Video / **Character Consistency** | 角色一致 = BLIP + GroundingDINO/SAM 抠角色 mask |

原文关键设计选择：*"Unlike approaches that rely on a single VLM to jointly train reward
scores for all dimensions, AnimeReward employs specialized VLM for different dimensions,
training them individually through reward score regression."*（**每维一个专用 VLM，
分别回归训练**，而非单 VLM 联合打分。）

**GAPO**（§3.4, Eq. 12–13）：gap 加权的 DPO 变体。`G_i = α^R̃(v_i)`（R̃ 归一到 [0,1]），
`L_GAPO = (G_w − G_l) · L_DPO`，α=2，DPO β=5000。作用是**放大偏好差距大的样本对、
压低差距小的**。训练数据：2k 偏好对，从每组 4 个自生成视频中取最好/最差。
配置：49 帧、16fps、480×720。

**论文的核心动机断言**（对"要不要做动漫专用奖励"的直接依据）：通用奖励模型
（为真实世界/T2V 构建）因**域差**会系统性误判动漫内容。

### 1.3 对静图的可迁移性只是部分（高置信，3-0）

六维里 **Visual Appeal、Character Consistency、Text-Consistency 可迁移到插画**；
Smoothness、Motion、Image-Video Consistency 是**视频专属**。

验证者点名：**角色一致性组件最值得复用**——BLIP 微调后把 GroundingDINO/SAM 抠出的
mask 与动漫 IP 角色关联，这正是插画域"角色一致性"缺的那块。

---

## 2. 一个必须知道的反对意见：Illustrious 拒绝偏好微调

（高置信，3-0）[Illustrious 2409.19946](https://arxiv.org/html/2409.19946v1) 附录 A.2.1
**原文**：

> *"Fitting a baseline model into human preferences [Rafailov et al. 2024] [Yang et al.
> 2023] [Lee et al. 2023] can degrade its performance on the true data distribution. This
> also reduces the diversity of image generation, limiting the model's applicability. Such
> overfitting makes future fine-tuning significantly more difficult compared to using an
> unbiased model... For this reason, to ensure broader public usability, we have decided to
> release non-fine-tuned base models."*

它引的三篇正是 DPO(Rafailov) / 扩散无奖励模型 RLHF(Yang) / T2I 奖励模型 RLHF(Lee)——
**"这是 RLHF/DPO"是论文自己的归类**，不是我们的解读。验证也确认全文无其他偏好微调
阶段（其贡献是 batch size / dropout 控制 / 训练分辨率 / 多级 caption；"GUIDED variants"
是 LECO 安全控制，不是美学优化）。

**射程界定（高置信，3-0）**：这是**关于"偏好调优该放在技术栈哪一层"的论证**，
不是"美学奖励 RL 在插画上失败"的实证。验证者原话：该反对**针对发布的 BASE 模型**，
且**很大程度由下游可微调性驱动**（"makes future fine-tuning significantly more
difficult"），**不应升格为"证据表明它不work"**。另注：论文只有 v1（2024-09-30 提交，
约 23 个月前），无修订软化该段，但领域已经推进。

> **对我们的含义**：做 **LoRA 后训练**（而非污染基座权重）不在它的反对射程内；
> 但"降低多样性"这条警告与 §4 的多样性坍塌证据互相印证，必须进留出评测。

---

## 3. 通用域方法学（动漫工作会继承的底座）

### 3.1 算法底座（高置信）

- **Flow-GRPO**（NeurIPS 2025，[2505.05470](https://arxiv.org/abs/2505.05470)）：
  **ODE→SDE 转换**（把确定性 ODE 转成在所有时间步匹配原模型边缘分布的等价 SDE，
  从而支持 RL 探索所需的统计采样）+ **Denoising Reduction**（训练时减少去噪步数、
  推理保持原步数）。会场三方确认（neurips.cc 海报页 / NeurIPS 托管 slides / 官方 repo）。
- **DanceGRPO**（[2505.07818](https://arxiv.org/abs/2505.07818)）：覆盖扩散与 rectified
  flow，"three key tasks and four foundation models (Stable Diffusion, HunyuanVideo,
  FLUX, SkyReels-I2V)"，增益 "up to 181%"（溯源到 HunyuanVideo T2V 的 VideoAlign 指标）。

### 3.2 canonical 奖励栈（高置信，靠一手论文用法汇聚）

**HPSv2/2.1、PickScore、ImageReward、LAION Aesthetic Predictor、CLIP Score、
UnifiedReward**；语义基准 **GenEval / T2I-CompBench / DrawBench / UniGenBench**。

各家实际用法：Flow-GRPO 训练用 GenEval 规则奖励 + OCR 奖励 + PickScore，评测用
Aesthetic Score / DeQA / ImageReward / UnifiedReward on DrawBench；DanceGRPO 报
HPS-v2.1 / CLIP Score / VideoAlign / GenEval；Pref-GRPO 语义用 UniGenBench / GenEval /
T2I-CompBench，质量用 UnifiedReward / ImageReward / PickScore / Aesthetic。

> **方法论注记**：有三条断言原本引自一篇综述（arXiv 2508.10316）主张同一分类法，
> 但在验证中 **0-3 被全票反驳**。因此本条**只建立在一手论文的汇聚用法**上——这反而
> 是更强的证据形式。

### 3.3 非标量奖励：细粒度分解（高置信）

**VisionReward**（[2412.21059](https://arxiv.org/html/2412.21059)）微调 CogVLM2 /
CogVLM2-Video 回答**层级分类法下的二元判断题**（图像：5 维 / 18 子维 / 61 题；
视频：9 / 20 / 64），再用**在人类偏好对上拟合的逻辑回归线性权重**合成标量：
`R = Σ w_i · 1[A_i = "yes"]`，权重由 `y = ΔX W^T` 学得。

论文明确对立于 ImageReward/PickScore/HPSv2，批评它们 *"lack of interpretability and
risk of unexpected bias"*、*"scoring mechanisms lack transparency"*。

**重要**：这是**可迁移的架构模板**（假想的动漫奖励模型可照此搭），**不是动漫先例**
——该论文零动漫内容。

---

## 4. Reward hacking：失败模式与机理（本节证据最硬）

### 4.1 签名式失败：奖励涨、画质塌（高置信）

[Pref-GRPO](https://arxiv.org/abs/2508.20751) Fig. 9 原文：*"Reward hacking under HPSv2.
HPSv2 scores continue to rise, but image quality collapses around step 160, manifesting as
severe over-saturation."* Fig. 2 关于 UnifiedReward：*"an unnaturally dark style, despite
the rising reward."* 第 936 行确认分裂："oversaturation for HPSv2, dark style for
UnifiedReward."

独立佐证：[《Understanding Reward Hacking in Text-to-Image RL》2601.03468](https://arxiv.org/abs/2601.03468)
（CVPR 2026）报告 HPS 诱发 "over-saturated colors" / "unrealistic and over-saturated images"。

**⚠ 验证附加的 caveat**：step-160 那张图是**单次运行**（64×H20、25 采样步、8
rollouts/prompt、5k prompts），**无种子重复、无误差棒**——崩塌"点"的精确位置不可当
定量结论用，趋势可用。

其他已记录的奖励特异性偏置：纯美学奖励 → **"油光"过度平滑纹理**，混入 CLIP 文图
对齐可缓解；优化 PickScore 会**降低画质**；优化 OCR 类文本渲染奖励会**降低美学**。

### 4.2 机理：GRPO 的"虚假优势"（高置信）

[Pref-GRPO](https://arxiv.org/abs/2508.20751) 原文：*"when pointwise reward models assign
nearly identical scores to images within the same group, the normalization becomes highly
sensitive to small perturbations"*；*"dividing by a small σ_r produces disproportionately
large normalized advantages, making the update direction overly dependent on noisy reward
differences"*；*"minimal score variations are exaggerated, pushing the policy toward
extreme, reward-hacked behaviors."*

**跨领域独立佐证**：LLM-RL 侧的 Dr. GRPO 去掉了标准差分母（`Â = R − μ`），理由是
"dividing by a small standard deviation can result in unstable gradients"，且"消除标准差
归一化可缓解"。

> **对我们的含义**：这解释了 SANA 那次"训练奖励涨 0.46、留出全噪声"的机理——
> 组内分数聚得紧时，噪声被 σ 放大成了优势信号。

### 4.3 多样性坍塌：质量指标看不见（中置信，2-1）

Flow-GRPO §5.3：*"In the Human Preference Alignment task, removing KL does not affect
image quality, likely due to overlap between PickScore and evaluation metrics"* 且
*"outputs converge to a single style, with different seeds producing nearly identical
results. KL regularization prevents this collapse and maintains diversity."*

独立后续佐证：GRPO-Guard（[2510.22319](https://arxiv.org/abs/2510.22319)）报告 Flow-GRPO
出现 "severe distortions in human body proportions and a marked reduction in facial
diversity"；DiverseGRPO（[2512.21514](https://arxiv.org/abs/2512.21514)）专为缓解此模式
坍塌而生（+171.4% BeyondFID，+18.8% DreamSim）。

置信度为**中**（票型 2-1，两处限定被提出）。**但这是本次调研对我们最有价值的一条**：
**hacking 可以表现为纯多样性坍塌而质量指标保持平稳** → **只看质量类留出指标抓不到它**。

### 4.4 动漫域唯一实证：美学 vs 动态性的取舍（高置信）

[2504.10044](https://arxiv.org/abs/2504.10044) Table 1（VBench-I2V）精确值：
Dynamic Degree **57.33 → SFT 47.33 → Ours 43.33**，同时 Total 85.73→86.24、
Subject Consistency 93.76→95.20、Motion Smoothness 98.70→99.13。

作者归因（原文）：*"videos with higher dynamic degrees are more prone to distortions and
artifacts, which in turn degrade overall visual quality and negatively impact human
preference ratings."* 人评佐证：对 Baseline 69.6% 胜、对 SFT 61.6% 胜（3 标注者、
500 条测试集）。

**验证的两点修正**：(a) 14 分降幅里 **SFT 单独占 10 分，GAPO 只贡献最后 4 分**——
说"对齐压制运动"高估了对齐的独立份额；(b) 作者把它归因为**真实人类偏好**（偏好无
瑕疵输出），**不是** reward hacking。

---

## 5. 缓解手段

| 手段 | 出处 | 证据强度 |
|---|---|---|
| **KL 正则**（逐任务调系数、保持小而近似恒定） | Flow-GRPO §5.3 | 高，但作者自陈不充分 |
| **成对偏好奖励**（绕开小 σ 放大） | Pref-GRPO | 中 |
| **对抗协同训练奖励**（参考图为正样本） | Adv-GRPO | 中，作者自述未独立验证 |
| **MPO / Pareto 支配筛对** | VisionReward | 高（结构性） |
| **artifact 奖励模型作留出正则** | 2601.03468 | 高 |

细节：

- **KL**：Flow-GRPO 原文 *"tuning the KL coefficient to keep the divergence small and
  nearly constant during training"*，并主张 *"KL regularization is not empirically
  equivalent to early stopping"*。β 逐任务调（App B.4：GenEval 与 Text Rendering 0.04，
  PickScore 0.01；另一 HTML 抓取报 0.004/0.001，验证者以 PDF 正文为准）。作者自己的
  Limitations 软化了这条——**充分性被后续工作实际争议**。
- **Adv-GRPO**（[2511.20256](https://arxiv.org/abs/2511.20256)）：GAN 式判别器损失
  （Eq. 6）`J_reward(φ) = −E[log R_φ(x_r)] − E[log(1−R_φ(x_g))]`，用**参考图作正样本**
  监督奖励模型，声称"largely avoid being hacked"。**置信中**：未评审的 2025-11 预印本，
  70.0%/72.4% 胜率是作者自办人评（12 评估者、400 prompts）。
- **MPO（多维偏好优化）**：DPO 变体，按 **Pareto 支配**筛训练对——只有当偏好样本在
  **每一个**维度上都不差时才保留该对，而非只看聚合总分。**这从结构上堵死了经典的
  hacking 交换**（奖励重的那维涨、解剖等其他维静默退化）。

### 5.1 领域共识（高置信，多个独立组）

标量偏好奖励模型是**人类判断的不完美代理**，**系统性易被 hack**——这正是整条非标量
奖励路线（分解式、成对、对抗、artifact 特异）的动机。除 Adv-GRPO / 2601.03468 外，
同前提的独立工作还有 GARDO(2512.24138)、RewardDance(2509.08826)、HPSv3++(2606.14657)、
centered reward distillation 等。

---

## 6. 留出检测协议（可直接落到我们的评测梯子）

（高置信，由各家实践汇聚）**三件缺一不可**：

1. **与训练奖励不相交的质量指标**，且在**不同于训练的 prompt 分布**上计算。
   实例：Flow-GRPO 在 DrawBench 上算 Aesthetic/DeQA/ImageReward/UnifiedReward，
   **从不**在 GenEval/OCR 的训练 prompt 分布上算；Pref-GRPO 明确划线：
   *"ImageReward, PickScore, and Aesthetic are disjoint from the training rewards and
   serve as the independent quality metrics"*（并正确排除了两侧都出现的 UnifiedReward）。
2. **显式的多样性/动态度指标**——因为质量可以平、多样性已塌（§4.3）。
   动漫视频论文用 VBench-I2V（含 Dynamic Degree）。
3. **人类胜率**做最终裁决（该论文：3 标注者、500 条测试集、win/tie 率）。

> **⚠ 验证标记的关键 caveat**："不相交"只在"**从未作为训练目标**"的意义上成立——
> ImageReward / UnifiedReward / PickScore / HPSv2 **都是在重叠人类偏好数据上训练的
> 偏好 RM**，它们之间存在共享偏置。真正独立的留出信号需要**不同性质**的指标
> （规则类、多样性类、人评），而非换一个偏好 RM。

---

## 7. 对 VRL 的可执行结论

1. **不要用单一 aesthetic 标量**——文献（§4.1 油光/过饱和/暗调）与本仓库 SANA 后验
   双重证实。至少混入文图对齐（CLIP/PickScore 类）或走分解式奖励。
2. **留出评测必须加多样性指标**——这是本次调研对我们最大的增量。SANA 那次的评测
   梯子（held-out + 不同判官 + 人眼）**缺这一条**，而 §4.3 说明质量平稳时多样性
   可能已经塌了。
3. **CogVideoX-5B + AnimeReward/GAPO 是有论文背书的可复现路径**：基座与本仓库
   `cogvideox` 家族对应，奖励模型与算法均已公开描述。
4. **角色一致性**（BLIP + GroundingDINO/SAM mask）是插画域最值得优先实现的动漫专用
   奖励维度——它在通用奖励栈里完全缺席。
5. **GRPO 的 σ 归一化风险**（§4.2）在我们的 GRPO 实现里同样存在；若出现组内分数
   高度聚集（典型：1–5 离散轴的 VLM 判官），需要考虑 `global_std` 或 Dr. GRPO 式
   去分母。
6. **MPO 的 Pareto 支配筛选**是我们的 `RewardRuntimeConfig` 多组件权重之外，值得
   评估的第二种组合语义（当前是线性加权）。

---

## 8. 非目标

- 本文不主张"动漫域 RL 一定有效"——§1.1 的空白意味着**没有先例可援引**，
  §2 还存在一条来自领域内领先模型的成文反对。
- 不复述通用域 RL 方法细节；只记录会被我们继承的接口面与失败模式。
- 数值（step-160、Dynamic Degree 降幅等）均附来源与 caveat，**不得脱离 caveat 引用**。

---

## 引用汇总

| 简称 | 链接 |
|---|---|
| Anime Video + AnimeReward + GAPO | https://arxiv.org/abs/2504.10044 |
| Illustrious | https://arxiv.org/html/2409.19946v1 |
| Flow-GRPO (NeurIPS 2025) | https://arxiv.org/abs/2505.05470 |
| DanceGRPO | https://arxiv.org/abs/2505.07818 |
| Pref-GRPO | https://arxiv.org/abs/2508.20751 |
| Adv-GRPO | https://arxiv.org/abs/2511.20256 |
| VisionReward + MPO | https://arxiv.org/html/2412.21059 |
| Understanding Reward Hacking in T2I RL (CVPR 2026) | https://arxiv.org/abs/2601.03468 |
| GRPO-Guard | https://arxiv.org/abs/2510.22319 |
| DiverseGRPO | https://arxiv.org/abs/2512.21514 |
