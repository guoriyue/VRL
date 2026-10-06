# SPRINT: 图像与视频 RL 的奖励可靠性、多维评分与稳定优化

状态：**planned**。2026-09-21 联网研究并修订；本次只更新研究计划，未实现或验证训练收益。

用户目标：让现有 diffusion / flow 图像、视频 RL 的 reward 更强、更可靠；不建设 LLM
agent 环境工厂作为前置条件。保留文件路径以延续引用。分层拆解/RGBA 是后续视觉能力
路线，agentic 多步任务是更后的可选扩展，见 §5；旧设计可在 git 提交 `2367741d3` 中查阅。

## 0. 决策与证据边界

优先顺序：**现有评分器离线体检 → 多维输出与校准 → 互补奖励/偏好比较 → 短程 RL
验证 → 必要的数据正则化与规模扩展**。依据见 §2 的原始论文和官方实现。

- 一次生成、最终一个训练标量可以支持复杂的 reward，不需要 agent episode。
- 本仓已有 `RewardOutput.components`、多评分器、零权重观测项；缺口是诊断信息的完整
  传递、领域校准、独立验证和失败样例闭环，不能再写成“reward 没有结构”。
- 视觉 reward 多数是人类偏好的代理，不应把评分器称为能证明答案正确的 oracle。
  VLM 写出理由、两个评分器一致、自检通过，都不是正确性的证明。
- 连续奖励关注组内有效排序。全不合格仍可有有效优势；不采用“30% 任务二元混合才开训”
  的任意门槛，也不将全失败自动归因为模型能力不足。
- 数值一致性、奖励有效性、策略分布漂移分别检查。允许有界 BF16 漂移是精度策略，
  不能用放宽 parity 掩盖奖励问题，也不能用严格 parity 证明奖励可信。
- 下文“论文报告”是作者在特定实验下的证据；“本仓建议”是待测假设，不是已经复现。
  调研覆盖 2025–2026 的相关工作并保留必要的评测基础，不声称穷尽所有最新方法。

## 1. 仓库现状与接入位置（路径相对仓库根目录）

| 位置 | 已有能力 / 本次确认的限制 | 后续接入点 |
|---|---|---|
| `vrl/rewards/types.py` | 样本 ID、metadata、逐样本 scores/components/timing；components 限有限浮点 | 保持训练接口；理由、版本和错误状态另存诊断记录 |
| `vrl/rewards/functions/registry.py::MultiReward.score_batch` | 加权求和，零权重仍执行；当前只取每个子 reward 的 `output.scores`，未合并子 `output.components` | 先测试并修复多轴诊断传递，定义命名空间，避免重复推理和 key 冲突 |
| `vrl/rewards/models/geneval_owl.py::GenEvalVerdict` | strict/partial/dense/why；why 不在 score 映射中 | 保留 dense 训练与 strict 评估的区别；导出失败原因 |
| `vrl/rewards/models/kling_video_reward.py` | 本仓已拥有 VideoAlign 推理适配；VQ/MQ/TA/Overall 映射 | 优先逐轴评估，不重新移植另一份 VideoAlign |
| `vrl/rewards/models/unified_reward_video.py` | 已有 alignment/physics/style/overall 的 pointwise 评分 | 先作为独立审计候选；不是现成的 pairwise/Flex 实现 |
| `vrl/rewards/models/{hpsv3,pickscore,aesthetic,ocr}.py` | 偏好、审美、文字等现有信号 | 在本仓样例上比较盲点，不按榜单直接换模型 |
| `vrl/rewards/models/motion_dynamics.py` | RAFT 流幅值；源码明确不判断时间顺序 | 静止退化诊断，不能单独代表运动质量或物理正确性 |
| `vrl/rewards/assets/{video_judge_prompts,kling_prompt_templates,hpsv3_prompts}.py` | 提示已按域隔离 | 评分 prompt/rubric 版本记录；不把模板塞进 trainer |
| `vrl/scripts/rewards/preflight.py` | 现有评分入口 | 复用输入/加载路径，增加离线质量报告，勿混同“能运行”与“评分正确” |
| `vrl/algorithms/grpo/continuous.py`、`vrl/trainers/online/trainer.py` | 已有 KL 与 sft_weight / clean-latent 正则化路径 | 核查具体算法支持后做消融，不宣称已有完整 DDRL 复现 |

## 2. 有用材料：URL、阅读位置、迁移价值与限制

以下只把论文原文、作者项目页、官方代码作为技术依据。论文写法与仓库后续功能可能不同；
实现时记录所用 revision，而非依赖可变的 main。

### R1 — VideoAlign / VideoReward：视频奖励先分维度，再聚合（优先）

- [论文](https://arxiv.org/html/2501.13918v1)，重点 §3.1–3.2、§5.1、附录 C/D。
- [官方仓库](https://github.com/KlingAIResearch/VideoAlign)，实现位置
  [inference.py](https://github.com/KlingAIResearch/VideoAlign/blob/main/inference.py)。
- **论文报告**：按画质 VQ、运动质量 MQ、文本一致性 TA 标注偏好；研究成对偏好、平局
  标签与维度解耦，并建立 VideoGen-RewardBench 检查奖励模型本身。
- **本仓建议**：先测已有 Kling 的逐轴排序；标注允许 A/B/tie/无法判断，按 prompt 留出。
- **限制**：偏好训练优于回归是该实验结果，不是通用定理；论文讨论 Flow-DPO/RWR/NRG，
  不应统称在线 GRPO。视频偏好评分不能代替物理真值。

### R2 — VisionReward：细粒度问题和可解释分量（优先）

- [论文](https://arxiv.org/abs/2412.21059)；[官方实现与数据入口](https://github.com/zai-org/VisionReward)。
- 具体位置：`VisionReward_Image/VisionReward_image_qa.txt`、
  `VisionReward_Video/VisionReward_video_qa.txt`、两目录的 `weight.json`，
  `inference-image.py` / `inference-video.py`。
- **官方实现**：把图像/视频偏好拆成细粒度问答，再加权评分。
- **本仓建议**：借鉴维度定义和人工标注表；先选择当前失败模式对应的少数维度，避免
  每个样本都运行整套昂贵 checklist。
- **限制**：yes/no 是模型判断而非执行证明；权重需在本仓分布校准。不要把细粒度等同准确。

### R3 — ArtifactReward：直接诊断评分器共同漏掉的伪影（优先）

- [原文](https://arxiv.org/html/2601.03468v1)，重点 §3、§4、Table 2、自动 prompt 优化附录。
- **论文报告**：偏好与语义奖励组合仍漏检结构伪影；用人工标记的正常/伪影样例和
  自动 prompt 优化构建互补的 artifact reward。
- **本仓建议**：收集现有 checkpoint 的肢体、几何、重复结构、纹理塌缩反例，先评估一个
  伪影评分器，零权重观察后再考虑加入训练。
- **限制**：主实验是 Janus-Pro，不能直接声称 Wan/SANA diffusion 已验证；“轻量”描述
  数据/适配方式，不保证 VLM 推理便宜。保留风格化、抽象画等合法反例以测误伤。

### R4 — Pref-GRPO：从绝对分数转向组内偏好比较（第二阶段）

- [论文](https://arxiv.org/html/2508.20751v1)，重点 §3.2–3.3、附录 A.2。
- [官方仓库](https://github.com/CodeGoat24/Pref-GRPO)，`fastvideo/rewards/`；
  [reward_paths.py](https://github.com/CodeGoat24/Pref-GRPO/blob/main/fastvideo/rewards/reward_paths.py)
  给出 checkpoint 配置位置，README 有图像/视频配方。
- **论文报告**：很小的 pointwise 分差经组归一化可能放大；用组内两两直接比较的胜率
  作为 reward。主论文是 T2I，仓库后来增加视频支持。
- **本仓建议**：先在同一批缓存样本上对比 pointwise 排序和真实 pairwise 判断；测 A/B
  顺序偏置、平局、循环偏好与人工一致性。
- **限制**：把已有分数排序不等于 pairwise judge；全配对 O(G²)，视频成本需实测。
  相对比较仍会 reward hack，不是正确性的保证；稀疏配对属于另一个待测近似。

### R5 — UnifiedReward-Flex：按内容选择评价重点（第二阶段）

- [原文](https://arxiv.org/html/2602.02380v1)，重点 §3.2–3.3、§4.3；
  [作者项目页](https://codegoat24.github.io/UnifiedReward/flex)；
  [官方仓库](https://github.com/CodeGoat24/UnifiedReward)，README Model Zoo 的 Flex
  推理入口和 `benchmark_evaluation/`。
- **论文报告**：从语义和视觉证据出发建立分层评价标准，训练 reward model，并用于图像
  和视频 GRPO。
- **本仓建议**：按图像/视频、动作/静态场景选择 rubric；先在离线审计比较与固定 rubric
  的收益。固定评分器版本及评价规则再开一段训练。
- **限制**：长解释可能错误，动态规则带来不可比性；不把 VLM 理由当因果归因，也不因
  新模型名字相近就把现有 UnifiedReward-2.0 适配当作 Flex 支持。

### R6 — DDRL：可靠 RL 不只靠换 reward，还要约束策略漂移（第二阶段）

- [论文](https://arxiv.org/abs/2512.04332)；
  [官方技术页](https://research.nvidia.com/labs/cosmos-lab/ddrl/)，重点 Methodology 的
  Motivation / DDRL Framework / Practical Implementation 及 Results。
- **作者报告**：在图像和视频中，将奖励优化与独立真实/合成数据上的 diffusion loss
  结合；用 forward-KL 视角解释数据锚定，并用盲评检查收益。
- **本仓建议**：在已有 clean-latent SFT regularizer 上做受控消融：reward 相同，比较
  原配方与加数据正则；检查质量、语义和多样性。
- **限制**：KL 数字稳定也不能证明没有作弊；现有 sft_weight 不代表目标、权重和采样
  都与 DDRL 一致。数据质量与分布本身会影响结果。

### R7 — Flow-GRPO：图像在线 RL 的基础对照

- [论文](https://arxiv.org/abs/2505.05470)；
  [官方实现](https://github.com/yifan123/flow_grpo)；
  [作者报告中的 KL 对照](https://neurips.cc/media/neurips-2025/Slides/116065.pdf)。
- **论文/报告**：将在线 RL 用于 flow matching，覆盖组合生成、文字与偏好；展示 KL
  对 reward hacking 的缓解。
- **本仓建议**：保留参考模型约束作为对照，不把切换 reward 与算法、采样器变化捆绑。
- **限制**：特定 KL 实验有效不等于 KL 一定消除作弊，尤其需与 R6 的分布外问题区分。

### R8 — VBench：视频独立评估，而非一个总分

- [官方仓库与论文入口](https://github.com/Vchitect/VBench)，重点 README 的 dimensions、
  evaluation 和 custom videos 流程；实现位于仓库的 `vbench/` 目录。
- **官方项目**：分维度评估视频，包括主体一致性、运动平滑、动态程度等。
- **本仓建议**：留出未参与训练的维度，单独报告运动与画质退化；固定 FPS、抽帧和分辨率。
- **限制**：一旦作为训练 reward 就不再是独立证据；动态程度不是越大越好，静态场景
  合法，镜头抖动也会产生流量。基准分数仍需人工检查支持。

### R9 — RSA-FT：奖励对微小扰动的敏感性（探索项）

- [论文](https://arxiv.org/abs/2603.21175)；
  [CVPR 官方摘要/PDF入口](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Reward_Sharpness-Aware_Fine-Tuning_for_Diffusion_Models_CVPR_2026_paper.html)。
- **作者报告**：用生成器参数/生成图扰动来平滑 reward 梯度，缓解 reward hacking。
- **本仓建议**：先借鉴图像小扰动下的分数和排名稳定性诊断。
- **限制**：本次核查摘要层面的机制；梯度优化方案不能直接当作黑盒 GRPO 插件。
  诊断扰动必须保持任务语义，OCR 裁剪、改色等可能真的破坏目标，不能要求分数不变。

### R10 — GARDO：质量与多样性共同检查（探索项）

- [论文与作者项目入口](https://arxiv.org/abs/2512.24138)，摘要给出机制，后续实现前需精读公式。
- **作者报告**：不确定样本的选择性正则、更新参考策略，以及高质量样本的多样性奖励。
- **本仓建议**：先报告同 prompt 不同 seed 的多样性，在保持质量的子集中检查塌缩。
- **限制**：本次未复现；不能直接奖励像素差异（噪声也多样），移动参考也可能跟随坏策略。

### 背景材料

- CodeMidas 与 MiMo-V2.6 的 verifier / 环境纪律移至
  `SPRINT_rollout_admission_and_failure_attribution.md`。
- 检索发现 arXiv:2511.19356 的当前标题已是
  [Rethinking Reward Signals in Video GRPO: When Scores Become Targets](https://arxiv.org/abs/2511.19356)，
  与搜索摘要中的旧标题 Self-paced GRPO 不同；本计划不据旧摘要实施动态课程。

## 2b. 补充材料（2026-09-21 第二轮联网检索；同一格式，编号接续）

R1–R10 覆盖的是"评分器本身可靠不可靠"。这一组补的是用户点名的三个缺口：**局部编辑
的奖励怎么定义**、**分层拆解有没有可执行的检查**、**任务/难度过滤与可验证奖励在视觉
里长什么样**。每条标出与 §4 非目标的关系，供决策；本节不改变 §3 的 P0→P3 顺序。

### R11 — Edit-R1 / Edit-RRM：编辑指令拆成 Keep / Follow / Quality 三类原则再逐条验证（局部编辑，优先）

- [arXiv 2604.27505](https://arxiv.org/abs/2604.27505)（[HTML](https://arxiv.org/html/2604.27505v1)，
  [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Guo_Leveraging_Verifier-Based_Reinforcement_Learning_in_Image_Editing_CVPR_2026_paper.pdf)），
  重点 §3（原则生成与验证）、§4（GCPO 训练 reward model）、Table 2/3。
- **论文报告**：用 Seed-1.5-VL 把每条编辑指令展开成约 10 个问题，分三类——Keep（未提
  及区域不得变）、Follow（要求的改动是否发生）、Quality；结构性检查（物体移除是否彻
  底、位置偏移是否超过图像尺寸 10%）走规则，其余走 VLM + CoT；原则级 0/1 结果加权成
  0–10 分。7B reward model 在 EditRewardBench 78.2%（EditScore-7B 65.9%）；用它做
  GRPO（G=24，β=0.04）把 FLUX.Kontext 总分 5.77 → 6.24，人评 +23.2 GSB。
- **本仓建议**：这是"局部编辑 reward 的最小结构"——三类原则对应本仓可以直接落的
  三个 component：`keep`（遮罩外像素/特征恒等，可执行）、`follow`（VLM yes 概率，
  backlog F）、`quality`（现有 HPS/伪影信号）。先在 P0 的体检集上按三类分别标注。
- **限制**：原则由 VLM 写、多数检查由 VLM 判，仍是偏好代理；"位置偏移 >10%"这类规则
  阈值来自他们的数据。与 §4 非目标不冲突：不需要新 family，只需要 `reference_image`
  条件输入（已有）和三类 component。

### R12 — CoCoEdit：非编辑区用像素指标、编辑区用 VLM，两者分开正则（局部编辑，优先）

- [arXiv 2602.14068](https://arxiv.org/html/2602.14068v1)，重点 §3.2（遮罩获取）、
  §3.3（r_sim 定义）、§3.4（DiffusionNFT 上的区域正则 L_ner+ / L_er-）、Table 1。
- **论文报告**：Qwen2.5-VL-72B 定位编辑目标 → SAM 2 分割 → 膨胀得到编辑遮罩；非编
  辑区奖励 `r_sim = 0.5·SSIM_masked + 0.5·PSNR/40dB`，编辑区由 Qwen2.5-VL-32B 判；
  在 DiffusionNFT 上加两个区域正则：高奖励样本上保持非编辑区、欠编辑样本上放大编辑。
  GEdit-Bench-EN 上 PSNR +1.16–2.8 dB，编辑分保持 6.9–7.8。
- **本仓建议**：`r_sim` 是本仓可以今天就写的可执行检查（纯像素运算，有遮罩即可）；
  它和 R11 的 Keep 是同一件事的两种实现——先做像素版，VLM 版做交叉校验。本仓已有
  DiffusionNFT（`vrl/algorithms/diffusion_nft.py`），区域正则是它上面的一个 loss 项，
  不是新算法。
- **限制**：遮罩来自 VLM+SAM 流水线，遮罩错了 reward 就错——遮罩质量要单独审计；
  "遮罩内确实变了"这条防"什么都不改拿满分"的检查论文里靠 L_er- 而不是 reward。

### R13 — AutoRubric-T2I：从 256 个偏好对自动生成可解释 rubric，ℓ₁ 选出 20 条（图像，优先）

- [arXiv 2605.17602](https://arxiv.org/abs/2605.17602)（[HTML](https://arxiv.org/html/2605.17602v1)），
  重点 §3.1（seed 生成与失败驱动的迭代精炼）、§3.3（ℓ₁ 逻辑回归学权重）、Table 1/4。
- **论文报告**：VLM 对偏好对做 CoT 解释差异，抽出"客观、确定"的 rubric 语句；误排
  的难对触发新 rubric；每条 rubric 的分是 P(yes)，最终 `s = Σ w_j · s_j`，ℓ₁ 剪到 20
  条。MMRB2 OOD 62.5%（HPSv3 微调 59.4%）；Flow-GRPO on SD3.5-M：TIIF 65.3 → 71.6，
  UniGenBench++ 64.0 → 66.9；4×A6000 2–4 小时，只要 256 对。
- **本仓建议**：和 R2 VisionReward 是同一思路，但它给出了**从本仓自己的失败样例生成
  rubric** 的流程——P0 体检集里的人工偏好对正好是它的输入。rubric 级输出直接解释
  "HPSv3 高但违反 prompt 约束"这类 hack，是 §3 P1 "互补信号"的候选实现。
- **限制**：rubric 仍由 VLM 判 P(yes)；20 条对每张图 20 次推理，视频成本要测。

### R14 — （已移出）任务准入 / 难度过滤 / 失败归因

见 `SPRINT_rollout_admission_and_failure_attribution.md`：AdaGRPO 难度带、Qwen-Image-2.0-RL
的组内极差过滤、NGU 重试、CodeMidas / MiMo 的 verifier 自检与归因，以及本仓的
`rollouts/admission/` 块设计都在那里。本文只管"reward 本身可不可信"。

### R15 — Qwen-Image-Layered：分层拆解的数据、重建定义与评测指标（分层，参考）

- [arXiv 2512.15603](https://arxiv.org/html/2512.15603v1)，重点 §3.1（PSD 层提取与合并）、
  §3.2（RGBA-VAE、Layer3D RoPE）、§4.1 指标定义。
- **论文报告**：真实 PSD 用 psd-tools 抽层，过滤异常层、合并空间不重叠层减少层数，
  Qwen2.5-VL 写描述；重建定义 `C_i = α_i·RGB_i + (1−α_i)·C_{i−1}`；指标 RGB L1（按
  GT alpha 加权）、Alpha soft IoU、重建 PSNR/SSIM/rFID/LPIPS。**没有用 RL**。
- **本仓建议**：如果将来做分层任务，这里给了 oracle 数据来源（PSD）和可执行检查的
  精确定义（重建 + alpha IoU）；这两条不依赖任何学习模型，是视觉里少见的"跑测试"型
  验证。相关：[LayerDiffuse](https://arxiv.org/abs/2402.17113)（latent transparency，
  单层 RGBA 生成）、[LayerDecomp](https://arxiv.org/html/2411.17864)（CVPR 2025，带视觉
  效果的拆层 + consistency loss）、[RevealLayer](https://arxiv.org/pdf/2605.11818)
  （2026，遮挡感知拆层）。
- **阶段边界**：作为 §5 的后续分层任务资料保留；不是本轮 reward 体检工具的前置条件。

### R16 — Qwen-Image-2.0-RL：工业配方里的组内极差过滤、高噪声步聚焦、编辑身份保持（图像+编辑，参考）

- [arXiv 2606.27608](https://arxiv.org/abs/2606.27608)（[PDF](https://arxiv.org/pdf/2606.27608)，
  [alphaXiv 概览](https://www.alphaxiv.org/overview/2606.27608v1)），重点 §3 reward
  设计、§4 训练框架。
- **论文报告**：pointwise Likert + CoT 的 VLM reward；T2I 三维（对齐、审美、人像），
  编辑两维（指令遵循、**人脸身份 embedding 级一致性**）；prompt 经"组内 reward 极差
  过滤"进训练；按类别校准 reward 权重；rollout 用 CFG、策略优化不用（hybrid CFG）；
  聚焦高噪声 timestep 以防"只加表面细节"的 hack。Qwen-Image-Bench +2.61，T2I arena
  Elo +78，编辑 arena +93。
- **本仓建议**：三条机制本仓都有对应物——组内极差过滤见 R14；高噪声步聚焦 =
  `timestep_selection=sde_window` 取前段窗口；hybrid CFG = `replay_forward_with_latents
  (classifier_free_guidance=False)`。缺的是编辑身份保持这一 component（人脸/物体
  embedding 距离，可执行，无需 VLM）。
- **限制**：全文摘要未给过滤阈值与权重校准细节，需要读 PDF 对应章节；工业规模数据不可复现。

### R17 — VideoRLVR：视频生成真正的可验证奖励——符号检查器（视频，参考）

- [arXiv 2605.15458](https://arxiv.org/html/2605.15458)，重点 §3（Maze / FlowFree /
  Sokoban 的解析器与检查规则）、§4.2（early-step focus L=10/K=20）。
- **论文报告**：把生成视频解析成抽象状态（路径掩码、格子颜色、动作序列），用图算法
  /状态转移规则判对错；SDE-GRPO 只在前 10/20 步算梯度省 40% 时间；奖励是多个可验证
  分量的乘/加组合。Maze 72.2% 成功率，超过 Sora 2 / Kling V3 / Veo 3.1。
- **本仓建议**：这是"视频里什么任务能有 oracle"的答案：任务本身要带可解析的状态。
  与本仓 `videophy` / `video_world` 那类物理合理性任务的区别在于——那些没有符号检查
  器，只有 VLM 判断。若要一条真正可验证的视频 lane，从这类合成任务起步。
- **限制**：任务是谜题类合成数据，不是自然视频；收益是否迁移到自然视频质量论文只在
  OOD 基准上报告。

### R18 — SoliReward：视频 reward model 的标注噪声与过度优化（视频，参考）

- [arXiv 2512.22170](https://arxiv.org/html/2512.22170v2)，重点 §3.1（单项 Pass/Fail 标注
  vs 成对；Krippendorff α 0.49 vs 0.35）、§3.3（BT-WT 损失）、Table 3。
- **论文报告**：三个客观维度（物理合理、主体畸形、语义对齐）用单项二元标注，跨
  prompt 配对增加数据利用；Bradley-Terry with Win-Tie 迫使两个正样本分数接近，压缩
  reward 空间、减小组内 advantage 方差；ID 78.5% / OOD 80.1%（VideoAlign OOD 71.6%）；
  VBench2 Human Fidelity 0.900 vs 0.870。
- **本仓建议**：补 R1 的标注方法论——本仓 P0 做视频标注时用单项 Pass/Fail 而不是成对，
  一致性更高、成本更低。
- **限制**：论文不做对抗测试或 hack 测试集，"缓解 hacking"是通过损失形式间接得到的。

### R19 — Adv-GRPO：参考图作正样本训练对抗 reward，DINO 稠密特征替代标量（图像，探索）

- [arXiv 2511.20256](https://arxiv.org/abs/2511.20256)（[HTML](https://arxiv.org/html/2511.20256)）。
- **论文报告**：reward model 与生成器交替更新，参考图作正样本、生成图作负样本；用
  DINO 等视觉基础模型的稠密特征而非单标量；人评图像质量 70.0%、审美 72.4% 胜率。
- **本仓建议**：与 R6 DDRL 的"数据锚定"同源，但锚在 reward 侧。本仓有 `sft_latents`
  （目标图的 latent），是现成的参考正样本。
- **限制**：交替训练多一份 reward 优化器与判别器崩溃风险；摘要未给检测 hacking 的
  独立协议。

### R20 — 其余检索到、暂不展开的材料

- [Understanding Reward Hacking in T2I RL, 2601.03468](https://arxiv.org/html/2601.03468)：
  R3 ArtifactReward 的原文；诊断集数字（HPS 53%、Aesthetic 42% 识别伪影）在 §4。
- [Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling, 2602.11146](https://arxiv.org/pdf/2602.11146)
  与 [Video Generation Models Are Good Latent Reward Models, 2511.21541](https://arxiv.org/html/2511.21541v3)：
  在 latent 上打分，省 decode；前者明确记录了"代理 reward 与留出 golden metric 先同
  升后分道扬镳"的曲线，是 P3 观察项的范式。
- [A Systematic Post-Train Framework for Video Generation, 2604.25427](https://arxiv.org/html/2604.25427v1)：
  四维 reward（视频审美、图像审美、运动、对齐）+ GSB 人评；无过滤、无 hack 诊断——
  作为"工业报告不写什么"的对照。
- [Scaling MoE Video Pretraining for Embodied Intelligence, 2607.07675](https://arxiv.org/pdf/2607.07675)：
  六个专门 reward model（画质、对齐、动态度、运动一致、人体运动、物理），明确反对
  单标量。[PhyMotion, 2605.14269](https://arxiv.org/html/2605.14269v1)：结构化 3D 人体
  运动 reward。[PhyPrompt, 2603.03505](https://arxiv.org/abs/2603.03505)：语义先、物理
  后的动态 reward 课程。
- [Kwai Keye-VL-2.0, 2606.10651](https://arxiv.org/pdf/2606.10651)：理解侧的可验证
  reward 清单（grounding IoU、counting 精确匹配、OCR 归一化文本匹配）——生成侧对应的
  可执行检查就是本仓 `ocr`、`geneval_owl` 已有的那类。
- [MAR-GRPO, 2604.06966](https://arxiv.org/pdf/2604.06966)、[AR-GRPO, 2508.06924](https://arxiv.org/pdf/2508.06924)：
  AR/混合家族；后者把 CLIP/HPS 量化到三档以抗 hack。本仓无 AR 家族，仅记录。
- [MixGRPO 项目页](https://tulvgengenr.github.io/MixGRPO-Project-Page/)、
  [DanceGRPO 代码](https://github.com/XueZeyue/DanceGRPO)、
  [Flow-GRPO 代码](https://github.com/yifan123/flow_grpo)：本仓已实现的算法基线。

## 2A. 独立运行边界（2026-09-22 明确）

“框架内复用代码”与“必须启动训练才能使用”是两回事。采用同仓、可独立运行的 reward
评估工具，复用评分服务和协议；目前不拆新仓库、不另造一套评分服务。

| 层 | 职责 | 不负责 |
|---|---|---|
| 现有 reward service/runtime | 接受媒体、prompt、metadata，加载指定评分器并返回评分；支持本地或 HTTP | 标注集管理、比较候选、选择训练组 |
| 独立 reward 评估工具（本 sprint P0–P2） | 读取已有媒体，批量重评分、校准、比较、产出报告和冻结配置 | 启动 generator/trainer、修改训练中的评分配置 |
| rollout 准入（另一个 sprint） | 用已评分组和历史决定 keep/drop/retry，记录决策 | 重新定义评分器、在线调 rubric、证明视觉正确 |
| 训练/实验运行器（P3） | 加载冻结的奖励配置，运行短程 RL；导出 checkpoint 样本供独立工具复评 | 将评分器自检或报告系统塞入训练主循环 |

已有入口：`vrl/rewards/service/server.py` / `client.py` 是独立评分服务与客户端；
`vrl/scripts/eval/score_report.py` 可直接读取已有分数做统计，不加载模型；
`vrl/scripts/rewards/preflight.py` 当前偏向训练配置与合成媒体的连通性检查，不能把它
直接称为完整媒体重评分工具。新增离线入口应接受媒体清单与 reward 配置，而不要求
完整训练配置、generator family 或 Ray 训练集群；仅在所选后端需要时连接其运行环境。

验收必须包含：没有 trainer、optimizer 或 generation worker 的进程也能评分真实图像/
视频；更换候选评分器可复用同一媒体清单；只改聚合权重可复用兼容的原始轴分数。
缓存键须包含媒体、prompt/metadata、模型与预处理/rubric 版本；pairwise 还须包含双方及
顺序，不能误复用单样本分数。

reward 评估结果与准入账本共享稳定的 sample/group/run ID，但分开保存：离线候选比较
可能对同一样本产生多份分数，不能覆盖原训练奖励或改写历史准入决策。线上训练只读取
已选择且版本冻结的配置；校准不随每个 batch 自动生效。

## 3. 实施计划：先证明 reward 更好，再花训练预算

### P0 — 固定评分器体检集与可重现评分（无需生成器训练）

- 从已有生成结果采样图像、视频，覆盖当前模型、prompt 类型、seed 和早晚 checkpoint。
  先确认 artifact 存在，不虚构历史塌缩样例；不足的类别再补生成。
- 建立人工偏好对和失败类别标签。按 prompt/来源组划分 calibration 与 holdout，防止
  同一素材的扰动版本跨集合。保留 tie、无法判断和标注分歧。
- 图像反例：结构伪影、过饱和、重复纹理、错误文字、对象缺失；同时保留合法抽象风格。
- 视频反例：重复首帧、乱序、闪烁、身份漂移、抖动。反转只用于方向/因果明确的 prompt，
  静止只用于要求动作的 prompt；不能把所有编辑后的媒体一律标为坏。
- 逐样本记录 `sample_id/prompt_id/media_hash/model_revision/rubric_revision/preprocess`、
  各轴原始分数、聚合贡献、错误状态、耗时和检查原因。解析失败不是 reward=0，单独报错。
- 重复评分测噪声；固定 decode/FPS/抽帧/resize；测合理预处理变化下的排序稳定性。
- 报告：人工偏好一致率（明确 tie 计分规则）、按失败类型的误排序、排名相关、组内
  分差/饱和率、扰动敏感性、耗时。置信区间按 prompt 分组，避免重复样本虚增置信度。

交付（**计划路径，尚未创建**）：`docs/reports/reward_reliability/` 的基线报告与样本索引，
配套版本化的校准配置；大媒体存 artifact 目录，报告记录可追溯路径而非提交大量二进制。
验收：已有评分器在同一 holdout 上可比较，明确至少一种实际盲点及误伤；无证据则不升级。

### P1 — 多维评分、互补检测与聚合校准

- 修复/验证 §1 所列组件传递问题；一次评分保留所有可用轴，不重复加载模型。
- 图像主奖励与互补伪影/语义信号分开；视频至少保留 VQ/MQ/TA，光流只做适用域诊断。
- 在 calibration 上确定方向、尺度、权重和软惩罚，版本冻结后只在 holdout 验证。
  不用每批 min-max 自动漂移目标；总体质量指标不默认设硬门槛。
- 对“严重缺陷不能被审美高分抵消”的需求，比较加权和与有界软惩罚；硬拒绝仅用于
  明确无效的输入/输出契约。不得在看完 holdout 后继续调权重还称它独立测试。
- 挑一个证据支持的候选与原配方比较；不同时接入所有新 reward 模型。

验收：目标失败类型的误排序改善，正常作品误伤与其他轴退化处于预先声明容忍范围；
同时报告成本。阈值在读 holdout 前确定；样本不足/区间过宽则结论为不确定。

### P2 — 可选 pairwise，与 pointwise 公平比较

- 仅在 P0 显示 pointwise 分差小于评分噪声或人工分辨力明显不足时开展。
- 用同一批输出做真实 pairwise judge，加入交换 A/B 的测试及 ties；测比较预算。
- 在线接入必须保证真实 prompt group、跨 rank 分组与 sample 对齐正确。
  `REWARD_GROUP_ID_METADATA_KEY` 已有，不用 prompt 文本猜 group。
- 固定预算比较直接 pairwise、原 pointwise、校准后的 pointwise；不要把少用样本导致的
  变化误归因于 reward 更好。未经证明不改 GRPO 优势计算默认值。

### P3 — 短程 RL 与长期稳定性验证（需要 GPU，后续执行）

- 同一初始 checkpoint、prompt/seed 计划、采样器、学习率和训练预算：原奖励 vs 新奖励。
- 如需测试数据正则，再增加“新奖励 + 已有 clean-latent regularizer”一臂；不能一次换
  reward、family、算法、KL、采样器后宣称因果。
- 每个检查点保留媒体与各轴轨迹；并行观察训练奖励、留出评分、人工盲偏好、有效多样性、
  ratio/clipping/KL、梯度和非有限值。parity 用事先定义的数值容忍，不要求 BF16 位级相同。
- 配方升级依据独立偏好/质量证据及成本，不依据训练 reward 曲线上升。先做有限预算
  canary，再多 seed/更长运行验证，短程不崩溃不能证明长期无 hack。
- 把新出现的高分坏样本收进下一版 calibration；重新留出测试，版本变更不悄悄生效。

## 4. 架构边界与非目标

- **应改变**：评分诊断链路、失败样例评估、评分尺度与版本记录；新增评分器须由证据驱动。
- **保持**：RewardSample/RewardOutput 公共边界、family 接口、runtime parking、现有
  薄 reward adapter 的一致形状；`types.py` 的轻量导入边界有真实价值。
- **不新建**通用 TaskSpec/Check/Verdict 层来重复现有 registry 聚合；先补明确缺口。
- `REWARD_GROUP_ID_METADATA_KEY` 和 Kling 的特殊 token、输出 key 映射是 schema/protocol
  边界，应保留 ALL_CAPS。大型新 rubric/反例分类表应进入命名清楚的配置或资产；
  不把它们写成 trainer 里的硬编码词表。已有 prompt assets 无需为减少行数合并。
- 本轮不移植 RGBA family、不增加分层拆解任务；后续路线见 §5。无需开发 agent 数据
  工厂，也不照搬 6 容器 oracle。
- 不设通用“运动越大越好”“图像必须写实”或“所有任务都应二元可验证”的隐含目标。
- 本次研究没有下载模型、运行 reward GPU 推理或 RL；论文收益不是本仓已实现收益。


## 5. 后续路线：保留目标，按能力逐步进入

用户在 2026-09-22 明确：当前优先增强 reward，不代表删除未来 RGBA 或 agentic 方向。

1. **当前：独立 reward 评估与训练集成。** 完成 P0–P3，能可靠重评分、比较，并验证
   奖励改动确实改善现有图像/视频生成。
2. **后续：局部编辑、RGBA 与分层任务。** 用现有研究材料定义输入/输出、参考图/遮罩/
   层顺序和任务特定评分；再评估 family 能力与接入成本。RGBA 输出不等于语义拆层。
   重建一致只是必要检查，正常遮挡可重叠、背景可不透明，不沿用旧版一律禁止的约束。
   进入实施的证据：能运行的未训练基线、可信的正反例评分、可观测的学习信号；
   不强制任意二元混合率。独立评分工具应能接收这些新媒体/metadata，不为此重造架构。
3. **更后：有需要再做多步 agentic 视觉任务。** 仅当任务需要观察中间产物、选择编辑/
   工具动作并进行多步信用分配时进入；单次生成、RGBA、反复重评分本身都不要求 agent。
   后续 program 见 `../parked/SPRINT_agentic_visual_rl_program.md`。该文仍有已删除的
   Janus/token-AR 前置说明，启动前必须更新，不能按旧入口直接实施。

这里将可编辑、可分层的视觉生成保留为后续能力目标；agentic 是实现更复杂任务的一条
可选路线，不是所有视觉 RL 的必然终点。后续实施另开有基线与验收的 sprint，本轮不扩建。
