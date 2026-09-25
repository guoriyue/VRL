# SPRINT: 图像与视频 RL 的奖励可靠性、多维评分与稳定优化

状态：**planned**。2026-09-21 联网研究并修订；本次只更新研究计划，未实现或验证训练收益。

用户目标：让现有 diffusion / flow 图像、视频 RL 的 reward 更强、更可靠；不建设 LLM
agent 环境工厂。保留文件路径以延续引用。旧版的分层拆解/RGBA/TaskSpec 方案不再是本
sprint 的实施计划；历史设计可在 git 提交 `2367741d3` 中查阅。

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

### 背景材料（不作为视觉训练方案）

- [CodeMidas §3.2–3.4](https://arxiv.org/html/2609.22068v1)：迁移测试可靠性和对抗审核的
  思路，不搬代码任务的二元难度过滤。论文明确说全成功/全失败不能单独说明原因。
- [MiMo-V2.6 博客](https://mimo.mi.com/docs/zh-CN/news/latest/v2-6)：RL 算力、环境、grader
  三个扩展方向；7k 环境不是我们应达到的视觉数据量。
- 检索发现 arXiv:2511.19356 的当前标题已是
  [Rethinking Reward Signals in Video GRPO: When Scores Become Targets](https://arxiv.org/abs/2511.19356)，
  与搜索摘要中的旧标题 Self-paced GRPO 不同；本计划不据旧摘要实施动态课程。

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
- 不移植 RGBA family、不增加分层拆解任务、不开发 agent 数据工厂、不照搬 6 容器 oracle。
- 不设通用“运动越大越好”“图像必须写实”或“所有任务都应二元可验证”的隐含目标。
- 本次研究没有下载模型、运行 reward GPU 推理或 RL；论文收益不是本仓已实现收益。
