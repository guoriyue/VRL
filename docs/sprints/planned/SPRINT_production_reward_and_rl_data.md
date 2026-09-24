# SPRINT: 生产级 reward 系统与 RL 数据准备 — 业界做法对照

状态：**planned / research（2026-09-22）**。联网检索的业界实践，对照本仓现状列出差距。
不改代码。与三份相邻 sprint 的分工：

- `SPRINT_verifiable_visual_environments.md`：视觉 reward **模型**是否可信（评分器体检、
  多维、互补、pairwise）。
- `SPRINT_rollout_admission_and_failure_attribution.md`：打分之后**组的准入**与失败归因
  （含 MiMo-V2.6 技术报告精读）。
- 本文：reward **系统**（服务化、生命周期、失败策略、流式）和 RL **数据**（prompt 集的
  构建、过滤、难度、去重与泄漏、在训练中如何更新）。

## 0. 结论先行

1. **reward 系统**：业界收敛到同一形状——reward 是可插拔服务，控制面（谁醒、谁睡、
   何时打分）和数据面（样本直接发给 reward 端点）分离；按样本异步、与 rollout 流式重叠；
   **缺失/失败的分数必须走显式策略，不能静默为 0**；每个子 reward 在聚合后仍可观测。
   本仓已具备大部分（Ray actor / HTTP service / 同卡 parking / 冻结聚合 / 超时），缺口是
   失败策略的显式化、流式打分、多 reward 模型的独立副本与生命周期。
2. **RL 数据**：生产配方都把"prompt 集"当成一等工件，四步——**来源与分类法 → 质量过滤
   与去重/去泄漏 → 用目标模型离线测难度、只留中间带 → 训练中按阶段剔除已解决样本**。
   视觉侧（HunyuanVideo 1.5、Qwen-Image-2.0-RL、Lens-RL-8K、Flow-GRPO）做的是同一件事，
   只是"难度"用组内 reward 极差/方差表示。本仓有 manifest + provenance + `prompt_overlap`，
   缺分类法标注、离线难度剖面、训练中的集合更新。
3. **reward 模型评估**：只看"偏好准确率"不够；PPE / RewardBench 2 的结论是细粒度准确率
   与**低分位**（最差域）聚合最能预测下游 RL 效果，且要测 best-of-N 下的表现——
   RL 优化的正是 reward 的尾部。

## 1. reward 系统：业界做法

### S1 — verl Reward Loop（事实上的默认实现）

- 文档：[Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)；
  重构 RFC：[verl #5158](https://github.com/verl-project/verl/issues/5158)。
- **设计**：`RewardLoopManager`（控制器）把 batch 分给多个 `RewardLoopWorker`（Ray），
  worker 内 `asyncio.gather` 按**单样本**并发；`RewardManager` 子类只实现
  `run_single(data)`（naive / dapo / limit / remote）。三类 reward 共存：规则函数、判别式
  RM、生成式 RM（GenRM = 模板 + 调用 + 解析），混合情形写在一个自定义函数里。
- **资源两档**：colocate（默认，rollout 全部结束后才打分）/ standalone
  （`reward.reward_model.enable_resource_pool=True`，独立资源池，rollout 完一条打一条）。
  规则 reward 或独立池时开启 `enable_agent_reward_loop`，**打分与 rollout 流式重叠**。
- **自定义函数签名**：`data_source, solution_str, ground_truth, extra_info`，GenRM 另给
  `reward_router_address`；同步/异步自动识别。
- **本仓对照**：`RewardFunction.score_batch`（批级）、`InProcessRewardScorer`、
  `vrl/rewards/ray.py::_RewardActor`、`vrl/rewards/service/`（HTTP）、colocated parking
  ——资源两档都有。缺：**单样本粒度的流式打分**（今天 collector 在一个 request 全部生成
  完后整体送 reward；MILES 按 microgroup、verl 按单条）。`data_source` 字段对应本仓的
  `data.task_type` / manifest metadata，按数据源路由 reward 函数的能力本仓没有。

### S2 — verl-omni 多 reward 模型服务 RFC（与本仓场景最接近）

- [verl-omni #432](https://github.com/verl-project/verl-omni/issues/432)；
  [VeRL-Omni 发布博客](https://vllm.ai/blog/2026-05-14-verl-omni)；
  [仓库](https://github.com/verl-project/verl-omni)（diffusion / omni RL，已支持 MiniMax-H3、DiffusionOPD）。
- **设计**：reward 模型由独立 vLLM / vLLM-Omni server 托管，**控制面与数据面分离**
  （trainer 管部署，worker 直接请求端点，权重不进 trainer）。同步生命周期：唤醒所需部署
  → 等副本就绪 → 提交 → 等全部完成 → **在冲突工作前让相位托管的部署休眠**。部署分常驻
  与相位托管两类；每个 reward 模型有自己的副本、路由、并行拓扑（TP/PP/DP）。
- **两条原则**：聚合后每个模型的结果仍可观测；**缺失或失败的模型结果必须遵循显式策略，
  不能静默默认为 0**。
- **本仓对照**：wake/score/sleep = 本仓 CuMem parking（`InProcessRewardScorer` docstring）；
  聚合可观测 = `RewardOutput.components`（注意另一 sprint 记录的 `MultiReward` 未合并子
  components 的缺口）；冻结聚合缺轴时本仓已**抛错**（`registry.py:411-413`）。
  缺：**失败策略作为配置项**（drop 样本 / drop 组 / 重试 / 失败整步）——今天超时直接抛
  `deadline.timeout_error()`（`runtime.py:155`），整步失败；一次 VLM 解析失败没有"该样
  本不参与训练"的中间选项。

### S3 — 16 个异步 RL 库的横向总结（HF blog）

- [Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape)。
- **结论**：SLIME、MILES、PRIME-RL、AReaL、NeMo-RL 能同时支持 GRPO 与 on-policy 蒸馏，
  正是因为打分阶段是**接口**（HTTP 端点 / Ray actor / 同卡 forward 三选一）。三种部署：
  独立预处理池（PRIME-RL，打分与训练并行）、Ray actor、同卡 forward（简单但慢）。
  过期样本三种正交策略：**每样本版本号**、队列深度上限、重要性采样校正；推荐以每样本
  `model_version` 为基础，其他策略都变成可插拔的准入策略。
- **本仓对照**：三种部署本仓都有；`policy_version` 按请求而非按样本；continuous 路径有
  `max_stale`。与 admission sprint 的 `GroupOutcome.policy_version` 对齐即可。

### S4 — Relax（omni-modal 异步 RL 引擎）

- [arXiv 2604.11554](https://arxiv.org/html/2604.11554v1)。
- **设计**：控制面 / 计算面 / 数据面（TransferQueue）三平面；规则 reward 在 rollout 循环里
  直接算，GenRM 作为独立 Ray Serve 服务可单独扩缩与恢复；多模态字段（图像/音频/视频）
  在队列里**按字段独立读写**；一个 `max_staleness` 覆盖同步到异步全谱；**两级恢复**：
  无状态角色（advantage 计算、reward 服务）原地重启，有状态角色（actor、rollout）全局重启；
  微批一生成完就写入队列，消除长尾。
- **本仓对照**：reward 服务失败今天会让整个 run 失败；"reward 服务原地重启不影响训练
  状态"是值得记下的目标形态（reward 无状态，可以重启重打）。

### S5 — OpenRLHF / NeMo-RL（参考）

- OpenRLHF 远程 RM：[Agent/RL 训练指南](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
  （远程 reward server 走 HTTP，`reward_func` 自定义）。
- NeMo-RL：Ray 编排，Megatron 训练 worker、vLLM 生成 worker、Gym 环境与 judge 模型调度在同
  一 Ray 集群；已有 [Qwen-Image flow-GRPO 配方 PR](https://github.com/NVIDIA-NeMo/RL/pull/3408)。

## 2. RL 数据准备：业界做法

### D1 — LLM 推理 RL 的标准四步（配方已收敛）

- 综述：[100 Days After DeepSeek-R1](https://arxiv.org/html/2505.00551) §数据部分；
  [Light-R1](https://arxiv.org/pdf/2503.10460)、[Skywork-OR1 / Klear 等]
  (https://arxiv.org/pdf/2508.07629)、[Kimi k1.5 §2.1 RL prompt set](https://arxiv.org/html/2501.12599v1)。
- **质量过滤**：去掉含外链/图的样本；数学用 Math-Verify 可验证，代码必须带完整单测
  （Skywork-OR1：105k 数学 + 13.7k 代码）。
- **去重与去泄漏**：RL 集与 SFT 集之间做精确 + 近似去重；**9-gram 过滤**防与公开 benchmark
  重叠；更强的做法用 LLM 比较相似题（OpenMathReasoning / OpenThoughts3）。
- **难度**：用目标（或 SFT）checkpoint 对每题采样 k 次（常见 5 或 8），**只留中等通过率**，
  去掉全对与全错。Kimi k1.5 明确说 prompt 集的质量与多样性决定 RL 效果，并能缓解 reward
  hacking 与过拟合。

### D2 — POLARIS：难度分布的形状 + 训练中更新（最具操作性）

- [博客](https://hkunlp.github.io/blog/2025/Polaris/)。
- 每题 8 次 rollout（T=0.6）的通过率作难度；**删掉 8/8 全对**，把数据集做成"镜像 J 形"
  （难题多、易题少）；**每个训练阶段结束后删除准确率 > 0.9 的样本**，维持分布形状——
  模型变强后集合跟着变。采样温度分阶段抬高（7B：0.7→1.0→1.1）。最终 53K（7B）/ 30K（4B）。
- **本仓对照**：manifest 是静态的；admission sprint 的 `prompt_outcomes.jsonl` 账本正好
  提供"每阶段结束后按通过率/极差剔除"所需的数据。

### D3 — DAPO 动态采样（在线版过滤）

- MiMo-V2.6 §4.1 引用；本仓已有下界版本（零 advantage 组丢弃）。区别：DAPO 丢弃后**继续
  采样直到凑满 batch**，本仓丢弃后 batch 变小。见 admission sprint §2。

### D4 — 视觉生成的 prompt 集构建

- **HunyuanVideo 1.5**（[arXiv 2511.18870](https://arxiv.org/html/2511.18870v2) §4.2）：
  I2V 的 RL prompt 集**覆盖 100+ 类别**，从高审美图出发，VLM 生成候选 prompt 后**人工核验**
  图文一致；reward 四维（文本对齐、图像对齐、画质、运动）；探索靠同时变 seed 与 CFG，
  采样用 MixGRPO。T2V：**O(10K) 平衡 prompt 集**（LLM 生成 + 训练视频 caption），先用 SFT
  checkpoint 每 prompt 生成 N 个候选、人工 GSB 标注做 DPO，再上在线 RL。
- **Qwen-Image-2.0-RL**（[arXiv 2606.27608](https://arxiv.org/abs/2606.27608)）：
  **组内 reward 极差过滤**做 prompt 筛选；**按类别校准 reward 权重**。
- **Lens-RL-8K**（[arXiv 2605.21573](https://arxiv.org/pdf/2605.21573)）：8,406 个 RL
  prompt，**分类法驱动**构建，覆盖 T2I 广泛场景。
- **Flow-GRPO**（[论文](https://arxiv.org/pdf/2505.05470)，
  [README](https://github.com/yifan123/flow_grpo/blob/main/README.md)）：GenEval 用模板 + 随机
  组合生成 prompt；**测试集严格去重——只差物体顺序的 prompt 视为同一条，并从训练集删掉**。
- **运动分布平衡**：把每条视频 prompt 标到一个运动类型（静态场景、环绕运镜……）以平衡
  运动分布（检索摘要，[相关工作](https://arxiv.org/pdf/2510.26794)）。
- **ViPO**（[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/33bb58be3f0e903c75afa73d75b5c67e-Paper-Conference.pdf)）：
  30 万视频偏好对，三类均衡。
- **本仓对照**：`manifests/*/report.json` 已记录来源 URL、commit、`prompt_overlap`
  （`vrl/scripts/data/derive_text_video_targets.py:147`）；`vrl/trainers/data/provenance.py`
  在启动门检查 manifest 与 source report 一致。缺：**分类法标注字段**（类别、运动类型、
  难度档）、**语义级去重**（只差顺序/同义改写）、**离线难度剖面**、按类别的 reward 权重。

## 3. reward 模型评估：什么指标能预测 RL 效果

- **PPE**（[arXiv 2410.14872](https://arxiv.org/abs/2410.14872)）：12 域 12 指标 + 真实
  RLHF 实验标定。结论：**准确率是最好的预测量**；把各域分数按**更低分位**聚合（更看最差
  域）与下游相关性更高——衡量的是 reward 的鲁棒性。
- **RewardBench 2**（[arXiv 2506.01937](https://arxiv.org/html/2506.01937)）：同时报告与
  **best-of-N** 和 RLHF 训练的相关性。
- **视觉侧**：EditRewardBench、VideoGen-RewardBench、MMRB2 已被各论文当作 reward 模型的
  验收集（见 reward sprint R1/R11/R13）。
- **本仓对照**：`vrl/scripts/rewards/{analyze_scores,calibrate_scores,stress_media,rescore_media}.py`
  和 `vrl/rewards/{diagnostics,calibration,annotation}.py` 已提供健康报告、排序分歧、重复性、
  冻结组合拟合与人工偏好导入导出。缺：**best-of-N 视角**（在同一 prompt 的 N 个样本里，
  reward 选中的那张是否是人选的那张——这才是 GRPO 实际消费 reward 的方式）和**按域低分位
  聚合**的报告项。

## 4. 差距清单（按投入产出排序）

| # | 差距 | 业界依据 | 本仓落点 | 规模 |
|---|---|---|---|---|
| G1 | reward 失败走显式策略（drop 样本 / drop 组 / 重试 / 失败整步），不静默 0、也不一律整步失败 | S2 RFC；MiMo GAR "grader 输出不可用时回退原 advantage" | `vrl/rewards/runtime.py` 超时路径；`MultiReward.score_batch` | 小 |
| G2 | prompt 集分类法字段（类别、运动类型、难度档）+ 按类别报告 reward | D4 HunyuanVideo 100+ 类、Lens 分类法、Qwen 按类别权重 | manifest metadata、`provenance.py` spec、`analyze_scores` 分层（已有 `stratify ... by task metadata` 提交 `1aa5f8644`，复用） | 小 |
| G3 | 语义级去重与 train/eval 泄漏检查（顺序不同 / 同义改写 / n-gram） | D1 9-gram、D4 Flow-GRPO 顺序去重 | `vrl/scripts/data/common.py`，`report.json` 的 `prompt_overlap` 扩展为近似重叠 | 小 |
| G4 | 离线难度剖面：目标 checkpoint 每 prompt 采 k 个样本，记极差/方差，写进 manifest | D1、D2、R14 AdaGRPO | 复用 `rescore_media` + `eval.sampling`；输出进 admission 账本 | 中 |
| G5 | 训练中按阶段剔除已饱和 prompt（POLARIS）/ DAPO 式补采到满 batch | D2、D3 | admission sprint §2 策略 2/3 | 中 |
| G6 | reward 评估加 best-of-N 选择一致率与低分位域聚合 | §3 PPE、RewardBench 2 | `vrl/rewards/diagnostics.py` 新报告项 | 小 |
| G7 | 单样本/微组粒度的流式打分（生成完一组就打，不等整个 request） | S1 verl、MILES streaming reward、S4 Relax | collector 与 reward runtime 的边界 | 大，先测收益 |
| G8 | reward 服务原地重启不影响训练状态 | S4 Relax 两级恢复 | `vrl/rewards/service/` | 大，多节点前不做 |

knob 规则照旧：G1 的失败策略只在出现第一个需要非默认策略的 preset 时加配置；G4/G5 的
阈值是本仓测出来的数，不照抄论文（POLARIS 的 0.9、8/8 是 LLM 二元奖励上的数）。

## 5. 不做

- 不照搬 LLM 的二元通过率阈值到连续视觉 reward；用组内极差/方差表达难度。
- 不为"与 verl 同构"重写 reward 层；本仓的 Ray actor / HTTP / 同卡 parking 三档已覆盖，
  差距是策略与观测，不是形状。
- 不在单机上做多副本 reward 路由与两级恢复（G8）。
