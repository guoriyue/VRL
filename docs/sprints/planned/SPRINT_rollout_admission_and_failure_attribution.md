# SPRINT: rollout 准入与失败归因 — 一个 prompt 组进不进训练、不进是谁的错

状态：**planned（2026-09-21）**。从 `SPRINT_verifiable_visual_environments.md` 拆出：
那份管"reward 本身可不可信"，这份管"打完分之后，这个组该不该训、失败归给谁"。
两份共用同一个按 prompt 的结果账本。

## 0. 结论先行

1. "这个 prompt 组这次进不进训练、不进的话下次怎么办"是一个决策，今天在本仓分散
   在三处各做一半、互不通气、且不留记录：trainer 的零 advantage 过滤（只有下界）、
   `data.sampler`（不看历史）、NGU sprint（重试，只覆盖 continuous 路径）。
2. 它应该是一个块：`vrl/rollouts/admission/`，位于 reward 之后、trainer 之前，
   对应 MILES 的 `miles/rollout/filter_hub/`。第一步零行为变化：把现有过滤搬进去
   并开始记账；之后每个策略（难度带、重试、回流采样权重）带自己的测量再进来。
3. 归因按四层从下往上排除：精度 → verifier → 任务 → 模型。今天本仓只有第一层；
   "全失败"在前两层没排除之前不能记到模型头上——这是 CodeMidas §3.4 保留的区别。

## 1. 今天的三处半实现

| 位置 | 做了什么 | 缺什么 |
|---|---|---|
| `vrl/trainers/online/trainer.py:1043-1081` `nonzero_advantage_mask` / `adv_zero_rate` / `trained_prompt_num` | 训练时丢掉全同分的组 | 只有下界；静默丢；不记哪个 prompt、为什么 |
| `data.sampler`（`vrl/config/schema.py:214`，`vrl/trainers/data/prompt_sampler.py`） | rollout 前 random / without_replacement 抽 prompt | 不看历史结果，无法把预算倾向未解决的 prompt |
| `docs/sprints/planned/SPRINT_ngu_adaptive_sampling.md` | 全失败的组按概率重采，带预算 | 计划把决策放进 `rollouts/orchestration/continuous/producer.py`，legacy 全批路径用不上 |

参照：MILES `miles/rollout/filter_hub/{common_filters,dynamic_sampling_filters}.py` —
`apply_reward_nonzero_std_filter`、`apply_aborted_filter`，rollout 后训练前的可插拔过滤。

## 2. 块设计：`vrl/rollouts/admission/`

```python
# 归属：rollouts 域。输入是打过分的组，输出是决策；不碰 reward 计算，不碰 loss。
@dataclass(frozen=True)
class GroupOutcome:            # 一个 prompt 组一次 rollout 的事实
    prompt_id: str
    step: int
    policy_version: int
    rewards: tuple[float, ...]     # 组内每个样本
    components: dict[str, tuple[float, ...]]

@dataclass(frozen=True)
class Admission:
    decision: Literal["keep", "drop", "retry"]
    reason: str                    # "zero_std" | "above_band" | "below_band" | "retry_budget" | ...

class GroupAdmission(Protocol):
    def decide(self, outcome: GroupOutcome, history: PromptHistory) -> Admission: ...
```

- **账本** `prompt_outcomes.jsonl`（与 `rollout_stats.jsonl` 同目录）：每组一行
  `prompt_id, step, policy_version, reward_min/max/mean/std, decision, reason`。
  它是难度带、NGU 重试预算、采样权重回流三者共同的数据源，也是 §3 第 3 层的落盘物。
- **默认策略** = 现有 `nonzero_advantage_mask` 搬家（`reason="zero_std"`）。零行为变化，
  trainer 只多写一行账。
- **策略按测量进入**（knob 规则：无 preset 使用或无测量的旋钮不加）：
  1. 难度带（AdaGRPO / Qwen-Image-2.0-RL）：组内极差落在 `[lo, hi]` 才 keep。先在 OCR /
     GenEval 这类接近二元的 reward 上测"只训中等"是否赢过"全训"，带宽是本仓自己的数。
  2. NGU 重试：`SPRINT_ngu_adaptive_sampling.md` 的 `adaptive_sampling.py` 就是这里的
     `retry` 策略，owner 从 continuous producer 移到本块，legacy 路径同样可用。
  3. 回流：账本 → `data.sampler` 的每 prompt 权重（未解决的多抽、已饱和的少抽）。

## 3. 归因框架：四层从下往上排除

一个 prompt 组只有四种可见结果：全 0、全 1、混合、"reward 高但人看是坏的"。
今天前两种被静默丢掉，第四种靠事后看图。

```text
第 1 层  精度      rollout 与 replay 的 log-prob 一致？          已有：replay_parity / first_step
   │ 不一致 → 训练计算的问题，与任务无关，停
第 2 层  verifier  oracle 过所有检查？退化输入（原图原样返回、空图、
                   整图复制 N 份、高频噪声贴图）不过？              缺：oracle 不进环境
   │ oracle 不过 → verifier 或规格坏，任务下线送审
   │ 退化输入过 → verifier 有洞，先补检查再训
第 3 层  任务      同一模型、同一预算下该组是否有混合结果 / 极差在带内？  半有：adv_zero_rate 汇总
   │ 全 0（且第 2 层干净）→ 对当前模型太难，进难题池（NGU 重试或降权），不训
   │ 全 1 → 太简单或被 hack，跑对抗检查
第 4 层  模型      只有前三层干净，混合结果的组才产生梯度；这时的失败才记到模型头上
```

每层的产物都要落盘：

- 第 2 层：verifier 自检报告，每个任务一行（oracle 结果 + 退化输入结果），对应
  CodeMidas 的"6 个新容器：参考解必过、起点必不过"。任务进 manifest 前必须有这一行。
  判别器（VLM）本身也要过 oracle（目标图必 yes、源图必 no）并记一致率——MiMo 的
  "验证器交叉校验"落到这里就是两个判别器在同一批样本上的一致率。
- 第 3 层：§2 的账本。
- hack 观测：0 权重分量（已有）+ 对抗输入分数。SANA aesthetic 的 texture collapse 当时
  靠人看曲线发现；有第 2 层后应在"高频噪声贴图过 PickScore"这一步被提前抓到。

与 `SPRINT_verifiable_visual_environments.md` §0 的关系：那份说"连续奖励下全不合格仍可
有有效优势，不采用二元混合门槛"——同意；本框架第 3 层的判据是**极差在带内**而不是
二元通过率，AdaGRPO 正是这个连续版本。

## 4. 材料

### AdaGRPO：按 ODE 确定性 reward 给 prompt 分难度，中等难度才进训练（优先）

- [arXiv 2606.06828](https://arxiv.org/html/2606.06828v1)，重点 §3（难度剖面）、§4.2
  （easiest / hardest / medium / random 四臂消融）。
- **论文报告**：候选 batch 里每个 prompt 先用确定性 ODE 采样打一次分作难度剖面；只训
  最容易的会严重退化，只训最难的与随机差不多，中等难度收益最大——与 LLM 侧 20–80%
  pass-rate 过滤一致。
- **本仓建议**：§2 策略 1。难度剖面那次 ODE 采样可复用 `eval.sampling`。
- **限制**："中等"的带宽是他们数据上的经验值；多一次采样的成本要计入。

### Qwen-Image-2.0-RL：组内 reward 极差过滤（工业版）

- [arXiv 2606.27608](https://arxiv.org/abs/2606.27608)（[PDF](https://arxiv.org/pdf/2606.27608)）§4：
  "prompt curation via intra-group reward range filtering"。阈值未在摘要给出，读 PDF。
- 同一机制；reward 设计部分留在 reward sprint 的 R16。

### NGU：全失败按概率重采，带预算（本仓已有计划）

- [arXiv 2609.13443](https://arxiv.org/abs/2609.13443)、[作者博客](https://mnoukhov.github.io/posts/ngu/)；
  本仓 `docs/sprints/planned/SPRINT_ngu_adaptive_sampling.md`。§2 策略 2。

### CodeMidas §3.2–3.4：verifier 纪律（背景）

- [arXiv 2609.22068](https://arxiv.org/html/2609.22068v1)。可迁移的四条：verifier 由执行
  落地；先过 oracle（6 容器）；审核 agent 找规格与 verifier 的不一致；只保留混合结果，
  且"全失败 = 太难"只在前两条排除"测试坏了"之后成立。消融：过滤后 3k 胜未过滤 8k。
- 不搬：代码任务的二元判定、自动造任务的 agent 工厂。

### MiMo-V2.6 技术报告精读（2026-09-21；博客只有摘要，机制全在报告里）

- [博客](https://mimo.mi.com/docs/zh-CN/news/latest/v2-6)；
  [技术报告 PDF](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
  （本地副本：scratchpad `mimo_v26.pdf/.txt`）；开源环境
  [HF collection](https://huggingface.co/collections/XiaomiMiMo/mimo-v26)。
  章节：§4.1 算力、§4.2.1 code 任务的监督准确性/鲁棒性、§4.2.3 视觉任务、§4.2.4 cyber
  verifier 原则、§4.2.6 reward hacking、§4.3 groupwise grading、§4.3.3 行为正则、
  §5.4 router 冻结、§6.3 Sample Mixer、§7 Table 5。

**可迁移到本块的机制（按优先级）：**

1. **监督准确性 = 规格与 verifier 对齐，用 rollout 审计找假阳/假阴（§4.2.1）**。每个任务
   跑 4 次；审计 agent 拿到规格、测试、参考解和 4 条轨迹，先写下"正确解需要什么"，再
   逐条判断并与观测 reward 比较：**过测试但判错 → 假阳（verifier 漏）；不过测试但判对 →
   假阴（verifier 过严或环境失败）**。这就是 §3 第 2 层的具体操作：不是只跑 oracle，而是
   拿真实 rollout 让独立判别与 reward 对账。本仓对应：P0 体检集上"reward 高但人判坏 /
   reward 低但人判好"两个桶，按 prompt 落账。
2. **监督鲁棒性 = 重复执行（§4.2.1）**。F2P 在参考补丁前必须失败、P2P 必须通过，补丁后都
   通过，且 **8 次重跑稳定**——筛 flaky。本仓对应：同一样本重复打分 8 次看方差（VLM
   judge 的非确定性就是我们的 flaky test）；oracle 必过、退化输入必不过也要在重跑下成立。
3. **规格与 verifier 共用一个真值源（§4.2.4）**。cyber 任务把漏洞类型 + 崩溃位置从同一份
   sanitizer 报告里抽出来，既生成任务描述又做规则匹配；明确拒绝 LLM judge 当 reward
   ("returns different verdicts for the same PoC across runs and cannot serve as a stable
   reward")，拒绝差分测试（补丁不完整会翻转判决、污染梯度）。本仓对应：局部编辑的遮罩
   既定义任务（编辑什么）又定义检查（遮罩外恒等）；分层的 oracle 层既是数据又是重建检查。
4. **视觉任务分两类、两种打分（§4.2.3）**。开放式设计：先迭代 pointwise rubric（运行正确、
   指令遵循、布局完整、基础审美）**直到稳定，再**加 groupwise 组内比较给相对审美；
   高保真复现：规则相似度（像素级）为主，LLM 整体判断为辅。本仓对应：t2i 审美 = 开放式，
   编辑/分层 = 复现；顺序是 pointwise 先稳、groupwise 后加（与 reward sprint 的 P1→P2 一致）。
5. **确认的 hack 先归零再算组统计（§4.2.6, §4.3.2）**。训练中离线审计轨迹，确认作弊的
   有效 reward 置 0 后再算组均值/advantage；全程确认作弊率 < 2%。**hack agent** 在训练前
   反复探测环境直到找不到漏洞（每轮都发现清理没覆盖的新路径）。本仓对应：§2 的
   `Admission` 加 `reason="confirmed_hack"`，reward 归零在组归一化**之前**；退化输入
   清单就是我们的 hack agent，要迭代到全部不过为止。
6. **乘法门控而不是加权和（§4.3.1）**：`R = R_test · S_sol · S_beh`——rubric 只在通过测
   试的样本之间区分，失败的保持 0；全组都过时 rubric 乘积仍给出信号。本仓对应：可执行
   检查（遮罩外恒等、重建）作硬门控乘在前，VLM/审美分只在过门的样本间排序——直接回答
   reward sprint 里"严重缺陷不能被审美高分抵消"该用什么形式。
7. **组内排序重分配 advantage，守恒且封顶（§4.3.2）**：grader 只看混合结果的组，对通过样本
   按五个维度排名（方法合适、实现精确、改动最小、无副作用、符合惯例），
   `λ = ΣA_j / Σ f_j A_j`，正 advantage 质量从低质通过样本转到高质通过样本，总和不变，
   失败样本不动，再减组均值；grader 异步，输出不可用时回退原 advantage。消融（Figure 8）：
   没有它 pass rate 涨不动、轮数和 token 暴涨；有它 pass rate 持续涨、长度平稳。
   本仓对应：Pref-GRPO 的组内比较落到 advantage 层的一种守恒实现。
8. **动态采样过滤全过/全不过的组（§4.1，DAPO）+ Sample Mixer 按来源的接受率 r_i 和
   rollout 时长 t_i 分配并发（§6.3）**：25 个数据源 token 差 90×、时长差 66×；每源预算
   `(1+p_i)·B_i/r_i`，调度权重 `α·B_i/r_i + (1−α)·(B_i−A_i)⁺/r_i`，α=0.5 最稳。
   本仓对应：§2 账本记录的 `decision` 比例就是 r_i，是 continuous 路径按 manifest 混合
   分配 rollout 预算的输入。
9. **算力分配（§4.1）**：rollout 43.8% / 训练 43.5% / **grader 12.7%**。1,568 prompt × G=16，
   2.7–3.7B token/步。grader 占 1/8 是他们愿意为"更准的 advantage"付的价格。
10. **Table 5 的 verifier 分类**：可执行测试（code 3k）/ 规则检查（cyber 1k）/ rubric 判断
    （general 1k）/ 视觉打分（visual 2k）。本仓的 reward 也应按这四类标注可信度等级。

**不迁移**：mid-training 用作弊案例做"反思改写"对齐数据（LLM 专属）；MoE router 冻结
（§5.4，本仓无 router）；组相对长度惩罚与 segment 级格式惩罚（§4.3.3，token 级）；
partial rollout 与 KV 重 prefill（§4.1）。

### MILES filter_hub（实现参照）

- `~/Desktop/miles/miles/rollout/filter_hub/common_filters.py`：
  `apply_reward_nonzero_std_filter`、`apply_aborted_filter`。位置与本块相同。

## 5. 顺序与门槛

1. 建块 + 默认策略搬家 + 账本（零行为变化；CPU 测试即可）。
2. 用账本回看一次已有 run（`outputs/fullparam_smoke_20260919`、四卡 Wan run）：
   每步被丢的组占比、是否集中在固定 prompt——这是"要不要难度带"的第一份测量。
3. 难度带在 OCR 上做 全训 vs 中等带 的固定预算对照；赢了才加 knob。
4. NGU 按其 sprint 的 OCR 验收，owner 改到本块。
5. verifier 自检（第 2 层）跟着 reward sprint 的 P0 体检集走：退化输入清单就是那里的
   反例集，oracle 就是那里的人工偏好对。

## 6. 不做

- 不在本块里算 reward、不改 advantage 公式、不改 loss。
- 不用"训练 reward 上升"证明过滤有效；证据是固定预算下的留出评测。
- 不把静默丢弃改成静默重试：每个决策必须有 `reason` 且落账。
