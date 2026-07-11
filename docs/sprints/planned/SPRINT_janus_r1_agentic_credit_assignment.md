# SPRINT: Janus-R1 Agentic Credit Assignment Pilot

状态：**planned（2026-07-11）**。这是 agentic visual RL program 唯一立即开工项。目标不是增加轮数，
而是先让现有 `generate -> observe -> selfcheck -> regenerate/select` 闭环表达真实 action，并把 reward
分给真正造成结果的 segment。

## 0. 结论先行

Janus-Pro-R1 已经是本仓最低成本的 agentic probe，但当前 recipe 只能叫**mechanism-level R1**，不能
据此声称 self-refinement policy 学会了：

- `generate_with_refine()` 的 selfcheck 会看初始图 embedding，并输出 Yes/No。
- `Yes` 选择初始图，`No` 选择 regenerated image。
- `refine_mode=selfcheck` 下仍会先生成第二张候选图，再按 Yes/No 选择，所以第一版是**两候选选择器**，
  不是节省 generator call 的动态停止策略。
- 默认 `selfcheck_text` weight 为 0 且 `train=false`，它没有 RL 梯度。
- collector 只评分 selected/final image；trainer 给所有已训练 segment 同一份 sample advantage。
- 接受初始图时，当前 `final_image` trainable segment 会复制 initial token/path；真正生成过但未被选择的
  refined candidate 不在 trainable trajectory 中。这会重复计数 initial action，并丢掉 counterfactual。

所以本 sprint 首先修**事实与信用**，然后才跑训练。

## 1. 当前链路（grounded）

```text
prompt
  -> initial image tokens + decoded initial image
  -> [prompt, initial image embeddings, selfcheck prompt]
  -> selfcheck text / Yes-No decision
  -> regenerated candidate（当前总会计算，mode != never）
  -> Yes: select initial / No: select regenerated
  -> only selected image enters reward scorer
```

关键代码：

- `vrl/models/ar/janus_pro/model.py::generate_with_refine`
- `vrl/models/ar/janus_pro/runtime.py::JanusProR1ChunkGatherer`
- `vrl/trajectory/builders.py::build_ar_multisegment_trajectory`
- `vrl/rollouts/collector/batch_builder.py::TrajectoryRolloutBatchBuilder`
- `vrl/algorithms/grpo/multisegment.py::MultiSegmentTokenGRPO`
- `vrl/trainers/online/trainer.py::collect_training_batch`
- `vrl/config/presets/base/algorithm/token_grpo_multisegment.yaml`

## 2. 要回答的研究问题

按顺序回答，前一问否定就停止：

1. 同一 prompt/seed 下，initial 与 regenerated candidate 的 reward 谁胜并非恒定吗？
2. reward-oracle 选两者较好者，是否显著超过 best fixed policy（always-initial / always-refine）？
3. 当前 selfcheck 是否已经预测哪个候选更好？
4. 只训练 selfcheck action 后，held-out selection 是否捕获一部分 oracle headroom？
5. 只有 1–4 通过后，分别训练 image segments 是否继续提高 terminal reward？

如果第 2 问不成立，agent 决策没有可赚的 headroom；正确结论是停止，而不是加第三轮。

## 3. 正确的 trajectory 语义

### 3.1 Action segments

把 sampled actions 表达为：

```text
initial_image   — 第一次 image-token policy action
selfcheck_text  — 看到 initial artifact 后的 categorical decision/reasoning action
refined_image   — 第二次 image-token policy action（audit/pilot 中总是生成）
```

把 selection 表达为 deterministic episode fact：

```text
selected_image
selected_candidate = initial | refined
```

`selected_image` 继续作为 `GenerationOutput.output`，保证现有 reward/preview consumer 不变；但它不再伪装成
第三个独立 sampled image action。需要一个短期兼容映射时，应在 gather/builder 边界显式完成并带 deprecation
测试，不能让 `final_image` 同时表示“第二次采样”和“选择后的产物”。

pilot 应用显式的 versioned trajectory schema opt in；现有 `online_r1_grpo_*` recipe 在迁移裁决前保留 legacy
schema。这样可以先证明新 credit 方案，而不把一个研究性语义修正静默施加到已有实验。Phase 3 通过后再决定
迁移旧 recipe，届时必须把曲线不可直接比较写进 release note。

### 3.2 Named reward facts

对同一 reward function 产生：

```text
r_initial
r_refined
r_selected
oracle_choice = argmax(r_initial, r_refined)
selection_correct = selected_candidate == oracle_choice
oracle_regret = max(r_initial, r_refined) - r_selected
delta_refine = r_refined - r_initial
```

接近 tie 的样本用 recipe 中显式的 `decision_margin` 标为 neutral/masked，不让 reward-model 数值噪声训练
Yes/No。margin 是实验配置，不做 module-level `DECISION_MARGIN` 常量。

### 3.3 Segment credit

第一轮训练只启用 `selfcheck_text`：

```text
A_selfcheck = group_normalize(decision_reward)
decision_reward = -oracle_regret
```

这比简单的 ±1 标签保留“选错有多严重”的信息。initial/refined image segments 在这一轮关闭 loss。后续如果
进入 generator phase：

```text
A_initial = group_normalize(r_initial)
A_refined = group_normalize(r_refined - stop_gradient(r_initial))
```

具体 shaping 必须由单元测试锁定；绝不能继续把 `r_selected` 的同一 advantage 无差别广播给三个 segment。

注意：Janus 的 backbone/LoRA 参数可能由 text 与 image segment 共享。因此“selfcheck-only loss”只表示
**信用只从该 segment 进入**，不自动等于 image behavior 参数完全冻结。训练报告必须同时跟踪候选图分布漂移；
若需要严格隔离，再单独评估 adapter/module ownership，不在本 sprint 偷换概念。

## 4. 实施阶段与 KILL gates

### Phase 0 — 无训练 audit（最便宜的 KILL-RISK）

在固定 held-out prompt set 上，以同一批 candidate pairs 计算：

- initial/refined reward 分布、pairwise win/tie rate；
- always-initial、always-refine、random-select、current-selfcheck、reward-oracle；
- paired bootstrap confidence interval；
- 每策略实际 image forwards 与 wall time。

**KILL 0：**若 candidate preference 近乎恒定，或 oracle 相对 best fixed policy 的 paired improvement
置信区间包含 0，则关闭本 sprint 的训练阶段。数据说明“选择”没有价值。

### Phase 1 — Truth-preserving trajectory

- runtime 返回 raw initial/refined candidates、selected artifact 与 selection mask。
- agentic schema 的 gatherer 构建三个真实 sampled segments；selected artifact 为 non-trainable decoded fact；
  legacy schema 在迁移裁决前保持原输出。
- reward view 能按名字取 initial/refined/selected，不改变现有单 reward 默认路径。
- accepted-initial 样本不再把 initial action 复制成一个虚假的 final action。

**Gate 1：**固定 seed 下，重构前后的 `GenerationOutput.output` 与 selected tokens 逐位一致；raw refined
candidate 可回放；trajectory validator 全过。

### Phase 2 — Segment reward/advantage plumbing

- 新增一个有实质逻辑的 Janus-R1 credit assigner，输入 named reward tensors，输出
  `dict[segment_name, advantage]` 与 decision diagnostics。
- `TrainingBatch.advantages` 和 chunk/select helpers 支持 tensor 或同 sample-axis 的 named tensor dict。
- 默认单 segment 与现有 multi-segment recipe 仍走原 tensor fast path，行为逐位不变。
- `MultiSegmentTokenGRPO` 已有 dict advantage consumer；补全 trainer producer 与 mismatch fail-fast。

**Gate 2：**构造相反的 initial/refined reward，断言只有应被奖励的 Yes/No token 获得正 advantage；交换候选
后符号反转。缺 key、shape 不同、neutral mask 错位都必须测试失败。

### Phase 3 — Selfcheck-only RL

- 新 recipe 从现有 aesthetic 或 OCR R1 config 派生，image segment weights 设 0，selfcheck weight 非零。
- 固定 candidate-pair eval set，不用训练 reward 重新生成标签。
- 报告 current/selfcheck-after-RL 与四个固定基线，不只报告训练 reward。

**KILL 3：**held-out selfcheck 必须同时满足：

1. balanced accuracy 明显超过 majority/current baseline；
2. `r_selected` 相对 best fixed policy 的 paired CI 下界大于 0；
3. 至少捕获 25% oracle headroom：
   `(agent - best_fixed) / (oracle - best_fixed) >= 0.25`；
4. invalid/unterminated selfcheck rate 不恶化。

任一失败，不进入通用 agent runtime；先记录“决策信号/模型可学性不足”。

### Phase 4 — 可选 image-policy credit

只有 Phase 3 通过才测试 initial/refined segment 的独立 advantage。必须与 selfcheck-only、现有 shared-terminal-
reward recipe 做 ablation。若 generator 更新提高 candidate reward 却破坏 selector calibration，回退到 controller-only。

## 5. 代码改动面

### 应改

- `janus_pro/model.py`：保留 raw refined action/artifact，并分开 selected artifact。
- `janus_pro/runtime.py`：gather 真实 action segments 与 selection facts。
- `trajectory/builders.py` / reward view helper：支持命名候选 artifact，保持默认 selected view。
- trainer batch/advantage helpers：支持按 segment 的 named advantages。
- config：新增显式 opt-in 的 agentic pilot recipe/schema，默认旧 recipe 不变。
- tests：truth table、replay parity、credit sign、neutral tie、chunk/select/stack。

### 应保持

- `GenerationRuntime`、family registry 与 Janus runner 的 transport/CFG/KV-cache 逻辑。
- 单 segment trainer、diffusion trainer、现有 `online_r1_grpo_*` recipe 的默认行为。
- `MultiSegmentTokenGRPO` 的 per-segment loss consumer 形状；这里只补上游信用 producer。
- reward model 本身冻结；本 sprint 测 policy，不训练 judge。

### 薄边界与常量

- Janus gatherer 保留为薄 family adapter：它承担 chunk-result -> canonical trajectory 的真实协议边界。
- `JANUS_R1_SEGMENTS` 和 trajectory schema version 是模型/回放协议，可以保留为常量；但 reward 权重、
  decision margin、prompt 模板内容属于 recipe/config，不扩成散落的 `ALL_CAPS` 业务表。
- credit assigner 只有在集中候选 reward、mask、normalization 与 diagnostics 这些真实复杂度时独立成模块；不为
  一行字典映射新建空壳。

### 非目标

- 通用 `AgentEpisode`、外部 VLM controller、动态跳过第二次生成。
- 多 GPU load-balancing、跨请求 batching 或 serving API。
- 三轮以上反思、web search、image editor tool。
- 用训练集 reward 上升代替 held-out/counterfactual 评估。

## 6. 验收清单

- [ ] raw initial/refined/selected 三类 artifact 可追溯，sample identity 不变。
- [ ] sampled trajectory 不重复计数 accepted initial action。
- [ ] named rewards 与 segment advantages 有 shape/key fail-fast 测试。
- [ ] 旧 recipe 和单 segment路径回归无变化。
- [ ] audit 表包含 compute-matched baselines 与 oracle headroom。
- [ ] Phase 3 gate 给出明确 GO/KILL；KILL 时不解 park 后续 sprint。

## 相关文档

- `SPRINT_agentic_visual_rl_program.md`
- `../parked/SPRINT_agentic_image_episode_runtime.md`
- `../../NORTH_STAR.md`
