# SPRINT PROGRAM: Agentic Visual RL

状态：**planned / direction-decided（2026-07-11）**。第一项可执行工作是
`SPRINT_janus_r1_agentic_credit_assignment.md`；后续通用 runtime、外部 controller 和联合训练均按
各自的事件 gate 放在 `parked/`，不能平行开工。

## 0. 结论先行

本仓值得做 agentic RL，但要把它定义成**视觉生成策略在 episode 中观察中间产物、做语义级决策、调用
生成/编辑工具，并从后续视觉结果获得信用**，而不是把现有的去噪 step、静态 pipeline 或“多跑一次
generator”重新命名成 agent。

最短可信路线不是先造通用 agent framework，而是：

```text
Sprint 0（planned）
Janus-R1 现有闭环审计 + 正确信用分配
    │  gate: 自检决策在 held-out 上确实改善最终图
    ▼
Sprint 1（parked）
通用 AgentEpisode / ToolCall / ToolResult 契约与有界 orchestrator
    │  gate: 两种 generator tool + fake tool 的可复现 episode
    ▼
Sprint 2（parked）
冻结视觉工具，只训练多模态 controller
    │  gate: compute-matched 基线之上有显著收益
    ▼
Sprint 3（parked）
controller + generator 多策略信用分配与交替训练
```

这条路线服务 `docs/NORTH_STAR.md` 的同一个赌注：**visual-rl 是视觉生成式 RL 的专用引擎**。
它不是把仓库改成通用 LLM agent 平台，也不是把 generation runtime 改成 HTTP serving 产品。

## 1. 什么才算 agentic RL

一个 rollout 至少同时满足以下四条，才在本计划中叫 agentic：

1. **有 episode 状态**：后一个决策看得到前一步的图片、评分、工具结果或结构化历史。
2. **有可选择的语义动作**：例如 accept / refine / edit / select / stop，或选择工具与参数；不是固定执行的
   denoise timestep。
3. **动作会改变后续计算或最终选择**：静态 `generate -> score` 管线不算。
4. **RL 信号训练的是决策策略**：至少一个 controller/decision segment 的 log-prob 接到了与其决策因果
   相关的 advantage。

| 机制 | 是否算 agentic | 原因 |
|---|---:|---|
| diffusion 的 20 个 denoise step | 否 | 它们是一个生成动作内部的数值轨迹，没有语义选择 |
| 固定 `generate -> reward` | 否 | contextual bandit，没有中间观察后的第二次决策 |
| 固定 `generate -> regenerate` | 否 | 有多次模型调用，但没有可学习的分支/停止策略 |
| Janus `image -> selfcheck -> select/refine` | **候选** | 已有观察与决策，但当前 selfcheck 没训练，信用也不正确 |
| VLM 根据候选图选择 `refine`、参数和 `stop` | 是 | 多轮 observation/action/tool-result 闭环 |

## 2. 当前代码现实

### 已有，可复用

| 现成资产 | 代码位置 | 在 agentic RL 中的角色 |
|---|---|---|
| runtime transport boundary | `vrl/generation/protocols.py::GenerationRuntime` | 每次 generator/tool 调用；底层可本地或 Ray actor |
| one-call request/output | `vrl/generation/types.py` | 单个工具调用的 payload，不升级为整个 episode |
| serializable policy trajectory | `vrl/trajectory/types.py` | 保存某一个 policy/family 的 action、old log-prob、replay input |
| multi-segment AR replay/loss | `vrl/trajectory/builders.py`、`vrl/algorithms/grpo/multisegment.py` | Janus 最小闭环的训练地基 |
| Janus 三段闭环 | `vrl/models/ar/janus_pro/model.py::generate_with_refine` | 第一条可证伪 agentic pilot |
| policy-versioned rollout orchestration | `vrl/rollouts/orchestration/` | 未来每个 trainable policy 的版本纪律 |
| family-neutral generator registry | `vrl/families/registry.py` | tool adapter 选择 generator family 的来源 |

### 缺失，不能假装已有

- 当前 `RolloutCollector` 是一次性的 `request -> generate -> reward -> RolloutBatch`，不是 episode loop。
- 当前 `OnlineTrainer` 生产一份 sample-level advantage；算法虽能接 `dict[segment, advantage]`，trainer
  尚未生产它。
- 当前 `GenerationRequest` 只有单个 `policy_version`，没有 controller + 多工具的版本向量。
- 当前 tree 没有通用 `vrl/rollouts/envs` agent 契约；`SPRINT_world_model_as_env.md` 引用的是另一分支。
- 当前 `PipelineTopology` 是有意无环的 DAG；agent loop 是循环，不能硬塞成 cyclic pipeline。
- 当前没有可训练的通用 VLM/text controller family，也没有 controller 专用 reward-to-replay builder。

## 3. 四个 sprint 的边界

### Sprint 0 — Janus-R1 agentic credit assignment（现在做）

先修正现有三段轨迹的事实表达，分别评分 initial/refined/selected artifacts，训练 selfcheck decision，和
always-initial / always-refine / random-select / reward-oracle 做配对比较。它回答唯一值得先问的问题：
**中间视觉判断是否真的能带来额外 reward？**

详见 `SPRINT_janus_r1_agentic_credit_assignment.md`。

### Sprint 1 — Agent episode + tool runtime（过门再做）

在 `GenerationRuntime` 之上增加有界 episode orchestrator；一个 episode 可以发多个独立
`GenerationRequest`，但 runtime、family executor、collector 的单次调用语义不改。episode trace 只保存
artifact/trajectory 引用和 policy stamps，不保存 Ray handle、KV cache 或模型对象。

详见 `../parked/SPRINT_agentic_image_episode_runtime.md`。

### Sprint 2 — Frozen-tool controller RL（runtime 过门再做）

先冻结 image generator/editor/reward，只训练 VLM controller 的 tool-call / select / stop tokens。这样把
“agent 是否学会决策”与“generator 自身是否变好”分开，能给出因果清楚的负结果。

详见 `../parked/SPRINT_agentic_image_controller_rl.md`。

### Sprint 3 — Joint multi-policy RL（controller 证明价值后再做）

controller 与 generator 保持不同 trajectory、optimizer 和 policy version；第一版采用交替更新，不做一个
巨型异构 batch 或原子多模型 PPO。只有当 controller-only 留下明确的 generator-quality headroom 时才解
park。

详见 `../parked/SPRINT_agentic_image_joint_policy_rl.md`。

## 4. 系统裁决

### 4.1 Agent loop 在 orchestrator，不在 engine/pipeline

```text
AgentEpisodeOrchestrator
  ├─ controller turn -> controller runtime/replay
  ├─ tool call       -> GenerationRuntime.generate(...)
  ├─ observation     -> artifact reference + bounded metadata
  └─ stop/budget     -> final reward + per-policy trajectory refs
```

`PipelineTopology` 继续表达**一次工具调用内部**的 DAG stage；episode orchestrator 表达多轮循环。两个层级
不要合并。

### 4.2 Server-style execution 是实现手段，不是产品方向

agent episode 内部通常是顺序依赖，但训练时有很多 episode 同时处于 ready 状态。负载均衡发生在：

```text
多个 rollout episode 的 ready tool calls
        -> 按 family / shape / policy_version 分桶
        -> 分派到一个或多个 rollout GPU
```

所以 generator 适合藏在可排队的 runtime/actor 服务后面，而不是 controller 直接持有 Python model
engine。这让多个 episode 共用 GPU、做 backpressure 和故障隔离。**但第一阶段不需要新 HTTP/RPC API，
也不需要先做跨请求 continuous batching**；现有 Ray-backed `GenerationRuntime` 已是正确 transport seam。
真正出现多个兼容 ready calls 且 profile 证明 GPU underfill 后，再解
`../parked/SPRINT_cross_request_step_scheduler.md`，不要复制一个 agent 专用 scheduler。

### 4.3 一条 trajectory 只属于一个 policy/family

`TrajectoryBatch` 的 stack 契约要求同 family/task/axes/segments。controller token、AR image token 和
diffusion denoise action 不能拼进同一个 batch。episode manifest 负责关联：

```text
episode_id
  controller trajectories: [trajectory_ref@controller_version]
  tool trajectories:       [trajectory_ref@generator_version]
  artifacts:               [initial, refined, selected]
  rewards:                 [step rewards, terminal reward, tool cost]
```

这既保持 replay 数学正确，也避免用一个含糊的全局 `policy_version` 掩盖多模型 staleness。

## 5. 统一评估协议

任何 sprint 都不能只报“最终 reward 涨了”。至少报告：

- **质量**：terminal reward、初始到最终的 paired delta、任务成功率。
- **决策**：accept/refine/select 的 balanced accuracy、oracle regret、stop calibration。
- **计算**：每个 episode 的 generator calls、denoise/decode forwards、controller tokens、wall time。
- **稳定性**：invalid tool-call rate、max-step termination rate、policy-version rejection rate。
- **基线**：one-shot、always-refine、random selector、best-of-N、reward-oracle upper bound。

最关键比较必须 compute-matched。例如两次生成的 agent 不能只和一次生成 baseline 比；它至少要和
always-refine、random-select 与 best-of-2 比。

## 6. 架构卫生：改什么、保留什么

### 改

- 新增 episode/tool 的 typed contract 和显式 credit assignment，而不是把它们塞进 `metadata`。
- 让多 segment / 多 policy reward 与 advantage 有名字、有归属、可独立回放。
- 用 episode manifest 记录 policy-version vector 与 artifact lineage。

### 保持不变

- `GenerationRuntime` 保持单次调用 transport boundary。
- `RolloutCollector` 保持 one-shot visual bandit fast path；agentic collector/orchestrator 平行新增。
- family executor、trajectory validator、reward model registry 和现有 diffusion/AR trainer 路径保持兼容。
- `PipelineTopology` 保持 DAG；跨 family 的薄 tool adapter 保留，因为它是协议/框架适配边界。

### 常量与薄文件规则

- tool 名称、schema version、协议字段可以是 module-level 常量，因为它们是真正的 wire/schema boundary。
- prompt 模板、工具能力大表、backend taxonomy 不得作为巨型 `ALL_CAPS` 数据混进 orchestrator；放入
  recipe/config 或 family registry。
- 每个 family 的薄 adapter 只有在维持统一 tool protocol、lazy import 或现有 family 形状时保留；不要为
  少几行而把它们摊平，也不要创建只转发且没有边界价值的新文件。

### 非目标

- 通用网页/代码 agent 平台、任意 side-effect tool marketplace。
- 把 world-model denoise trajectory 当 agent MDP；后者仍由
  `SPRINT_world_model_as_env.md` 单独处理。
- 第一阶段同时训练 controller、generator、reward model。
- 为 agentic 标签重写 serving、Ray 或跨请求 scheduler。

## 7. Program 完成标准

- [ ] Janus pilot 证明或否决“视觉自检 + 选择”在 compute-matched 条件下的增益。
- [ ] 若通过，episode/tool contract 能在 fake runtime 与至少两个真实 visual family 上稳定复现。
- [ ] controller-only RL 在 held-out prompt 上超过固定策略与 compute-matched selection baseline。
- [ ] 每个 trainable policy 都有独立 old-log-prob、version stamp、advantage 和 replay path。
- [ ] 负结果会关闭下游 sprint，而不是用更复杂的 agent loop 掩盖。

## 参考

- [GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning](https://arxiv.org/abs/2601.18543)
  — controller 把 image generators 当工具，使用最终图 pointwise reward 与反思 pairwise reward。
- [Gen-Searcher: Reinforcing Agentic Search for Image Generation](https://arxiv.org/abs/2603.28767)
  — search-grounded generation，SFT 后用 text/image 双重 reward 做 agentic GRPO。
- [ImageEdit-R1: Boosting Multi-Agent Image Editing via Reinforcement Learning](https://arxiv.org/abs/2603.08059)
  — 冻结的专用视觉工具之上训练高层协作/决策，是 Sprint 2 的边界参考。
