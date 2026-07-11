# SPRINT: Agentic Image Controller RL with Frozen Visual Tools

状态：**parked（2026-07-11）**。

**解 park 事件（全部满足）：**

1. `SPRINT_agentic_image_episode_runtime.md` Phase 0–3 通过；
2. Janus pilot 在 held-out、compute-matched 评估中超过 best fixed selector；
3. 选定的 VLM controller checkpoint 通过 image-observation + sampled-token log-prob replay probe。

## 0. 目标与裁决

训练一个多模态 controller，让它观察 prompt 与候选图片，选择结构化视觉工具调用、修订条件、选择结果并
停止。**visual generators/editors 与 reward model 全部冻结；唯一 RL policy 是 controller。**

这是第一条通用 agentic image RL 曲线。冻结工具是有意的：如果最终质量上涨，可以归因于决策策略；若
失败，也能区分是 controller/reward/episode 问题，而不是多个同时漂移的 generator。

## 1. 当前缺口

本仓不能直接把一个 VLM 名字填进 config 就开训：

- family registry 当前覆盖 visual generators，不含通用 VLM controller runtime。
- `GenerationRequest` 没有 typed image observation 字段。
- AR rollout batch builder 假定输出是 image tensor，并填 `videos=images.unsqueeze(2)`。
- `RolloutCollector` 在一次 generation 后立即打 terminal visual reward，不懂 multi-turn episode。
- `OnlineTrainer` 的 advantage 是 sample-level terminal tensor，不懂 controller turn/tool-cost return。

因此 Phase 0 是兼容性 probe，不允许先建完整 trainer 再发现 checkpoint 无法 replay。

## 2. Controller action space

第一版动作词汇必须小、可校验、可由 token log-prob 精确 mask：

```text
GENERATE {tool, prompt_spec, seed}
REFINE   {candidate_id, tool, revised_prompt, seed}
SELECT   {candidate_id}
STOP     {reason}
```

可以有自由文本 reasoning，但只有结构化 action tokens 进入主 policy loss；reasoning 是否训练做单独 ablation。
tool schema 由 versioned config/schema 维护，不把 prompt template 或 capability taxonomy 写成大段 module-level
`ALL_CAPS` 常量。

第一条 recipe 限制最多 2 个 generator calls、3 个 controller turns。先学会 generate/judge/refine/select，
不做搜索和开放式工具规划。

## 3. Phase 0 — VLM rollout/replay probe（KILL-RISK）

选一个能读取 image artifact、按 HF-style causal logits 回放的开源 VLM checkpoint，验证：

1. prompt + image observation 的 processor 输出可序列化重建；
2. rollout sampled action tokens、mask、old log-prob 可捕获；
3. trainer-side replay 对同 tokens 的 log-prob 在容差内一致；
4. LoRA/全参训练面与 image encoder freeze 规则明确；
5. 单卡或目标 topology 放得下 controller + frozen tool lifecycle。

**KILL 0：**任一 replay identity/processor reconstruction 无法可靠实现，停止该 checkpoint 方案；不能用
字符串 action 后处理绕过 old-log-prob correctness。

交付应是小 probe + 结果记录，不是提前注册一个半成品 family。

## 4. Controller trajectory 与 batch

controller 每一 turn 产生自己的 categorical `TrajectoryBatch`：

```text
family = selected_controller_family
task = agentic_image_control
segments:
  action_text:
    action token ids
    old log probs
    action mask
    replay inputs for prompt + image observations
context:
  episode_id / step_id / artifact refs / tool schema version
```

image pixels/embeddings不重复塞进所有 step context；trajectory 存可重建的 artifact refs，replay loader 按
storage policy 取 observation。processor 输出若是精确 replay 必需且成本可控，作为 typed replay input 保存。

新 controller batch builder 可以复用 `RolloutBatch` 的通用字段并令 `videos=None`，但不得继续调用假定 image
output 的 `_pack_ar_tokens()`。薄 builder 是必要的 trainer-adapter boundary，不应为复用几行代码把 visual
generator 语义塞进 controller。

## 5. Reward 与 credit

### Outcome reward

```text
R_outcome = visual_reward(selected_artifact, prompt)
```

### Decision shaping（有界）

```text
R_total = R_outcome
          - lambda_call * extra_generator_calls
          - lambda_invalid * invalid_actions
          - lambda_budget * forced_termination
```

所有系数属于 recipe，并分别记录原始分量。工具成本 penalty 防止 controller 永远用满 budget，但不能大到
让策略退化成永远 STOP。

第一版用 episode-level GRPO：同 prompt 采样 G 个 controller episodes，以 `R_total` 做 group-relative
advantage，并广播到该 episode 的合法 action tokens。之后只有在长 horizon 数据证明需要时才引入
return-to-go/per-step advantage；不在三 turn 任务上先造 value model。

可选的 pairwise reflection reward 必须来自 candidate-pair oracle（哪张图更好/是否值得 refine），与
terminal visual reward 分开记录，不能用 judge 的自然语言解释直接当未校准标量。

## 6. 训练阶段

### Phase 1 — Behavior bootstrap

纯 RL 前先要求 controller 的 structured-action validity 达到可用水平。可接受两种来源：

- 外部准备并记录 provenance 的小型 SFT/cold-start adapter；或
- grammar-constrained rollout + 极小 action space 已能达到低 invalid rate。

本 sprint 不顺手建设通用 SFT 平台。若需要 SFT，只定义 artifact/import contract 与最小复现实验。

**Gate 1：**未训练/冷启动 controller 在 held-out 上 invalid action < 5%，能在 budget 内结束 > 95%。否则
RL reward 主要在学语法，不进入昂贵 visual rollout。

### Phase 2 — Frozen-tool GRPO

- 同 prompt 采样多个 controller episodes；generator seeds 纳入 episode sample identity。
- generator、reward model、processor frozen；只同步 controller policy version。
- strict on-policy first；没有学习信号前不引入 stale multi-policy complexity。
- 每次 eval 固定 prompt set，但保留随机 seeds 评估稳定性。

### Phase 3 — Compute-matched evaluation

至少比较：

1. one-shot generator；
2. always-refine（同两次调用预算）；
3. random select；
4. best-of-2 by same reward judge（明确是 oracle-like upper baseline）；
5. heuristic critique/rewrite controller；
6. RL controller。

报告 reward、task-specific success、calls/episode、latency、invalid rate、oracle regret；必要时用独立 judge
复核，防止对训练 reward hacking。

**KILL 2：**RL controller 必须在 held-out 上相对最强 compute-matched 非 oracle baseline 的 paired CI
下界大于 0，同时没有以更多平均 tool calls 偷预算。否则不解 park 联合训练。

## 7. 代码边界

### 应改

- 新增一个经 Phase 0 证明的 controller runtime/replay family 或专用 registry。
- episode collector 把多 turn controller trajectories 与 terminal reward 转成 trainer groups。
- controller batch builder/evaluator 支持 multimodal replay input 与 action-token mask。
- config 增加 tool budget、reward components、controller sampling 与 artifact policy。

### 应保持

- visual generator families、`GenerationRuntime`、reward model 都冻结且 API 不变。
- controller trajectory 与 generator trajectory 分开；后者这一 sprint 只做 provenance，不训练。
- one-shot `OnlineTrainer` 与 visual generator AR/diffusion batch builders 不承载 controller 特例。
- family adapter 可以薄，因为它维持统一 replay/runtime 形状和 lazy import boundary。

### 非目标

- generator joint update、reward model update、multi-agent debate。
- web/search/reference retrieval；这是验证 agentic 核心后才可能开的独立数据/安全 sprint。
- 长 horizon value model/GAE、跨模型原子 PPO、异步 staleness。
- 用 wall-clock 更慢但 calls 更多的 agent 与 one-shot-only baseline 做唯一比较。

## 8. 验收

- [ ] VLM processor/replay parity probe 有结果与 fail-fast tests。
- [ ] controller action schema、mask、artifact observations 可精确回放。
- [ ] 冻结工具，只有 controller 参数/版本变化。
- [ ] invalid/budget/stop 行为有指标与单元测试。
- [ ] compute-matched held-out 对比达到 GO/KILL 结论。
- [ ] 只有 GO 才触发 `SPRINT_agentic_image_joint_policy_rl.md`。

## 参考

- [GenAgent](https://arxiv.org/abs/2601.18543) — SFT cold start + pointwise final-image reward + pairwise
  reflection reward 的直接参考。
- [Gen-Searcher](https://arxiv.org/abs/2603.28767) — text/image dual reward；search 工具明确留到后续。
- [ImageEdit-R1](https://arxiv.org/abs/2603.08059) — 冻结视觉专家、训练高层决策的边界参考。
