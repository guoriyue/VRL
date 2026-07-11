# SPRINT: Joint Controller + Generator Agentic RL

状态：**parked / high-risk（2026-07-11）**。

**解 park 事件（全部满足）：**

1. `SPRINT_agentic_image_controller_rl.md` 的 frozen-tool RL 在 held-out、compute-matched 评估中通过；
2. reward-oracle 分析显示仍有显著 generator-quality headroom，而不是只剩 selector/reward noise；
3. episode trace 已稳定记录每个 policy 的 trajectory 与 version stamp；
4. 目标硬件能容纳至少一种交替 train/rollout lifecycle，且 weight-sync 不是主导停顿。

## 0. 为什么现在不做

当前 `OnlineTrainer` 假设一个 model、optimizer、algorithm、weight syncer 和单一 rollout policy version。
controller categorical tokens、AR image tokens 与 diffusion denoise actions 又是三种不同分布，不能 stack 成
一个 `TrajectoryBatch` 或用一份 PPO ratio。

在 controller-only 还没证明价值前同时训练 generator，会引入：

- 多 policy staleness 与版本组合；
- terminal reward 到多个调用的长程信用；
- controller 与 generator 共同漂移造成的非平稳性；
- 多模型驻留/换入换出和 optimizer state 显存；
- 无法判断 gain 来自决策还是基础生成质量。

所以联合训练是事件触发项，不是“agentic RL 应该看起来更完整”就开工。

## 1. 第一版裁决：交替更新，不做原子联合 PPO

```text
Epoch A: freeze generator(s), collect episodes, update controller@c
barrier: publish controller@c+1

Epoch B: freeze controller@c+1, collect episodes, update one generator@g
barrier: publish generator@g+1

repeat
```

每个 phase 只有一个 behavior policy 变化，能复用现有 trainer/weight-sync discipline。episode manifest 记录：

```text
{controller: c, generator: g, reward_model: immutable_id}
```

不允许一个 phase 内用 `controller@c` 采前半 episode、`controller@c+1` 采后半，也不允许一个全局整数代表
两个模型的 freshness。

只有交替方案证明稳定并且 profile 显示 barrier 成为主要瓶颈，才讨论多 trainer coordinator 或并行 actor-
learner；第一版不造分布式多策略控制平面。

## 2. 轨迹与 optimizer ownership

```text
controller trajectory
  distribution = categorical text/tool tokens
  optimizer/controller policy version = independent

generator call trajectory
  distribution = categorical AR image tokens OR flow/gaussian denoise
  optimizer/generator policy version = independent

episode manifest
  links both trajectories + artifacts + rewards
  owns no gradients, models, actors, or live caches
```

同一个 episode 可以链接多个 generator call trajectories，但每次 update 只消费同 family、同 version、同
trajectory schema 的 batch。异构工具共同出现时，未被更新的工具视作 frozen environment component。

## 3. Credit assignment

### Controller

沿用 controller sprint 的 terminal outcome、tool cost、invalid/budget penalty。短 horizon 先 episode-level
GRPO；不要为了“多步”自动引入 value head。

### Initial generator call

候选方案：

```text
R_initial = quality(initial)
            + beta * downstream_selected(initial)
```

必须与只用 terminal reward 的 baseline 比较，确认 shaping 没让初始 generator 学会迎合 selector 而牺牲独立
质量。

### Refine/edit generator call

使用进步信号而非把 terminal reward 原样广播到所有调用：

```text
R_refine = quality(refined) - quality(best_artifact_before_call)
```

它近似该工具调用的 marginal contribution。若 reward judge 的 pairwise noise 太大，用 margin/mask，不要把
小数差异当精确信用。

### 多于两次调用

不在第一版支持。只有两步结果稳定后才评估 return-to-go、leave-one-call-out counterfactual 或 learned value。
长 horizon 算法不能先于可验证的两步 credit。

## 4. 实施阶段

### Phase 0 — Offline credit audit（KILL-RISK）

用 frozen-tool controller sprint 已保存的 episodes，离线计算：

- per-call quality delta 与 terminal reward 的相关性；
- 哪类 controller action 真的改变结果；
- generator oracle headroom；
- reward disagreement/tie/noise。

**KILL 0：**如果 refine marginal contribution 大多为零/负，或 residual headroom 主要来自 selector error，
不训练 generator，继续 controller/data/reward 工作。

### Phase 1 — One generator, alternating update

- 只选一个 generator family；其他 tools frozen。
- controller phase 与 generator phase 各自 strict on-policy。
- 每个 optimizer step 验证 consumed trajectory policy stamp。
- 记录 controller/generator reward、KL、clip fraction、version pair 与 artifact drift。

**Gate 1：**旧 policy trajectory 被拒绝；两 phase 各自 old/new log-prob 比率正确；单独关闭任一 phase 能复现
对应 frozen baseline。

### Phase 2 — Joint outcome evaluation

比较：

1. frozen base generator + heuristic controller；
2. frozen base generator + RL controller；
3. RL generator + frozen heuristic controller；
4. alternating controller + generator；
5. one-shot generator RL，使用相同 generator update/compute budget。

**KILL 2：**联合方案必须相对第 2、3、5 项的最强者有 held-out paired gain，且没有更多平均 calls、reward
hacking 或 one-shot quality collapse。否则保留更简单方案。

### Phase 3 — 性能/并发（只在正确性后）

profile 多 episode ready calls、模型换入换出、weight-sync drain。如果 GPU underfill 来自可合并请求，再解
cross-request scheduler；如果停顿来自 model lifecycle，则修 lifecycle。不能因为 agent 有多模型就默认
“需要新 RPC/server”。

## 5. 架构卫生

### 应改

- 增加显式 multi-policy run coordinator，职责仅是 phase/barrier/version manifest，不吞并 trainer。
- credit assignment 按 policy/call 输出 named rewards/advantages。
- checkpoint 保存 controller/generator optimizer 与 version pair 的一致恢复点。

### 应保持

- 每个 trainer 继续拥有一个 model/optimizer/algorithm；不把它泛化成巨型 `dict[str, Any]` trainer。
- 每个 `TrajectoryBatch` 保持单 family/task/distribution 可回放。
- family runtime、thin adapter、reward registry 与 one-shot recipes 保持不变。
- version/schema 常量可以保留；policy 列表、reward 权重、tool taxonomy 属于 config，不做大块 `ALL_CAPS` 数据。

### 非目标

- 同时训练多个 generators、reward model 或三层以上 agents。
- 原子多模型 PPO、跨 policy importance ratio、任意 staleness 容忍。
- search/browser side effects、world-model-as-env、robotics actor-critic。
- 以 LOC 减少为理由合并 controller/generator trainer；一致的单-policy形状更利于 grep/debug/correctness。

## 6. 验收

- [ ] offline audit 证明 generator marginal headroom 存在。
- [ ] controller/generator trajectory、optimizer、policy version 完全隔离。
- [ ] alternating barrier 可恢复、拒绝混合 version episode。
- [ ] per-call credit 与 terminal-only baseline 有 ablation。
- [ ] 联合方案超过 controller-only、generator-only 与 compute-matched one-shot RL。
- [ ] 失败时回到最简单获胜方案，不以增加 agent 轮数补救。

## 相关文档

- `../planned/SPRINT_agentic_visual_rl_program.md`
- `SPRINT_agentic_image_episode_runtime.md`
- `SPRINT_agentic_image_controller_rl.md`
- `SPRINT_cross_request_step_scheduler.md`
