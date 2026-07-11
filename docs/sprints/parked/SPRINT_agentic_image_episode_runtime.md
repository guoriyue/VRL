# SPRINT: Agentic Image Episode + Tool Runtime

状态：**parked（2026-07-11）**。

**解 park 事件：**`../planned/SPRINT_janus_r1_agentic_credit_assignment.md` 的 Phase 3 gate 全部通过，证明
中间视觉决策在 compute-matched 条件下能改善 held-out reward。没有这个证据，通用 runtime 只是提前抽象。

## 0. 目标

增加一个视觉专用、有界、可回放的 agent episode 层：controller 可以观察 prompt 与已有 artifact，输出
结构化 tool call，调用已有 image/video generator runtime，接收结果后继续或停止。

核心裁决：**episode loop 位于 `GenerationRuntime` 之上；一次 tool call 仍是一个普通
`GenerationRequest -> GenerationOutput`。** 不改 family executor，不把 loop 塞进 `PipelineTopology`，也不
把现有 one-shot `RolloutCollector` 改成同时服务两种范式。

## 1. 最小使用场景

第一版只支持纯函数式视觉工具，最多两次生成调用：

```text
prompt
  -> controller: generate(tool, prompt_spec, seed)
  -> image artifact
  -> controller: accept | refine(tool, revised_prompt, seed) | stop
  -> optional second image artifact
  -> controller: select(candidate_id) + stop
  -> terminal visual reward
```

不接 web/search、filesystem mutation、任意 Python tool 或真实外部 side effect。image editing 可以作为第二个
真实 tool family 做契约验证，但不能扩大工具安全模型。

## 2. Typed contracts

建议落在 `vrl/rollouts/agentic/`，名字可在实现前小幅调整，但职责必须保持：

```python
@dataclass(frozen=True, slots=True)
class PolicyStamp:
    policy_id: str
    version: int | None

@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    media_type: str
    content_hash: str | None
    storage_ref: str | None

@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, object]
    parent_step_id: str

@dataclass(slots=True)
class ToolResult:
    call_id: str
    artifacts: tuple[ArtifactRef, ...]
    trajectory_ref: str | None
    policy_stamp: PolicyStamp
    metrics: dict[str, object]
    error: str | None = None

@dataclass(slots=True)
class AgentStep:
    step_id: str
    observation_refs: tuple[ArtifactRef, ...]
    controller_action_ref: str
    tool_call: ToolCall | None
    tool_result: ToolResult | None
    reward_facts: dict[str, float]
    done: bool

@dataclass(slots=True)
class AgentEpisode:
    episode_id: str
    prompt: str
    steps: list[AgentStep]
    policy_stamps: dict[str, PolicyStamp]
    selected_artifact: ArtifactRef | None
    termination_reason: str | None
```

这些是**serializable facts**。controller token tensors 与 generator denoise/image-token tensors 继续存在各自的
`TrajectoryBatch` 中，episode 只存引用；Ray actors、KV cache、CUDA tensors、model modules、scheduler queue
不得进入 contract。

## 3. Tool boundary

```python
class AgentTool(Protocol):
    name: str
    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult: ...
```

第一版提供：

1. `GenerationToolAdapter`：校验 arguments，构建 `GenerationRequest`，调用现有
   `GenerationRuntime.generate()`，把 output/trajectory 变成 refs。
2. `FakeImageTool`：确定性、无 GPU，用于 episode state machine、取消、budget、trace 测试。
3. 第二个真实 family adapter：证明 contract 不依赖 Janus 私有字段。

tool registry 应从 config + import path 或现有 family registry 构建。不要在 orchestrator 放一个巨型
`TOOL_CAPABILITIES = {...}` 业务词表；允许的 tool names 是 recipe 输入，schema/version 才是协议常量。

## 4. Episode orchestrator

### 状态机

```text
READY_CONTROLLER
  -> READY_TOOL      （合法 tool call）
  -> READY_CONTROLLER（tool result 变成 observation）
  -> TERMINAL         （stop / success / budget / invalid / runtime error）
```

必须 fail closed：

- `max_controller_turns`、`max_tool_calls`、`max_generated_pixels/frames` 是 recipe budget。
- tool allowlist 与 argument schema 在调用前验证。
- 每个 step/call 有稳定 ID、parent link、seed 与 policy stamp。
- runtime error 变成 typed termination，不留半个“成功” episode。
- cancellation/release 通过现有 runtime lifecycle；episode 不持有 actor handle。

### 并发与负载均衡

单个 episode 内有数据依赖，不能并行执行它自己的下一步；吞吐来自**多个 episode 的 ready queue**。第一版
只需公平地把 ready calls 交给现有 runtime。以后若 profile 证明多个同 family/shape/version calls 让 GPU
underfilled，再复用 `SPRINT_cross_request_step_scheduler.md` 的 gate。

负载均衡因此通常和“有很多 rollout episode / rollout GPU”有关，但不是只有多 GPU 才有意义：单 GPU 上
多个 episode 也能用 queue/backpressure 隐藏 controller/reward 间隙。没有 ready-call 并发时，server-style
runtime 不会凭空加速。

## 5. Policy version 与 replay

episode 记录 version vector，而不是一个全局整数：

```text
controller -> controller@17
flux_tool  -> flux@4
janus_tool -> frozen / None
reward     -> reward-model-id（provenance，不是 trainable policy version）
```

规则：

- 每个 tool call 的 `GenerationRequest.policy_version` 仍只描述该 generator。
- controller trajectory 自带 controller stamp。
- episode manifest 固定所有 stamps；中途权重变化时，新 call 必须被拒绝或显式开启新 episode，不能静默混合。
- 不调用 `stack_trajectory_batches()` 合并不同 family/task 的 trajectory。
- terminal reward 可由后续 credit assigner投影到多个 policy trajectory，但原始 reward fact 只存一份。

## 6. 实施阶段

### Phase 0 — Contract + deterministic fake

- dataclasses/protocol、JSON round-trip、schema validation。
- fake controller + fake image tool 跑 accept、refine、invalid、budget、runtime-error paths。
- episode artifacts/trajectory refs 不泄漏 live runtime state。

**Gate：**相同 prompt/seeds/policy stamps 得到 byte-stable episode manifest；所有 terminal path 有明确原因。

### Phase 1 — One real generator tool

- adapter 调用一个现有 `GenerationRuntime`。
- one-shot output、trajectory、metrics 与直接调用 runtime 一致。
- lifecycle plan 的 acquire/release 行为不被 episode 层复制。

**Gate：**direct call 与 tool-adapted call 的 selected artifact/trajectory parity。

### Phase 2 — Two visual families + bounded queue

- 第二 family 验证 family-neutral tool contract。
- 多 episode ready queue，显式 backpressure、cancellation 与 per-episode budget。
- telemetry：queue wait、tool execution、controller wait、reward wait、calls/episode。

**Gate：**fake stress 下无 starvation；不同 policy version 不共 batch/episode；一个 tool failure 不污染其他
episode。

### Phase 3 — Handoff to controller RL

输出 controller replay 所需 action-token refs、per-step observation lineage、terminal reward 与 group identity。
不在本 sprint 实现 optimizer/GRPO。

## 7. 架构卫生

### 应改

- 新增平行的 `rollouts/agentic` contracts/orchestrator/tool adapters。
- artifact store 增加稳定 ref/lineage（若现有 artifact policy 能满足则复用，不再造 store）。
- orchestration telemetry 增加 episode/call 维度。

### 应保持

- one-shot `RolloutCollector` 与 `OnlineTrainer` fast path。
- `GenerationRuntime` 的 `generate/release/is_colocated` transport contract。
- `PipelineTopology` 的 DAG invariant。
- family-specific 薄 adapters：它们是统一 protocol 与不同 request schema 之间的必要 framework boundary。

### 非目标

- 通用 agent SDK、任意 MCP/web/browser tools。
- 跨请求 token/denoise-step batching；没有 profile 证据前只做 request queue。
- 多策略训练、GAE/value model、reward model training。
- 把 episode 历史全部复制进每个 `GenerationRequest.metadata`。

## 8. 验收

- [ ] deterministic fake 覆盖所有 state transitions 与 terminal reasons。
- [ ] 一个 direct generator call 与 tool adapter 逐位/逐字段 parity。
- [ ] 两个真实 visual families 共用同一 tool protocol，无 family switch in orchestrator。
- [ ] episode trace 可序列化、可限额、不含 runtime state。
- [ ] policy-version vector 和跨 family trajectory isolation 有 fail-fast tests。
- [ ] 输出足以让下一 sprint 训练 controller，但本 sprint 没污染 one-shot trainer。

## 相关文档

- `../planned/SPRINT_agentic_visual_rl_program.md`
- `../planned/SPRINT_janus_r1_agentic_credit_assignment.md`
- `SPRINT_agentic_image_controller_rl.md`
- `SPRINT_cross_request_step_scheduler.md`
