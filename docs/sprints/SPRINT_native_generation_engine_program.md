# SPRINT PROGRAM：Native generation engine

状态：**active（2026-07-13）**。当前主线是继续构建 wm-infra 自己的
generation/rollout engine；FlashDreams 与 SGLang-Diffusion 是后续可选的模型执行
provider，不是新主引擎，也不是 native path 的替代品。

## 0. 结论先行

wm-infra 已经不是“等待接入某个推理引擎”的 trainer 外壳。仓库已经自己拥有：

- rollout collector 面向的 runtime 协议与显式 activate/offload/shutdown 生命周期；
- serializable worker launch contract、Ray worker/resource ownership 与 chunk dispatch；
- full-sequence denoise loop、SDE action/log-prob trajectory、replay tensor export；
- token-autoregressive prefill/decode/cache scheduling、token log-prob trajectory；
- policy version、versioned weight slots、continuous rollout 的 admission/drain 纪律。

因此当前工作不是把 engine 从零换成 FlashDreams、SGLang 或 vLLM，而是把这套
native engine 继续做完整，并在它下面接可选执行 provider：

```text
wm-infra native generation engine
  ├─ request / plan / Ray lifecycle / policy version       owned here
  ├─ trajectory / log-prob / replay / reward / trainer     owned here
  ├─ native full-sequence denoise execution                        default + oracle
  ├─ native token-autoregressive execution                         default token path
  └─ optional denoise execution providers
       ├─ FlashDreams controlled fork
       └─ SGLang-Diffusion upstream API
```

“自研 engine”不等于立即重写 PyTorch、Diffusers 中每一层 transformer forward。
rollout/control plane 与模型 forward ownership 是两层正交工作。前者已经是 native
并处于 active；后者继续由
[Diffusion native transformer executor](parked/SPRINT_diffusion_native_transformer_executor.md)
的 profiling 触发条件控制，不能因为接外部 provider 就静默解 park，也不能把
Diffusers-backed forward 误写成“没有自研 engine”。

## 1. 当前代码现实

| 边界 | 已有事实 | 本 program 的处理 |
|---|---|---|
| Runtime | `GenerationRuntime` 定义 activate/generate/offload/shutdown 和 policy version | 保留为唯一 collector-facing runtime |
| Distributed launch | `GenerationRuntimeLaunchContract` 只传 primitive config 与 import path，拒绝 live tensor/module/pipeline | 继续作为 worker 构造与进程隔离边界 |
| Worker | `GenerationWorkerCore` 构造 family executor、安装版本化权重、执行 chunk、sleep/wake 与故障清理 | 外部 provider 必须进入此生命周期，不能另起旁路服务 |
| Full-sequence × denoise | `bindings/full_sequence_denoise.DiffusionChunkExecutorBase` 组合 prompt/prepare、共享 denoise loop、trajectory、decode 与 gather | native path 是语义 oracle；provider 只能从明确的 family-executor boundary 接入 |
| Token-autoregressive × token | `bindings/token_autoregressive.ARChunkExecutorBase` 与 `composition.token_autoregressive.TokenAutoregressiveLoop` 拥有 prefill/decode/cache/token trajectory | 保持 native；本轮不做第二个 token-autoregressive full engine |
| Replay | family model 导出 replay tensors/context，trainer 侧按 trajectory schema 重放 | provider 输出必须投影到同一 schema，不能让 trainer 认识 provider 私有对象 |
| Native transformer forward | Wan/Cosmos 当前主要仍调用 Diffusers transformer | 保持事件门控的独立 Layer-B sprint，不在本 program 重复实现 |
| Cross-request step batching | 已有 parked 的 family-neutral `StepScheduler` 方向 | 不建 provider 专用 scheduler；仍由真实 underfill + workload gate 解 park |

当前 `ModelFamilyEntry.policy_semantics` 显式表达 trainable policy 的 generation regime、
step、action distribution 与 trajectory layout；executor/gatherer/provider 由独立 binding 表达。
旧 `collector_kind` 已删除。不要把 FlashDreams/SGLang 名字塞进 semantics，也不要把
一个 native runtime 拆成两个顶层 rollout backend。

native transformer executor 目前不解 park：计划中的 Wan native executor/attention/
weight mapper 尚不存在；独立的 SD3.5 bf16/FA2、**compile-off** full-rollout profile 已把
attention 从旧 fp32 profile 的 34% 降到约 9%。另一组 compile A/B 通过削减 launch/
elementwise 得到收益，Cosmos NCU 又没有找到稳定的单卡无损 kernel 杠杆；这三组证据
不能混写成“bf16 + compile 把 attention 降到 9%”。
当前证据支持继续优化 engine 的 correctness/lifecycle，而不是为了“更像从 scratch”
重写 Diffusers transformer ownership。只有 production shape 的新 profile 证明存在
显著且可兑现的结构瓶颈，或新算法必须拥有 block semantics，才触发 Layer-B sprint。

### 1.1 Evidence-backed ownership audit（2026-07-13）

本 program 用行为消费者判断 ownership，不从目录名或“native”命名反推能力：

- 代码会改变控制流、写入 trainer 消费的数据或在不变量失败时 raise，才算 wm-infra
  已拥有的能力；
- 代码最终调用 upstream module/object，说明 wm-infra 只拥有外层 orchestration；
- sprint 文档里 planned/parked 且生产路径没有消费者的能力，不计入当前 engine。

按这个标准，当前边界是：

```text
outer reliability / provider selection       incomplete
RL lifecycle / policy / trajectory / replay  native and strongest
transformer / kernel / provider scheduling   mostly upstream or parked
```

#### 已经成立的优势

| 能力 | 生产代码证据 | 判定 |
|---|---|---|
| RL trajectory | `generation/steps/denoise/loop.py` 产出 observations/actions/log-probs；binding 与 `trajectory/builders.py` 构造 trainer-facing trajectory，validator 检查 role 与 axis | 这是 trainer-facing source of truth，不是 inference-only artifact wrapper |
| Policy freshness | worker 按 request version 激活 slot；slot 被逐出时 `RayGenerationExecutor` 丢弃整条 request | mixed-policy partial result fail closed |
| GPU/runtime lifecycle | `GenerationRuntime` 暴露 activate/generate/offload/shutdown；Ray runtime 对 activate/offload/shutdown 做 single-flight 与资源清理 | shared-GPU handoff 与 terminal ownership 是 engine 行为 |
| Bounded rollout worker liveness | a background probe watches owned workers out of band | an unreachable worker kills the fleet, closes admission, and hands checkpoint resume to the supervisor |
| Full-sequence denoise + token-autoregressive | denoise step 自有 SDE loop；token-autoregressive composition 自有 position-major bounded row loop与cache row routing；两者汇入同一 `GenerationOutput`/trajectory | 一个顶层 engine 可以保留两种不同数学执行形态 |
| RL group integrity | sample chunk OOM 时有序二分，gather 再检查完整覆盖 | OOM 不会静默少样本、重复样本或重排 GRPO group |

#### 尚未成立的能力与代价

| 缺口 | 当前代码事实 | 已有承载 sprint |
|---|---|---|
| Provider selection | `ModelFamilyEntry` 当前只有一个 `executor_cls`，launch contract 没有 provider identity/schema/provenance | 本 program N2 + [multi-engine conformance](parked/SPRINT_multi_engine_rollout_conformance.md) |
| Full model ownership | diffusion backbone 最终调用 Diffusers transformer；pipeline/scheduler/text encoder/VAE 仍大量来自 upstream | [native transformer executor](parked/SPRINT_diffusion_native_transformer_executor.md)，继续 profile-gated |
| Cross-request batching | full-sequence denoise binding跑完整 denoise loop；token loop只在单 request内按 position和row chunk执行，不存在跨请求 admission/key/cache pool，不同 request不共享 forward | [cross-request step scheduler](parked/SPRINT_cross_request_step_scheduler.md)，继续 workload-gated |
| Video trajectory capacity | denoise 前预分配完整 observations/actions，并可选再分配 previous means/reference predictions | [paged trajectory store](parked/SPRINT_paged_trajectory_store.md)，只在目标 video profile 证明容量/搬运瓶颈后解 park |

因此本 program 的架构结论不是“native 一定比外部 engine 快”，而是替换成本不对称：
替换顶层 engine 会重做 trajectory、replay、policy freshness、resource ownership 与 trainer
handoff；把 FlashDreams/SGLang 放在 execution seam 下方，只需适配 build/update/readiness/
output mapping。正确 ownership 是：

```text
wm-infra owns RL truth and lifecycle
native / FlashDreams / SGLang may own execution behind that contract
```

以下任一事实出现时才重新评估该结论：外部 engine 单独覆盖所有目标 family 且能证明同等
version/replay/lifecycle 语义；provider adapter 的长期维护成本超过第二套顶层 control plane；
production profile 证明 native request scheduling 而非 model compute 是主瓶颈；或新算法必须
直接拥有 transformer block/cache semantics。

## 2. 本 program 的顺序

严格按下面的依赖顺序推进；每一段的退出条件是下一段 wm-infra integration 的入口条件。
唯一允许的重叠是 FlashDreams fork 的 F0–F2 generic primitive：它不触碰 wm-infra runtime，
可以与 Sprint 0 尾部并行；F3 adapter integration 不能越过 Sprint 0 gate。

### Sprint 0 — Native engine contract + oracle（当前）

The reliability gate is complete under
[Rollout worker liveness](done/SPRINT_rollout_worker_liveness.md).
This program keeps that liveness/lifecycle owner and does not duplicate it.

1. 把 request/output、policy version、weight install、failure cleanup、trajectory/replay
   这些 engine-owned 语义钉成 provider-independent contract tests。
2. 以 native full-sequence denoise 与 token-autoregressive 路径作为唯一行为基线。
3. 先覆盖 fake/model-free 与 CPU 数学路径；需要真实模型/GPU 的门在 GPU 可用后单独跑，
   不用 mock 结果冒充通过。
4. 不预建一个只有 native 实现的抽象 provider 层，也不假设所有 provider 都能逐
   denoise step 调用。共同边界先保持现有 `GenerationChunkExecutor` + native trajectory。
5. 当前 contract hygiene 必须在 provider integration 前清零：
   - `tests/architecture/test_generation_rollout_boundaries.py` 全绿；generation 不得反向
     import rollout/trainer 类型。该反向 import 已修复；当前剩余问题是
     `ray/launcher.py` 再次读取 schedule 字符串并比较 `"continuous"`。应由中立
     composition boundary 解析 schedule-derived fact，再把 primitive
     `versioned_weight_sync` 传入 generation；不能保留第二处 schedule 解释；
   - 删除 `GenerationRequest.priority`。全仓生产审计显示它只有赋值，没有 scheduling
     consumer；活的 `RayActorJob.priority` 来自 `assignment.estimated_cost`，是不同概念。
     未来 cross-request scheduler 若真正消费 request admission priority，再以 typed、可测试
     语义重新引入；
   - OOM split/gather 后仍必须覆盖完整 group、保持 prompt/sample 顺序，并且所有 chunk
     结果属于同一个 request policy version。

两个只剩 `__pycache__` 的已删除 package 目录属于 ignored one-shot 生成物：验证前清理同源
缓存即可，不为它们创建 sprint、源码占位或 import compatibility package。

Sprint 0 的 provider-integration reliability gate 只有一个：provider startup、capability、
generation 与 weight-update 的 blocking calls 必须进入 native configured deadline。超时必须
拒绝 partial output、关闭 admission、完成 terminal cleanup，并把失败交给 process
supervisor。一个 isolated real-Ray CPU blocking-actor test 是 production promotion gate；
in-process actor recovery、fleet identity、request retry/replay 与 digest ACK 不是 provider
integration 前置条件。

### Sprint 1 — FlashDreams execution provider

执行
[FlashDreams execution provider](planned/SPRINT_flashdreams_execution_provider.md)：
创建受控 fork，抽出可重放 step-level sampling primitive，再由 wm-infra adapter 接回
native runtime。它优先于 SGLang，因为 causal streaming world-model family 是当前
native model coverage 的真实增量，而不是重复已有 T2I family。

### Sprint 2 — Self-Forcing causal family

执行
[Self-Forcing causal family](parked/SPRINT_self_forcing_causal_family.md)：
在 FlashDreams provider 上过 renoise-logprob、cache replay 与真权重 generation 门。
这是模型/数学 sprint，不再 vendor 一份平行 transformer stack。

### Sprint 3 — SGLang-Diffusion execution provider

执行
[SGLang-Diffusion execution provider](parked/SPRINT_sglang_diffusion_execution_provider.md)：
先使用官方 `POST /rollout/generate` 与 upstream pin；只有 contract 中确实缺少必要
primitive 时才建 tracking fork。它用 full-chunk/process adapter 验证现有共同边界，
不实现 FlashDreams 私有的 step API。首个 pilot 用官方已覆盖的 T2I 路径，不能一开始
就把 Wan/video 支持写成既成事实。

### Sprint 4 — Multi-engine conformance

执行
[Multi-engine rollout conformance](parked/SPRINT_multi_engine_rollout_conformance.md)：
native 是 oracle；FlashDreams 与 SGLang 逐项证明 trajectory、replay、version、
lifecycle 和 failure semantics 一致。只有通过这一门的 provider 才能进入默认/production
training recipe；provider-specific smoke 不依赖第二个 provider 已完成。

## 3. Native engine 当前 sprint 范围

### N0 — 语义 ownership 固化

为以下不变量建立一处测试来源：

- collector 只依赖 `GenerationRuntime`；
- worker 只从 serializable launch contract 构造 executor；
- request 在进入 execution 前已有确定的 policy version；
- 一条 trajectory 不跨 policy version；
- partial weight update 不会把 desired/active version 标成成功；
- provider failure 关闭 admission，并由 native runtime 完成清理；
- rollout output 在进入 trainer 前已经是 native trajectory schema。

These tests describe the wm-infra engine rather than a model-library API.
External provider adapters must pass the same suite instead of copying private
expectations. Blocking-call deadlines, partial-result rejection, and terminal
supervisor handoff are enforced by the
[completed operation-deadline sprint](done/SPRINT_rollout_worker_liveness.md);
external providers must preserve that boundary.

### N1 — 保留两种 provider 粒度

共同边界先使用已经存在的 `GenerationChunkExecutor`：

```text
FlashDreams:
  family executor owns chunk-autoregressive state machine + native trajectory
    -> reuses outer SampleChunk planning/OOM/gather
    -> private step-level FlashDreams adapter predicts flow

SGLang:
  SGLang chunk executor calls one full provider rollout
    -> maps provider response into native chunk/trajectory result
```

这两条路径没有一个诚实的共同 `predict_one_step` 接口。不要为了形式统一，让 SGLang
伪造 step call，或让 FlashDreams 绕过 native family executor/trajectory。只有 build/install/close
等控制动作在两个真实实现中出现相同语义和重复复杂度时，才提取新的小协议；否则继续用
provider 自己的 process/runtime builder + `ModelFamilyEntry.executor_cls` +
`GenerationChunkExecutor` 这三个边界组合。

policy version、Ray placement、admission/drain、trajectory schema、reward 与 replay
仍不进入 provider 私有 API。

### N2 — Provider provenance 与可复现构建

现有 `ModelFamilyEntry` 继续表达 family semantics、trainer binding 与默认 native
executor；不能为了给 Qwen-Image 增加 SGLang 路径而复制一份 family entry，也不能把
provider pin/capability 重复塞进每个 family。

FlashDreams F3 作为第一个真实外部实现时，增加一个 typed provider binding source of
truth：provider provenance（commit/image digest、adapter/wire schema）只构造一次，其 family
bindings 只声明实际实现的 executor/runtime builder/capability。launch 前由
`family entry + explicit provider choice` 派生 `ResolvedGenerationExecution`。已有 native
实现的 family 在未选择 provider 时继续走原 executor；只由外部 provider 实现的新 family
必须显式选择该 provider，缺省时 fail closed，不能伪造一个 native binding。resolved struct
的每个字段必须被 import validation、launch、readiness/schema/digest mismatch 或 capability
rejection 实际消费；否则删除或在定义处标注 display/provenance-only。

config 不能把 provider 名塞进 `policy_semantics`。选择必须在构造
`GenerationRuntimeLaunchContract` 前完成；未注册 family/provider binding、pin 不匹配、
capability 未实现时 fail closed。测试从这一个 typed source 发现 case，不维护平行
`SUPPORTED_PROVIDERS` 常量。

provider 环境策略按 autograd boundary 区分，不能统一写成“都隔离”：

- FlashDreams 是 lazy/optional Python dependency；rollout worker 环境与 trainer replay 环境
  都必须安装同一 immutable fork pin/build，trainer 在本进程构图并 backward；
- SGLang 是 HTTP child runtime，运行环境与 wm-infra trainer 隔离，由 worker-owned process
  adapter 管理。

native-only config/import path 不得 import 任一外部 dependency。构建都借鉴 slime 的“固定
upstream + 可重复应用变更”，但不复制长期膨胀的单文件 patch：通用变更优先 upstream，
小型 tracking fork commit 保持可 cherry-pick，每个 commit 有独立测试。

### N3 — Native path 持续可用

接入 provider 后，native full-sequence denoise 与 token-autoregressive 路径仍必须：

- 无外部 engine dependency 即可 import、resolve config 与运行 CPU tests；
- 保持默认选择，除非 recipe 显式选择一个已经通过 conformance 的 provider；
- 能对相同 seed/schedule 生成 oracle trajectory，供 provider parity 调试；
- 不因 provider 的 HTTP/process 生命周期改变 native Ray worker 的终止语义。

## 4. 架构卫生

### 应改变

- 第一个真实外部实现落地时，从 native 与外部两条路径提取最小 execution seam。
- 把 provider source pin、adapter schema、capability 与 family binding 放在独立 typed
  provider manifest；validation 从该 source of truth 派生，family entry 不复制 provenance。
- 把 provider 私有 response 映射为 native typed trajectory；不把松散 metadata 当长期
  wire protocol。

### 保持不变

- `GenerationRuntime` 保持薄，因为它是 collector/public transport boundary。
- `GenerationChunkExecutor` 保持薄，因为它是 family execution 与 distributed gather
  的协议边界。
- `GenerationRuntimeLaunchContract` 保持独立，因为 pickle/process boundary 必须拒绝
  live model 与 pipeline。
- `build_family_runtime_bundle` 与注册的 replay runtime builder 即使很薄也保留：它们提供
  lazy import、worker-side construction 与跨 family 一致形状。
- `PolicySemantics` 保持 typed schema boundary；它不表达 provider。旧
  `CollectorKind` 已删除，不再维护第二套扁平 taxonomy。

### ALL_CAPS 与数据来源

- 不新增手写的 `SUPPORTED_PROVIDERS`、`SUPPORTED_MODELS` 或 capability 大表。
- provider 名称和能力从实际 typed provider binding 派生；新增实现只有一个 construction
  site。
- 现有 `FAMILY_REGISTRY` 保留：它是刻意集中的 family taxonomy/source of truth，不是
  workflow 中重复维护的业务大表。provider 与 family 是可选的多对多执行关系，因此用
  独立 typed binding 表达；它不能复制 family semantics，也不能再派生平行常量。
- schema key、环境变量名、immutable commit/digest 字段可以保留为常量，因为它们是真实
  protocol/config boundary。

### 非目标

- 不重写 trainer、reward、trajectory 或 Ray runtime 来迎合外部 API。
- 不做第二套 token-autoregressive rollout engine；该 binding 继续走 native engine，vLLM 只可作为明确的 attention
  kernel/backend 边界。
- 不提前解 park native transformer executor、cross-request `StepScheduler`、physical
  pipeline 或 NCCL weight transport；各自已有独立 gate 与 sprint。
- 不为了 LOC 减少 flatten protocol adapter 或 family facade；跨 family 一致性比省几行更重要。
- 不把 FlashDreams WebRTC/serving control plane 或 SGLang HTTP schema 提升为 wm-infra
  public API。

## 5. Program Definition of Done

- [ ] Native full-sequence denoise 与 token-autoregressive 均通过统一 runtime/lifecycle/version contract。
- [ ] Generation architecture boundary 全绿；`vrl/generation/` 不反向 import rollout/trainer。
- [ ] `GenerationRequest` 每个公开字段都有非日志行为消费者；删除当前没有生产
      consumer 的 `priority`。
- [ ] OOM degradation 后 group coverage/order 与 request policy version 仍完整一致。
- [ ] Native full-sequence denoise trajectory 是 provider conformance 的明确 oracle。
- [ ] FlashDreams provider 通过 Self-Forcing collect → replay round-trip。
- [ ] SGLang provider 通过一个官方支持 T2I family 的 collect → replay round-trip。
- [ ] 两个 provider 的 config 不能声明未实现 capability；validation 来自同一 typed source。
- [ ] 任一 provider weight update 只有在全部 worker ACK 后才推进 active policy version。
- [ ] provider 崩溃、超时、partial update 与 shutdown 均 fail closed，无 orphan process/actor。
- [ ] recipe 不选择外部 provider 时，native path 的 import、config resolve 与行为不变。
- [ ] 真 GPU parity/performance 结果记录后才能把 provider 标成 production-ready。

## 6. Negative result / rollback

- FlashDreams step API 如果无法保持 upstream generate/finalize parity，则停止 provider
  integration；保留 native engine，不在 wm-infra 复制其私有 scheduler。
- SGLang 官方 API 如果无法提供 replay 所需数据或版本一致性，则停在 experimental；
  先提交最小 upstream change，不能在 adapter 用猜测重建缺失语义。
- 外部 provider 若只在速度上持平或更慢，不删除实现，但不进入默认 recipe；它仍可作为
  特定 model coverage provider。
- 任一 provider 的失败都不改变 native engine 的主线状态。

## 参考

- `vrl/generation/protocols.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/execution/worker.py`
- `vrl/generation/ray/`
- `vrl/generation/bindings/full_sequence_denoise/executor.py`
- `vrl/generation/bindings/token_autoregressive/executor.py`
- `vrl/generation/steps/denoise/loop.py`
- `vrl/generation/composition/token_autoregressive/token_loop.py`
- `vrl/models/steps/denoise/base.py`
- `vrl/models/steps/token/base.py`
- `docs/MODEL_TAXONOMY.md`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
- `docs/sprints/reading/SPRINT_diffusion_rollout_system.md`
- `docs/sprints/info/SPRINT_ray_generation_engine_map.md`
- `docs/sprints/info/SPRINT_rollout_performance.md`
- `docs/sprints/done/SPRINT_gemm_utilization.md`
- `docs/sprints/parked/SPRINT_diffusion_native_transformer_executor.md`
- `docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`
- `docs/sprints/parked/SPRINT_paged_trajectory_store.md`
- `docs/sprints/parked/SPRINT_weight_sync_transport_seam.md`
- `docs/sprints/done/SPRINT_slime_overlap_strategy.md`
