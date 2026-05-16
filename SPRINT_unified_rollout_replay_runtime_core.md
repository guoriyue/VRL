# SPRINT：统一 Rollout / Replay Runtime 核心立足点

状态：proposed。

## 核心结论

这个 repo 的立足点不应该是“包了很多生成模型”，而应该是：

```text
面向多模态生成模型在线 RL 的 trajectory-aware rollout / replay runtime。
```

核心 invariant：

```text
family model 只适配模型私有的 forward / state / loading 逻辑。
rollout、replay、trajectory routing、signal construction、scheduling、weight sync 必须由 VRL 主路径统一拥有。
```

如果新增一个模型 family，需要改的范围应该收敛到：

```text
vrl/models/diffusion/<family>/model.py
vrl/models/diffusion/<family>/runtime.py
```

或：

```text
vrl/models/ar/<family>/model.py
vrl/models/ar/<family>/runtime.py
```

不应该改：

```text
vrl/trainers/
vrl/algorithms/
vrl/engine/trajectory/
vrl/distributed/ray/rollout/
```

除非这个 family 暴露了一个真正新的 generation lane，而不是一个普通新模型。

## 当前代码状态

关键路径已经基本成型：

```text
vrl/models/ar/
  Janus / NextStep concrete adapters

vrl/models/diffusion/
  DiffusionModelBase
  SD3.5 / Wan / Cosmos concrete adapters

vrl/engine/ar/
  AR decode loop
  active row scheduling
  KV cache row abstraction

vrl/engine/diffusion/
  diffusion executor
  denoise layout / gather / SDE window

vrl/engine/trajectory/
  TrajectoryBatch
  TrajectorySegment
  TrajectoryResolver
  TrainingView

vrl/models/interfaces/
  RuntimeBundle
  RuntimeModel
  ReplayModel
  ReplayResult

vrl/rollouts/evaluators/
  ReplayResult + TrajectoryBatch -> TrajectorySignalBatch

vrl/trainers/online/
  OnlineTrainer consumes evaluator output and algorithm input
```

现在最大的问题不是模型支持不够多，而是主路径 contract 还不够硬：

- 还没有 fake family contract tests 证明新增 family 不需要改 trainer / algorithm / Ray runtime。
- `Trajectory` 已经是重要对象，但还没有被文档和测试明确成系统 IR。
- evaluator / signal builder 虽然存在，但仍需要更强约束，防止 family 自己拼训练信号。
- engine lane 的性能职责需要可观测 benchmark，不然容易退化成 helper / glue。
- 没有 import boundary test 防止 concrete family 反向污染 trainer / algorithm / engine core。

## 非目标

本 sprint 不做：

- 不新增真实模型 family。
- 不追求 CUDA kernel、custom attention、compiler 级优化。
- 不重写 OnlineTrainer 或全部 algorithm。
- 不改变 GRPO / DPO / DiffusionNFT 数学语义。
- 不把 AR 和 diffusion 强行压成同一个 engine loop。
- 不把 family registry 先做成完整插件系统。
- 不继续做纯命名清理，除非它直接服务 contract 或 benchmark。
- 不把 concrete model 私有 state 放进 `TrajectoryBatch`。

## 目标数据流

本 sprint 要把唯一主路径写清楚，并用测试固定：

```text
Prompt / Request batch
  -> RuntimeBackend.rollout(...)
  -> OutputBatch
  -> TrajectoryBatch
  -> RolloutBatch
  -> RuntimeModel.replay_forward(...)
  -> ReplayResult
  -> TrajectorySignalBatch
  -> AlgorithmInput
  -> Algorithm.compute_loss(...)
```

边界语义：

```text
TrajectoryBatch
  rollout record / source of truth

ReplayResult
  current policy replay output

TrajectorySignalBatch
  algorithm-ready signal:
  current log_prob + old log_prob + mask + optional ref log_prob + reward / advantage view

Algorithm
  不理解 SD3 / Wan / Cosmos / Janus / NextStep 私有字段。
```

补充约束：

```text
Algorithm 可以理解 objective family，例如 TokenGRPO 理解 token log-prob，
DiffusionNFT 理解 diffusion NFT loss。

Algorithm 不应该理解 rollout storage、TrajectoryResolver、family replay layout、
RolloutBatch plumbing，或 metadata 里的 model / batch / timestep 后门。
```

以 `DiffusionNFT` 为例，合理边界是：

```text
允许：
DiffusionNFT 负责 diffusion-specific objective math。

禁止：
DiffusionNFT 直接读取 RolloutBatch、TrajectoryResolver、denoise segment 名，
或通过 AlgorithmInput.metadata["rollout_batch"] / ["model"] / ["timestep_index"]
绕过 signal/input contract。
```

如果 DiffusionNFT 需要不同于 log-prob 的训练输入，应该由 trainer-side
signal/input builder 先构造 typed diffusion training input，再交给 algorithm。
这不改变 DiffusionNFT 的数学语义，只把 rollout storage extraction 从
algorithm 中移出去。

## 设计原则

### 1. Trajectory 是系统 IR

`TrajectoryBatch` 必须是 rollout record 的唯一 source of truth。

允许保存：

```text
actions
old log_prob
masks
reward views
replay input tensors
serializable context / metadata
segment / axis schema
```

禁止保存：

```text
KV cache handle
diffusers pipeline object
scheduler mutable object
Ray actor handle
CUDA graph handle
torch generator mutable state
```

### 2. ReplayResult 只表达 current policy output

`ReplayResult` 不能混入：

```text
old log_prob
reward
advantage
mask source of truth
trajectory source of truth
```

这些必须由 `TrajectorySignalBuilder` 从 `TrajectoryBatch` 和 `ReplayResult` 组合出来。

### 3. Family adapter 不能拥有训练主路径

family adapter 可以拥有：

```text
model loading
private sampling state
forward_step / replay_forward
runtime bundle construction
family-specific request kwargs
family capability template selection
```

family adapter 不应该拥有：

```text
algorithm loss assembly
advantage construction
trainer replay loop
Ray weight sync protocol
generic trajectory slicing
generic signal schema
```

### 4. Engine lane 必须有真实调度职责

`vrl/engine/ar` 的职责：

```text
prefill / decode split
active row scheduling
KV cache row transport
decode chunking
token-axis progression
AR replay shape contract
```

`vrl/engine/diffusion` 的职责：

```text
denoise step scheduling
SDE window selection
diffusion request layout
chunk gather
latent / timestep replay slicing
diffusion memory counters
```

engine 不应该 import concrete family。

### 5. Benchmark 是核心产物

没有 benchmark，runtime 只是 glue。

本 sprint 需要建立最小 benchmark schema，让每次改动能回答：

```text
rollout samples/sec
replay steps/sec
GPU peak memory
trajectory bytes
reward artifact bytes
Ray worker startup / resident cost
weight sync latency
AR decode forwards saved by KV cache
diffusion denoise replay memory
```

## Sprint 产物

本 sprint 不新增 architecture docs。核心 thesis 只保留在这份 sprint 里，落地方式必须是测试、代码边界和 benchmark。

### 产物 1：Fake family contract tests

新增：

```text
tests/contracts/test_fake_diffusion_family_contract.py
tests/contracts/test_fake_ar_family_contract.py
tests/contracts/fakes/
```

Fake diffusion family 必须覆盖：

```text
register fake diffusion family
build RuntimeBundle
build collector
rollout emits TrajectoryBatch
replay_forward returns ReplayResult
TrajectorySignalBuilder emits TrajectorySignalBatch
algorithm consumes AlgorithmInput(signals=...)
```

Fake AR family 必须覆盖：

```text
register fake AR family
run AR decode loop through engine lane
emit token trajectory
replay token log_prob
build token TrajectorySignalBatch
run TokenGRPO loss path
```

测试要求：

- 不依赖真实 SD3 / Wan / Cosmos / Janus / NextStep。
- 不依赖 GPU。
- 不启动真实 Ray cluster。
- 不 mock 掉 trajectory / replay / signal 主路径。
- fake model 的 tensor shape 足够小，但必须走真实 contract。

验收标准：

```text
新增 fake family 时，测试不得修改 trainer / algorithm / engine trajectory / Ray rollout 主路径。
```

### 产物 2：Import boundary tests

新增：

```text
tests/architecture/test_import_boundaries.py
```

规则：

```text
vrl/engine/ 不允许 import vrl.models.ar.<family> 或 vrl.models.diffusion.<family>
vrl/algorithms/ 不允许 import concrete family modules
vrl/trainers/ 不允许 import concrete family modules
vrl/distributed/ray/rollout/ 不允许 import concrete family modules
vrl/models/ar/<family>/ 可以 import vrl.engine.ar 和 vrl.models.interfaces
vrl/models/diffusion/<family>/ 可以 import vrl.engine.diffusion 和 vrl.models.interfaces
```

允许例外：

```text
vrl/rollouts/family_registry.py
vrl/scripts/<family>/train.py
tests/
```

验收标准：

- architecture test 能在普通 `pytest` 中运行。
- 新增错误 import 时测试失败，并打印具体文件路径和 import 行。

### 产物 3：Signal path hardening

目标：

```text
Algorithm 只消费 TrajectorySignalBatch 或 AlgorithmInput.signals。
Evaluator 是 ReplayResult -> TrajectorySignalBatch 的唯一组合边界。
```

需要检查并收紧：

```text
vrl/rollouts/evaluators/types.py
vrl/rollouts/evaluators/trajectory.py
vrl/rollouts/evaluators/ar/
vrl/rollouts/evaluators/diffusion/
vrl/algorithms/trajectory.py
vrl/algorithms/grpo/
vrl/algorithms/diffusion_nft.py
vrl/trainers/online/trainer.py
```

要求：

- `ReplayResult` 不带 old log_prob / mask / advantage。
- `SegmentSignal` 或 `TrajectorySignalBatch` 是 algorithm-ready 的唯一 signal。
- trainer 不手动理解 family segment 私有字段。
- algorithm 不直接读 `TrajectoryResolver`。
- algorithm 不通过 `AlgorithmInput.metadata` 接收 `model`、`rollout_batch`、
  `timestep_index` 这类训练主路径对象。
- diffusion-specific algorithm 可以保留 diffusion objective math，但必须从
  typed training input / signal 读取张量，而不是自己解析 rollout storage。

验收标准：

- 增加测试覆盖 AR token、AR multisegment、diffusion timestep 三种 signal。
- 任何 evaluator 返回非 `TrajectorySignalBatch` 时 trainer 保持 hard fail。
- algorithm input 类型检查不接受裸 `ReplayResult`。
- contract / architecture test 能抓出 algorithm 直接 import
  `TrajectoryResolver` 或读取 `AlgorithmInput.metadata["rollout_batch"]` 这类后门。
- `DiffusionNFT` 的迁移允许分阶段进行，但 sprint 完成时必须明确剩余后门是
  temporary compatibility，而不是新的长期 contract。

### 产物 4：Runtime benchmark harness

新增：

```text
benchmarks/rollout_runtime/
  bench_fake_ar.py
  bench_fake_diffusion.py
  metrics_schema.py
```

最小输出 JSON schema：

```json
{
  "family": "fake_ar",
  "engine_lane": "ar",
  "samples_per_second": 0.0,
  "replay_steps_per_second": 0.0,
  "trajectory_bytes": 0,
  "reward_artifact_bytes": 0,
  "peak_cuda_bytes": null,
  "weight_sync_seconds": null,
  "engine_counters": {}
}
```

要求：

- fake benchmark 默认 CPU 可跑。
- GPU counters 在 CUDA 不可用时必须是 `null`，不能失败。
- 输出稳定 JSON，方便后续 CI 或人工比较。
- benchmark 不参与 algorithm 正确性，只测 runtime path。

验收标准：

- `bench_fake_ar.py` 和 `bench_fake_diffusion.py` 都能生成 JSON。
- `metrics_schema.py` 定义并校验每个 metric 字段。
- engine counters 必须从 `GenerationMetrics.engine_counters` 或统一 runtime metrics 聚合，不允许每个 benchmark 私自发明一套字段。

### 产物 5：Family adapter surface tests

新增：

```text
tests/contracts/test_family_adapter_surface.py
```

把 new-family checklist 编码成测试，不写成额外 docs。测试至少检查：

```text
family registry dynamic import path 可解析
runtime builder 返回 RuntimeBundle
bundle.model 满足 RuntimeModel
model.replay_forward(...) 返回 ReplayResult
evaluator 输出 TrajectorySignalBatch
family module 不 import trainer / algorithm 私有路径
```

验收标准：

- fake family 和现有 registry entries 都通过 surface test。
- 失败信息指出具体 family、具体 import path 或 contract 字段。

## 分阶段执行

### Stage 1：建立 fake family contract tests

新增 fake family 测试，不先接真实模型。

重点不是模拟全部模型行为，而是强制系统主路径成立：

```text
rollout -> trajectory -> replay -> signal -> algorithm
```

这一步完成后，repo 才有可证明的 extensibility。

### Stage 2：加入 import boundary 和 family surface tests

把架构边界变成测试，防止后续新增 family 把 trainer / algorithm / engine core 污染掉。

同时把 new-family checklist 编码为 `test_family_adapter_surface.py`，避免额外文档变成维护负担。

### Stage 3：收紧 signal / algorithm 边界

基于 contract tests 暴露的问题，清理 evaluator 和 algorithm 的直接耦合。

判断标准：

```text
ReplayResult 是 current output。
TrajectorySignalBatch 是 training signal。
Algorithm 不理解 family 私有字段。
```

更细的判断标准：

```text
Algorithm-specific 不等于 family-specific。

DiffusionNFT 可以是 diffusion-specific。
DiffusionNFT 不可以是 SD3 / Wan / Cosmos rollout-storage-specific。
```

Stage 3 不要求重写 DiffusionNFT 数学，但要把这类后门列出来并收敛：

```text
AlgorithmInput.metadata["model"]
AlgorithmInput.metadata["rollout_batch"]
AlgorithmInput.metadata["timestep_index"]
algorithm 内部直接调用 TrajectoryResolver.from_batch(...)
algorithm 内部硬编码 replay segment 名，例如 "denoise"
```

### Stage 4：加入 runtime benchmark harness

先用 fake family 做稳定 benchmark，再逐步让 SD3.5 / Janus / Cosmos 接入同一 schema。

### Stage 5：删除不服务主路径的 glue

只有在前面 contract 和 benchmark 立住之后再删除。

删除判断：

```text
Does this file help unified rollout / replay / trajectory / signal / runtime?
```

如果答案是否定的，它应该被移动到 family adapter、recipe helper，或删除。

## 验收标准

本 sprint 完成时必须满足：

- 有 fake AR family contract test。
- 有 fake diffusion family contract test。
- 有 import boundary test。
- 有 family adapter surface test。
- 有 CPU 可跑的 fake runtime benchmark。
- `rg "vrl.models.ar.<concrete>|vrl.models.diffusion.<concrete>" vrl/trainers vrl/algorithms vrl/engine vrl/distributed/ray/rollout` 没有非法引用。
- 所有 evaluator 输出仍然是 `TrajectorySignalBatch`。
- 新增 fake family 不需要改 trainer / algorithm / engine trajectory / Ray rollout 主路径。

建议验证命令：

```bash
pytest tests/contracts tests/architecture tests/rollouts tests/models
ruff check vrl tests benchmarks
python benchmarks/rollout_runtime/bench_fake_ar.py --output /tmp/fake_ar.json
python benchmarks/rollout_runtime/bench_fake_diffusion.py --output /tmp/fake_diffusion.json
```

## 风险与处理

### 风险 1：fake tests 太假，不能证明真实 family 可接入

处理：

- fake tests 只负责证明主路径 contract。
- 真实 family 仍保留 SD3.5 / Janus / Cosmos 的 integration tests。
- fake tests 不替代真实模型测试，只补上 extensibility contract。

### 风险 2：signal 类型继续膨胀

处理：

- 不急着新增 `PolicyOutput`。
- 先固定现有边界：

```text
ReplayResult = current replay output
TrajectorySignalBatch = algorithm-ready signal
```

- 只有当 contract tests 证明 `ReplayResult` 名字不足以表达 policy replay output 时，再做命名迁移。

### 风险 3：把 algorithm-specific 误判成 family-specific

处理：

- 不要求所有 algorithm 都变成通用数学模块。
- 允许 `TokenGRPO` 理解 token-level PPO，允许 `DiffusionNFT` 理解 diffusion NFT。
- 禁止 algorithm 读取 family 私有 storage layout、具体 segment 名、具体 model adapter
  plumbing。
- 如果某个 objective 需要特殊训练输入，先定义 typed training input / signal，再让
  algorithm 消费它。

### 风险 4：benchmark 变成另一个维护负担

处理：

- benchmark 先只覆盖 fake family。
- schema 小而稳定。
- benchmark 不进慢速真实模型路径。
- 真实模型 benchmark 后续按需要接入。

### 风险 5：family registry 全局状态污染测试

处理：

- 给 tests 增加临时 registry helper。
- contract tests 必须 restore `FAMILY_REGISTRY` 和 alias cache。
- 不让 fake family 永久进入生产 registry。

## 后续可选方向

完成本 sprint 后再考虑：

- 将 `vrl/scripts/common` 改名或迁移到 recipe 层。
- 把 trajectory storage / reward artifact lifecycle sprint 接到 benchmark schema。
- 给 Ray resident rollout 加真实 worker startup / weight sync benchmark。
- 给 AR KV cache 和 diffusion denoise loop 各自加性能 profile gate。
