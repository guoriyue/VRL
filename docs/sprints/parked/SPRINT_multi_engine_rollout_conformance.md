# SPRINT：Native + external execution provider conformance

日期：2026-07-13

Status: **parked**. Under the program's strict ordering, the trigger is completion
of the SGLang Qwen-Image S4 pilot; this sprint cannot unpark with only the first
FlashDreams binding. Every provider must also pass the shared configured
blocking-call deadline gate, which remains unfinished. The completed
[worker process-health sprint](../done/SPRINT_rollout_worker_liveness.md) covers
actor-process reachability only: its health concurrency group can respond while
the default group is busy or hung, so it cannot replace a business-call deadline.
C0 native oracle fixtures already live in the active native-engine program and
do not wait for this sprint to create a separate framework. This sprint must not
prebuild a fake/native-only provider framework or force step-level and full-chunk
providers behind a false shared internal API.

## 0. 结论先行

wm-infra native engine 是语义来源和默认路径。FlashDreams 与 SGLang-Diffusion 只有在
相同的 trajectory、replay、policy-version、weight transaction、lifecycle 与 failure
contract 下通过，才能作为可选 execution provider。

“能生成图片/视频”只证明 inference wiring；“能返回 old log-prob”也只证明 collection
的一半。RL provider 的最小正确闭环是：

```text
native request + committed policy version
  -> provider collect
  -> native typed trajectory
  -> native replay on the same transition/conditioning
  -> old/fresh log-prob parity
  -> reward/artifact
  -> transactional next-version install
```

本 sprint 的任务是建立一个长期 conformance harness，而不是维护一张手写“支持模型”
宣传表。

## 1. Scope

### Provider matrix

| Path | 本 sprint 角色 | 首个 representative |
|---|---|---|
| Native diffusion | oracle | Qwen-Image 或当前最小可复现 diffusion family |
| FlashDreams | causal streaming provider | Self-Forcing Wan2.1-T2V 1.3B |
| SGLang-Diffusion | server/process-backed provider | Qwen-Image T2I |
| Native categorical AR | unified-engine regression guard | Janus-Pro |
| Native continuous AR | unified-engine regression guard | NextStep-1 |
| Native multi-segment AR | unified-engine regression guard | Janus-Pro-R1 |

AR 本轮没有 external provider。把 AR 放进 matrix 是为了证明 external diffusion seam
没有把一个 native generation engine 拆成两个不一致的 runtime。

### 不在 scope

- 不把所有 23 个 registry entry 都跑成真实模型矩阵；
- 不用一个 shared seed 强求不同 engine bitwise identical output；
- 不比较 trainer algorithm 优劣或 reward curve；
- 不做 native transformer layer parity；该工作仍属于 parked native-executor sprint；
- 不把 conformance harness 变成 benchmark leaderboard。

## 2. One source of truth

### Test discovery

两个正交 source of truth 分别声明：

- `ModelFamilyEntry`：`policy_semantics`、`family_build`、显式 gatherer/runtime capability
  与默认 native `executor_cls`；`collector_kind` 仅为兼容投影；
- typed provider binding：只构造一次的 immutable source/schema provenance、provider
  process/build path、实际 capability、family-specific executor binding 与 representative
  conformance fixture。

tests 从 family/provider binding join 发现 case，不把 provider provenance 复制进每个
family entry，也不维护平行
`SUPPORTED_PROVIDERS / SUPPORTED_MODELS / EXPECTED_CAPABILITIES` 常量。capability 只有在
生产代码存在非日志 consumer、并有正负测试时才可声明。

### Readiness 不混用

README 的 recipe readiness 有四级，provider 结果不得跳级：

- **Planned**：未接通；
- **Integrated**：model/runtime wiring 与 rollout parity 已有，但完整 experiment recipe 或
  environment contract 仍缺；
- **Runnable**：config、entrypoint、runtime path、structural tests 与完整 recipe/environment
  contract 已有，训练质量尚未证明；
- **Validated**：真实训练证明 optimizer update、非平 reward、artifact 与 changed
  weights。

provider wiring/parity 通过但 recipe/environment 未闭合时只能到 Integrated；完整 contract
通过最多到 Runnable，不能自动标成 Validated。性能更快也不能替代真实训练曲线。

## 3. Harness design

### C0 — Native oracle fixtures

为每类 transition 建最小、可序列化 fixture：

```text
diffusion Gaussian transition
categorical AR token transition
continuous AR Gaussian transition
multi-segment AR trajectory
```

fixture 包含 request、seed derivation、policy version、observations/actions/timesteps、
old log-prob、replay inputs 与预期 schema；不包含 live model、Ray actor、provider
session/cache 或 HTTP response。

fixture 分两类：

- **long-term contract fixture**：小、确定、留在
  `tests/generation/conformance/fixtures/`；
- **one-shot real-model capture**：只用于回答一次 parity 问题，结果摘要写入 sprint/
  conformance report 后删除原始大 tensor/media，除非它成为稳定回归资产。

### C1 — Collection mapping

每个 provider adapter 必须证明：

- request family/task/shape/seed 没有丢失；
- output sample order、prompt index 与 sample index 正确；
- observations/actions 的 transition 方向一致；
- timestep dtype/value/domain 一致；
- old log-prob reduction 维度和 normalization 一致；
- replay conditioning 的每个字段都有 producer 与 trainer consumer；
- artifact 与 trajectory 属于同一个 sample；
- policy version 来自 native committed state，不来自 provider 推测。

对 SGLang，显式检查 `[T+1]` latents → T transitions。对 FlashDreams，显式检查
`temporal_chunk` 与 `renoise_transition` 是两个不同且被 grouped replay 消费的逻辑轴，
并检查每个 chunk 的 terminal clean producer；不得 flatten 后丢失 cache boundary。

### C2 — Replay parity

三层 gate 依次运行：

1. **CPU analytic**：已知 Gaussian/categorical distribution，sample-time 与
   replay-time log-prob 相等；
2. **fake model**：provider mapping 后 native evaluator round-trip；
3. **real model**：同 checkpoint/conditioning/precision 下 old-vs-fresh ratio≈1，
   使用现有 drift guard 的明确 tolerance。

provider 自报 log-prob 与 native replay log-prob 都记录；trainer 仍以 native trajectory
和 evaluator 为 contract。若两者 reduction convention 不同，先在 adapter 边界做有测试
的明确转换，不能在 trainer 中加 provider 分支。

### C3 — Weight/version transaction

对每个 provider 注入：

- complete update；
- one-worker rejection；
- provider-specific state checksum/digest mismatch（同一 provider 只定义一种 named-state
  identity）；
- key-schema mismatch；
- timeout；
- update 后 generate；
- in-flight generate 与 update 的竞争。

验收不变量：

```text
all required ACKs agree
  -> commit active policy version
otherwise
  -> keep previous committed version
  -> close admission
  -> terminate the runtime with unknown worker state
```

不支持 versioned slots 的 provider 只可声明 strict/draining。只有 request 能绑定具体
version、且旧 version 仍可执行时，才能声明 non-draining continuous。

### C4 — Lifecycle and failure

fake/process/local-Ray tests 覆盖：

- activate single-flight；
- request deadline；
- malformed provider output；
- provider process/actor death；
- partial chunk completion；
- offload during/after drain；
- shutdown during activation/generation/update；
- repeated shutdown；
- cleanup failure retry；
- no orphan actor/process/port/GPU lease。

一个 chunk 失败时不能拼接其他 chunk 的未知 partial result。丢弃整个 request，关闭当前
runtime，并让当前 attempt fail closed；process supervisor 从 latest complete checkpoint
启动新 attempt，不在同一进程重试该 request。

### C5 — Performance report

correctness 全绿后才测：

- steady-state samples/s；
- end-to-end collect wall time；
- provider serialization/transport time；
- peak GPU/host memory；
- weight update time；
- activation/offload/restart time；
- trajectory bytes/sample；
- quality/reward distribution sanity。

native 是对照，不是必须被击败的“fallback”。外部 provider 可以因 model coverage 进入，
但默认 recipe promotion 需要写明它的优势是 coverage、parallelism 还是 throughput。

## 4. Long-term assets

预期 canonical 结构在实现时按真实 imports 收敛，不预建空目录：

```text
tests/generation/conformance/
  test_runtime_contract.py
  test_trajectory_mapping.py
  test_weight_version_transaction.py
  test_provider_failure.py

vrl/scripts/eval/
  <provider>_rollout_replay_parity.py

docs/sprints/info/
  SPRINT_execution_provider_conformance_results.md
```

- CPU/fake/local-Ray tests 是长期资产。
- 每个 provider 的真实 parity entrypoint 若可复用，是长期资产，放
  `vrl/scripts/eval/`。
- 临时 checkpoint copies、base64 payload dumps、large trajectory tensors、profiling
  traces 是 one-shot；结论记录后按“same source + same lifecycle”一起清理。
- 不删除其他 sprint/contributor 的历史 outputs。

## 5. Architecture hygiene

### 应改变

- 复用 `GenerationChunkExecutor` + native trajectory 作为共同 seam，并让同一 harness
  参数化运行；只有真实重复的 control actions 才另提协议。
- provider response 在一个 adapter 边界转换成 native types。
- readiness/capability 从 registry 与测试结果派生。

### 保持不变

- `GenerationRuntime`、`GenerationChunkExecutor`、Ray worker adapter、family
  runtime builder 保持独立；它们分别是 public、cross-family、framework 与 lazy-import
  boundary。
- diffusion 与 AR 的 family-specific mapping 可以是薄文件；一致目录形状提升 grep/debug
  价值，不为减少 LOC flatten。
- native trajectory/evaluator 保持唯一 trainer-facing schema。

### Data/constant rules

- 不新增手写 provider/model/capability ALL_CAPS table。
- schema keys、wire version、fixture expected values 可以是常量，因为它们是真实
  protocol/test boundary。
- conformance report 的 `ProviderResult` 字段必须有行为 consumer（promotion gate/
  validation）或显式标注 display/provenance-only；只在 log/test 中读取的字段应删除。
- 不把 provider-specific prompt templates、backend taxonomy 或 model vocabulary 放进
  harness workflow code。

## 6. Definition of Done

- [ ] native diffusion、三类 native AR representative 通过统一 runtime regression。
- [ ] FlashDreams Self-Forcing 通过 collect → native trajectory → replay parity。
- [ ] SGLang Qwen-Image 通过 `T+1` trajectory mapping 与 replay parity。
- [ ] partial/mismatched/timeout weight update 都不推进 policy version。
- [ ] strict 与 non-draining capability 无法被错误声明。
- [ ] provider failure 丢弃 whole request 并终止当前 attempt，无 partial chunk 混合。
- [ ] terminal cleanup 后无 orphan process/actor/resource。
- [ ] performance report 分离 model compute、transport 与 lifecycle overhead。
- [ ] wiring/parity 不会越过 Integrated；Runnable 不会自动标成 Validated。
- [ ] 不选择 external provider 时 native imports/config/tests 行为不变。

## 7. Negative result

- 某 provider 只过 inference、不过 replay：标记 inference-only，不进入 RL recipe。
- 某 provider 只过 strict version transaction：保持 strict-only，不做 continuous。
- 某 provider correctness 通过但 transport 过慢：保留 implementation，默认关闭，并在
  独立 transport sprint 解决。
- 第二个 provider 不需要适配第一个 provider 的内部 step seam。若共同
  `GenerationChunkExecutor`/trajectory 边界仍携带 FlashDreams 私有概念，删除这些
  概念；不在 trainer 添加第二套分支。
- 任一外部失败都不降低 native engine 的 readiness。

## 参考

- `docs/sprints/SPRINT_native_generation_engine_program.md`
- `docs/sprints/done/SPRINT_rollout_worker_liveness.md`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
- `docs/sprints/planned/SPRINT_flashdreams_execution_provider.md`
- `docs/sprints/parked/SPRINT_self_forcing_causal_family.md`
- `docs/sprints/parked/SPRINT_sglang_diffusion_execution_provider.md`
- `vrl/generation/protocols.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/execution/worker.py`
- `vrl/trajectory/types.py`
- `vrl/trajectory/validation.py`
- `vrl/rollouts/evaluators/`
- `README.md`
