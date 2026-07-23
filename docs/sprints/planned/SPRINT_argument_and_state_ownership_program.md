# SPRINT：Argument and state ownership program

状态：**planned（2026-07-22）**。

## 0. 结论先行

仓库的主要层次是必要的，不应把 OmegaConf、public schema、resolver/builder、runtime
dataclass、wire payload 和 async ownership state 压成一层。当前问题不是“类太多”，而是：

1. 同一个决定在两个层里各存一份；
2. public 参数进入了调用链，却没有真正影响行为；
3. runtime DTO 保存了可从既有真值派生的字段；
4. family-owned 配置仍集中在全局 schema；
5. 测试为了观察内部行为，反向把 test-only 字段塞进 production DTO。

本 program 的原则是：**保留边界，删除镜像；保留协议，拒绝无效输入；保留 owner state，
把纯派生值改成 property 或构造时局部变量。**

本次审计已覆盖目标定义、构造点、生产 reader、测试 reader、YAML writer、OmegaConf path、
`getattr`/`.get()` adapter、registry dotted string、`__all__` 和历史 Sprint 判定。发现的正确性
问题先处理，结构清理后处理；不能用一次大重构同时改变行为与状态形状。

## 1. 哪些层应该存在

| 层 | 唯一职责 | 判定 | 不应拥有 |
|---|---|---|---|
| OmegaConf / YAML | defaults 组合、override、插值、实验值 | **KEEP** | runtime 默认的第二副本 |
| Pydantic public schema | 用户可写 key、类型、枚举、跨字段约束、unknown key | **KEEP** | family runtime 对象、已解析 torch/Ray 值 |
| gate / resolver / builder | 路径、backend、precision、topology 等一次性决策 | **KEEP** | parse 后丢弃的影子结果、匿名 tuple/dict |
| runtime config dataclass | 某个 runtime/controller 真正读取的已解析值 | **KEEP** | generation 镜像、controller-only 镜像、无 reader 字段 |
| request/result/batch DTO | 跨函数、进程、actor 或算法的协议 | **KEEP** | test-only 观测、可由同一 payload 推导的冗余字段 |
| mutable owner state | single-flight task、设备驻留、版本、恢复 ticket、FSM | **KEEP** | 仅为了日志重复保存的 summary |

dataclass 的放置规则：

- public schema 类型靠近它所描述的 public config；family-specific schema 靠近 family，而不是
  永久堆在 `vrl/config/schema.py`。
- runtime config 靠近唯一行为 consumer；如果 controller 与 trainer 共用同一 batch 决策，
  构造一个共享的 immutable plan，而不是复制字段。
- wire DTO 靠近协议边界，不因为字段少就内联。
- math result 靠近算法函数，只保留 caller 真正消费的输出。
- async state 靠近 owner；目标状态、已安装状态、驻留状态和 in-flight task 不能互相派生时都保留。
- 同一对象内能从 retained fields 无歧义得到的值，用 property；构造后不再需要的 provenance 留在
  resolver 局部变量。

## 2. 审计判定总表

### 2.1 必须先修的行为问题

| Suspect | 证据摘要 | 判定 | 落点 |
|---|---|---|---|
| DPO `gradient_checkpointing` | `bool("off") is True`；共享 resolver 已有正确语义 | **FIX** | contract truthfulness |
| DPO `max_grad_norm` 默认 | 构造 `OfflineDPOTrainerConfig` 却从 online `TrainerConfig.max_norm` 取默认 | **DERIVE** | contract truthfulness |
| scalar prompt group remap | list 分支同步 batch/trajectory；scalar 分支只改 batch；evaluator 优先读 trajectory | **FIX** | contract truthfulness |
| `AlgorithmInput.metadata` | NFT 只读三个固定字符串键；缺 timestep 默认为 0，负数会索引最后一列 | **FIX** | contract truthfulness |
| Janus/NextStep `image_size` | public request 传入，decoder 忽略；Janus 还用可被 `python -O` 删除的 assert | **FIX** | contract truthfulness |
| Flux/HunyuanVideo/Echo `negative_prompt` | public 参数可非空，family 直接 `del` | **FIX** | contract truthfulness |
| `DiffusionSamplingParams.sde` | 唯一 parser 总会构造；Optional 和 `None` branch 没 producer | **FIX** | contract truthfulness |
| SDE window policy | request knob 存在 loop-owned `DenoiseSDEParams`，loop 只消费 resolved window | **MOVE** | contract truthfulness |
| public data no-op keys | `data.source`、preprocessing 的 `metadata_schema/target_text/media_type` 无行为 reader | **REMOVE** | contract truthfulness |
| `production` OPEN dict | 只有 `enabled` 活；8 个 `report_path` 无 reader，拼错 key 也通过 | **FIX/REMOVE** | contract truthfulness |

### 2.2 Config 与 argument owner

| Suspect | 证据摘要 | 判定 | 落点 |
|---|---|---|---|
| validation/build 重复 parse | Root schema result 被丢弃；precision/reward 随后再次解析 | **DERIVE** | config ownership |
| `build_configs()` dict、reward tuple | production caller 用字符串下标和 `[0]/[1]` | **FIX** | config ownership |
| data loader 选择 | schema 和 runtime 各实现一次 format → loader 映射 | **DERIVE** | config ownership |
| `TrainerConfig.resume_from` | checkpoint loader 读 raw cfg；dataclass 字段无 consumer | **REMOVE** | config ownership |
| `TrainerConfig.gradient_checkpointing` | model setup 读 cfg/helper；OnlineTrainer 不读该字段 | **REMOVE** | config ownership |
| `TrainerConfig.samples_per_chunk` | generation 直接读 rollout/request；trainer 副本只校验/日志 | **REMOVE** | config ownership |
| `microbatch_size`、host memory budget | 控制 optimizer replay/backward，却位于 rollout YAML owner | **MOVE** | config ownership |
| microbatch size + GAS | 同一 batch split 决策保存两份 | **DERIVE** | config ownership |
| request sampling blacklist | flatten 后用 `_REQUEST_SAMPLING_EXCLUDES` 负向排除，新增 key 默认泄漏 | **FIX** | config ownership |
| Ray worker defaults | schema、runtime dataclass、`from_cfg()` literal、base preset 多份 | **DERIVE** | config ownership |
| FSDP/DDP defaults | schema、strategy ctor、builder fallback 重复 | **DERIVE** | config ownership |
| supervisor CLI defaults | dataclass 与 argparse 重复，负 retry/backoff 可进入 runtime | **DERIVE/FIX** | config ownership |
| Trainer/Actor known-key 表 | runtime dataclass metadata 已声明 YAML owner，schema 再手写 online 字段 | **DERIVE** | config ownership |
| family model schema | family Pydantic classes与映射集中在全局 schema，family runtime 又读 raw dict | **MOVE** | family config ownership |

### 2.3 Stored payload 与 result 字段

| Suspect | 证据摘要 | 判定 | 落点 |
|---|---|---|---|
| `runs_in_isolated_subprocess` | 只有 registry writer 和 test reader | **REMOVE** | runtime payload |
| `supports_policy_replay` | capability bool 与 replay recipe 双状态；Magi 被迫提供不可达 raise-builder | **DERIVE/FIX** | runtime payload |
| `DiffusionBackboneOutput.metrics` | 只有测试读 transformer call count；测试已有 recording transformer | **REMOVE** | runtime payload |
| `RenoiseStepResult.next_sample_mean/std_dev` | 前者 test-only，后者零 reader | **REMOVE** | runtime payload |
| CausVid resolved revisions/source | pin/下载已在 resolver 完成；resolved struct 只有两个 Path 被消费 | **REMOVE** | runtime payload |
| generation request/output fields | `return_artifacts`、`priority`、sample `prompt_id`、output `prompts` 零行为 reader | **REMOVE** | runtime payload |
| executor `plan(..., sample_rows)` | 三个实现均不读；不是 shared protocol 参数 | **REMOVE** | runtime payload |
| distributed resource summaries | rollout GPU count、colocation、trainer reservation均由 retained topology 决定 | **DERIVE** | runtime payload |
| distributed context booleans | `distributed` 由 strategy、`is_primary` 由 rank 决定 | **DERIVE** | runtime payload |
| `BundleLayout.trainer_bundle_indices` | test-only；bundle GPU/CPU layout 已表达真实 reservation | **REMOVE** | runtime payload |
| `_bundle_cpu(..., gpu_id)` | body 只读 bundle index | **REMOVE ARG** | runtime payload |
| `TrainState.total_reward/total_loss` | 只累加并 checkpoint roundtrip，无输出或控制流 reader | **REMOVE** | runtime payload |
| reward scoring count/metadata | metadata 已在 context；expected count/batch size可从 rows/outputs得到 | **REMOVE/DERIVE** | runtime payload |
| `RunMetrics.run_dir` | 构造后零 reader | **REMOVE** | runtime payload |
| MultiSegment inner input writes | 确定 callee TokenGRPO 只读 signals/advantages | **REMOVE ASSIGNMENT** | runtime payload |
| online metrics CSV | header、row dict、format order 三份同一 schema | **DERIVE** | metrics IO contract |
| Kling prompt table | 大型 domain vocabulary/template 混在 model workflow | **MOVE** | runtime payload |

### 2.4 Trajectory 与 token runtime

| Suspect | 证据摘要 | 判定 | 落点 |
|---|---|---|---|
| `RolloutBatch.dones/videos/prompts` | reward 后继续搬运，trainer/algorithm 不消费 | **REMOVE** | trajectory single source |
| numeric `TrajectoryBatch.group_ids` | 与 trainer-remapped `RolloutBatch.group_ids` 双写；稳定身份已在 sample rows | **REMOVE/MOVE** | trajectory single source |
| `RolloutBatch.observations/actions` | 是 trajectory role tensor 的 flat mirror，但仍有多个 consumer | **DERIVE THEN REMOVE** | trajectory single source |
| `TrainingView/LossUnit` | loss units 只有 validator reader；batch validator 已检查相同 invariants | **REMOVE** | trajectory single source |
| `primary_segment` | 活语义藏在 view/context；Janus R1 不能总派生成第一个 segment | **MOVE** | trajectory single source |
| trajectory metrics count/axes | 永远等于 sample rows/axes，ops 还需同步维护 | **DERIVE** | trajectory single source |
| iteration schedule metadata | typed fields与 metadata 重复；sample count可从 batches得到 | **DERIVE** | trajectory single source |
| AR trajectory context keys | 一部分行为必需，一部分可派生，一部分只能算 provenance | **DERIVE/ANNOTATE** | trajectory single source |
| token per-sequence family/task/dtype/max tokens | loop 每次只服务一个 request，生产无异质 producer | **REMOVE** | token state thinning |
| token request/sample IDs 与 metadata | test-only；metadata 只承载 `row_index` | **REMOVE/FIX** | token state thinning |
| token `finished/remaining_tokens` | 前者可由 position 派生，后者 test-only | **DERIVE/REMOVE** | token state thinning |
| sequence key / batch key 与 IDs | 为尚未实现的 cross-request scheduler 预存 future-only state | **REMOVE** | token state thinning |
| one-field result wrapper、step positions list | caller 立即解包；scalar position 已证明 envelope invariant | **REMOVE** | token state thinning |

## 3. 实施 Sprint 与依赖

1. [Contract truthfulness and no-op inputs](../done/SPRINT_contract_truthfulness_and_noop_inputs.md)
   先修 silent no-op、错误 derivation 和 stale group IDs。
2. [Config argument ownership and resolution](SPRINT_config_argument_ownership_and_resolution.md)
   建立一次解析和明确 owner；依赖 Sprint 1 的 public key 决策。
3. [Online metrics IO contract](SPRINT_online_metrics_io_contract.md)
   先建立稳定 row protocol，供 supervisor与 continuous telemetry共同使用。
4. [Runtime payload smallest truth](SPRINT_runtime_payload_smallest_truth.md)
   机械删除 dead/derived payload；可与 Sprint 2 的后半段并行，但不能抢先改变 config shape。
5. [Trajectory and rollout single source](SPRINT_trajectory_rollout_single_source.md)
   先迁 consumer，再删 flat mirrors；依赖 scalar remap 修复。
6. [Token loop state thinning](SPRINT_token_loop_state_thinning.md)
   删除当前单请求 runtime 中为未来 cross-request 功能预存的状态。
7. [Family model config ownership](SPRINT_family_model_config_ownership.md)
   承接 `SPRINT_config_as_signatures.md` deferred P3/P4；在 shared config resolution 稳定后迁移。

现有 [Continuous stage contracts and baseline](SPRINT_continuous_stage_contracts_and_baseline.md)
仍是独立 program 的首个 Sprint。它新增 metrics 前，应先复用 Sprint 3 的 typed metrics row，避免
继续同步手写 header/dict/format list；不需要把 continuous ownership state并入本 program。

## 4. 明确保留

- `PolicySemantics`、family registry entry、dotted import path 和 build/runtime facade；
- OmegaConf → Pydantic → resolver → runtime config 的职责分层；
- `ModelBuild` raw family config wire boundary，但 family 侧必须尽早 parse；
- `RuntimeBundle`、scheduler/raw handle、reward request/artifact/result 的独立验证边界；
- `ReplayModel` 统一签名、family-uniform stage methods、lazy import helper；
- Ray lifecycle/placement、continuous producer/queue/owner、park/offload ticket、single-flight task；
- `ContinuousRolloutConfig` → frozen `ContinuousRolloutSettings`；
- `RolloutBatchBuildContext.device="cpu"`；
- reward call-scoped report 和 transport records；
- trajectory/sample identity、request/family/task provenance；
- checkpoint/revision/SHA/file names、environment keys、architecture dimensions、protocol names、
  transition tables和测试 fixture constants。

## 5. ALL_CAPS 与薄函数专项

### 改变

- Kling 的大型 prompt vocabulary/template table移到独立、具名的 prompt protocol module。
- `_REQUEST_SAMPLING_EXCLUDES` 在 typed 正向投影原子落地后删除；不能先删 guard。
- 手写 online schema key 集合从 typed owner 派生。

### 保持

- precision token/rule、offline DPO allow-list、health metric subset、trajectory role invariant；
- checkpoint/model revision/file/SHA、Ray env 名、统计表 `_T_ONE_SIDED_95`；
- registry table与 dotted strings；
- protocol facade、lazy import、framework adapter、test fake、跨 family 一致的薄 runtime/build 文件；
- production contract/path/backend validator：三者是不同失败边界。

### Non-goals

- 不为减少 LOC 合并合法协议层。
- 不把 family-specific trajectory context 全改成 dataclass；plain dict 是可扩展 wire payload。
- 不删除 async ownership、public capability 或 transition state。
- 不实现 parked cross-request scheduler。
- 不做全仓 YAML default sweep；只删除确认等于单一默认且没有“显式实验 pin”价值的项。
- 不重开历史 done Sprint；新证据在本 program 中明确 supersede 旧判定。

## 6. 历史判定更正

- `SPRINT_diffusion_request_branch_dead_fields_cleanup.md` 曾因测试读取而保留
  `DiffusionBackboneOutput.metrics`；更新后的规则明确 test-only reader 仍是 dead。
- `SPRINT_resolved_struct_field_audit.md` 曾因 `image_size` 被传递而判为 live；本次读取函数体后确认
  Janus/NextStep 并未消费，属于 live caller / dead semantics。
- `SPRINT_trajectory_views_types_dead_fields_cleanup.md` 只删除部分 view 字段；剩余 `loss_units`
  仍只有 validation reader。
- `SPRINT_rollout_wire_diet.md` deferred 的 observations/actions 迁移由 Sprint 5 承接。
- `SPRINT_config_as_signatures.md` deferred P3/P4 由 Sprint 7 承接。

## 7. Program verification

每个 child Sprint 都必须：

1. 为 true/false 两条路径加回归；
2. 对删除项重跑 symbol、string key、`getattr`、registry dotted path 搜索；
3. 对 derived property 覆盖 true/false topology；
4. load 全部 bundled experiments并 import dotted entrypoint；
5. 只跑 CPU/mocked-Ray 测试，不启动 Ray cluster，不运行 GPU；
6. `ruff` touched files、`git diff --check`；
7. 不 commit、不 push。

本次只读基线：

```text
config/scripts/ray-config/registry: 421 passed, 1 skipped
model/algorithm/reward/math:         95 passed, 1 skipped
runtime/trajectory/token:           139 passed
Ray resource/placement pure tests:   94 passed, 2 deselected
bundled experiments:                 64 loaded, 0 entrypoint import errors
```

这些测试集有重叠，不能相加当作“总测试数”；它们证明审计入口可在 CPU 环境稳定复现，不证明待修
行为已经正确。

## 8. Definition of Done

- [ ] 每个 public key 只有一个语义 owner，unsupported value fail fast。
- [ ] 每个 resolved/runtime field 有非日志 production consumer，或显式 provenance 注释。
- [ ] 每个派生字段改为 property/局部变量，构造 API 无法传入矛盾状态。
- [ ] trainer、generation、collector、trajectory不再保存彼此的镜像。
- [ ] family-specific schema 与 runtime owner相邻，registry 仍保持 lazy。
- [ ] protocol、async ownership、transition table和跨 family 一致形状未被破坏。
- [ ] 所有 child Sprint 的 CPU gates 通过。

## 9. References

- `AGENTS.md`
- `vrl/config/schema.py`
- `vrl/config/validation.py`
- `vrl/config/builders.py`
- `vrl/trainers/core/types.py`
- `vrl/generation/types.py`
- `vrl/families/registry.py`
- `vrl/rollouts/batch/core.py`
- `vrl/trajectory/types.py`
- `vrl/trajectory/views.py`
- `vrl/ray/resources.py`
- `vrl/scripts/common/online.py`
- `docs/sprints/done/SPRINT_config_as_signatures.md`
- `docs/sprints/done/SPRINT_resolved_struct_field_audit.md`
- `docs/sprints/done/SPRINT_rollout_wire_diet.md`
- `docs/sprints/done/SPRINT_trajectory_views_types_dead_fields_cleanup.md`
