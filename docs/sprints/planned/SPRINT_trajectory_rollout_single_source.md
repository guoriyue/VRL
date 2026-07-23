# SPRINT：Trajectory and rollout single source

状态：**planned（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：

- [Contract truthfulness and no-op inputs](../done/SPRINT_contract_truthfulness_and_noop_inputs.md)
  的 scalar group-remap修复；
- [Runtime payload smallest truth](SPRINT_runtime_payload_smallest_truth.md) 可并行，但不应同时改同一
  batch constructor。

## 0. 结论先行

`RolloutBatch` 是 trainer-ready batch；`TrajectoryBatch` 是 generation/replay record。两者都必要，
但当前同时保存 group IDs、observations/actions、training view和结构 metrics，导致 select、move、
stack、continuous consume都要双写。

目标 source of truth：

```text
GenerationSampleRow.group_id/sample_id/trajectory_id
    = stable generation identity/provenance

RolloutBatch.group_ids
    = trainer-remapped numeric advantage grouping

TrajectoryBatch.segments/axes/sample_rows
    = replay trajectory facts

TrajectoryBatch.primary_segment
    = producer-declared training semantics when it cannot be inferred
```

numeric trainer grouping不再放进 trajectory；replay observations/actions不再平铺在 RolloutBatch。
迁移必须按 reader逐个完成，不能先删字段再用 fallback掩盖。

## 1. T0 — Pin ownership contract

先加 architecture tests：

- generation stable IDs不能由 numeric trainer group重新派生；
- prompt collection只允许改 `RolloutBatch.group_ids`；
- trajectory selection/stack不能偷偷创建 numeric grouping副本；
- replay tensor只来自 `TrajectoryResolver` / typed segment roles；
- primary segment必须存在且 trainable；
- context不是控制面字段的默认逃生口。

本阶段只加 contract，不改 payload shape。它防止后续 phase在迁移中引入第三份状态。

## 2. T1 — 删除 trainer batch 的 post-reward transport

### 删除

- `RolloutBatch.dones`：所有 builder都写全 1，无 trainer/algorithm/model reader；
- `RolloutBatch.videos`；
- `RolloutBatch.prompts`。

后两者在 reward已经完成后才塞进 trainer batch，后续只有 batch ops搬运/切片和 continuous
byte estimate读取。reward scoring仍从 generation output/`RewardScoringInput`读取 decoded media与
prompt；不能误删 reward boundary数据。

同步修改：

- batch builders；
- `select_batch` / device move / stack；
- continuous byte estimate，只估算真正进入 trainer queue的 tensors；
- tests/fakes。

测试：

- reward有/无 decoded media；
- select/split/move；
- strict/continuous collector；
- trainer接收的 rewards/extras/trajectory不变；
- queue byte estimate不再计入已释放 media。

## 3. T2 — Numeric group IDs 只留在 `RolloutBatch`

前置 Sprint已经让 scalar/list remap两分支同步，保证迁移前行为正确。本阶段：

1. 删除 `TrajectoryBatch.group_ids`；
2. builders只构造 `RolloutBatch.group_ids`；
3. `TrajectorySignalBuilder.group_ids` 从 batch读取；
4. select/stack/move/continuous consumer删除 trajectory双写；
5. `GenerationSampleRow.group_id` 保持 string identity，不转成 trainer numeric group。

测试：

- scalar/list remap；
- distinct tensor/device path；
- group-local split；
- GRPO advantage grouping；
- select/stack后 group order；
- trajectory序列化仍含 sample rows稳定 identity。

不能把 numeric group IDs改成 trajectory property：它们在 collection后被 trainer remap，既不是
generation record事实，也不能从 string ID在所有 batch组合场景唯一恢复。

## 4. T3 — 删除 `TrainingView/LossUnit`，保留 primary语义

### 证据

`build_training_view()` 生成的 `loss_units` 在 production只被
`TrajectoryValidator.validate_training_view()`再次校验。`validate_batch()` 已经检查：

- required role；
- axis一致；
- replay input refs；
- trainable segment结构。

唯一真实行为值是 `primary_segment`；Janus R1的显式选择不能一律派生成“第一个 trainable”。

### 修复

- 增加 `TrajectoryBatch.primary_segment: str | None`；
- producer在需要时显式写；
- validator要求显式值存在于 segments且 trainable；
- single-segment可在 builder中派生；
- 删除 `LossUnit`、`TrainingView`、`build_training_view`；
- 删除 `RolloutBatch.training_view`；
- 删除 context `"primary_segment"` 镜像；
- `TrajectorySignalBuilder` 读 typed trajectory field。

不要新增一个只包 primary name的 context/dataclass。

测试：

- single-segment默认；
- multisegment显式 primary；
- unknown/non-trainable primary失败；
- 缺 action/old-log-prob/mask仍由 batch validator失败；
- Janus R1 segment选择不变。

## 5. T4 — Migrate flat observations/actions

`RolloutBatch.observations/actions` 目前仍有真实 reader，不能机械删除。按 consumer迁移：

1. SDE log-prob evaluator；
2. token log-prob evaluator；
3. denoise replay base；
4. online trainer diagnostics/training slices；
5. tests/fakes。

每个 reader通过 `TrajectoryResolver` 和明确 segment role获取 tensor。AR 的旧 `observations` 是
`prompt_input_ids.unsqueeze(1)`，不能用一个跨 family粗暴 property假装所有 trajectory都有统一
observation role；需要从对应 replay input或token segment显式解析。

迁移 gate：

- 每迁一个 consumer，加入缺 role、错 axis、错 segment false test；
- 全仓确认无 `.observations/.actions` production reader后才删除字段；
- 删除 batch ops中对应 select/move/stack；
- 不保留 deprecated property或 silent fallback。

parity tests：

- denoise full sequence；
- denoise timestep slice；
- token AR；
- multisegment token；
- CPU-offloaded replay tensors；
- select/split/streaming microbatch。

## 6. T5 — Derived trajectory/iteration summaries

### `TrajectoryMetrics`

删除 stored：

- `num_samples`，改为 `len(sample_rows)` property；
- `axis_lengths`，改为从 `axes` property派生。

`values` **KEEP**，它是 serializable telemetry/provenance extension；在定义处明确标注
telemetry/provenance-only。select/stack ops不再同步 count/axis map。

### `RolloutIteration`

- `sample_count` 从 batches派生；
- typed `rollout_id/policy_version/mode/prompt_count` 保留；
- metadata只保存 continuous extras；
- 需要写 batch context时临时组合 typed schedule metadata，不保存第二个 mutable truth。

现有 continuous stage identity Sprint未来加入的 `batch_id/attempt` 仍应是 typed schedule fields，
不能塞回 metadata。

测试：

- empty/non-empty trajectory；
- select/stack后的 count/axes；
- strict/continuous iteration；
- policy version与typed schedule metadata一致；
- metadata override不能覆盖 typed identity。

## 7. T6 — AR context payload diet

`TrajectoryBatch.context` / `RolloutBatch.context` plain dict shape **KEEP**：family-specific wire payload
需要可扩展，generic orchestration也会 overlay metadata。不要为每个 family新建 TypedDict/dataclass。

逐 key 建立三类 ledger：

### Behavior — KEEP

- `temperature`：token evaluator重放 log-prob；
- Emu3/GLM `image_height/image_width`：重建结构 mask；
- NextStep `guidance_scale/num_steps/noise_level`：flow replay。

### Derivable — REMOVE

- Emu3/GLM/Janus/LlamaGen/NextStep `image_token_num`；
- NextStep/R1 `image_size`；
- Janus静态 `uncond_source`；
- R1中重复的固定 prompt/token-id项。

### Provenance — KEEP only with annotation

- 不参与 replay但无法从 output恢复的 sampling值，例如 family-specific `top_k/top_p`；
- R1 stage/refine选择；
- 非行为 family的 guidance metadata。

保留的 provenance key必须在 writer处注释用途，并有序列化/展示 consumer；不能只因“也许以后用”
保留。删除前对 literal key、`.get()`、dict merge、test fixture、artifact serialization逐项搜索。

## 8. What changes / what stays

### 改变

- numeric group、primary segment、replay tensors各有唯一 owner；
- trainer batch不再携带 reward后无用 media/prompt/dones；
- structural metrics改派生；
- context删除可恢复副本。

### 保持

- RolloutBatch与TrajectoryBatch两个边界；
- trajectory request/family/task/sample rows provenance；
- axes、segments、reward views、replay inputs；
- context plain dict extension；
- stable generation IDs；
- async CPU ownership与continuous queue item。

## 9. ALL_CAPS / thin functions

保持：

- `FORBIDDEN_TRAJECTORY_METRICS`：serialization denylist；
- `REQUIRED_TRAINABLE_ROLES` 与派生 singleton roles：trajectory schema invariant；
- `TrajectoryResolver`、validator、ops薄模块：共享 protocol和cross-family一致边界。

不为减少 LOC把 resolver调用散回 family代码，也不把 role strings复制成新的 allow-list。

## 10. Non-goals

- 不改变 reward数值、advantage normalization或optimizer batch大小。
- 不把所有 family context硬类型化。
- 不实现新 trajectory storage format。
- 不与 continuous stage identity Sprint合并。
- 不运行 GPU/Ray。

## 11. Acceptance gates

- trajectory builders/validation/ops；
- collector、prompt collection、strict/continuous mocked tests；
- denoise/token evaluators；
- online trainer CPU fakes；
- storage serialization；
- old field/string-key全仓复扫；
- `ruff` touched files、`git diff --check`。

## 12. Definition of Done

- [ ] numeric group IDs只有 RolloutBatch一个 owner。
- [ ] primary segment只有 TrajectoryBatch一个 owner。
- [ ] replay tensor无 RolloutBatch flat mirror。
- [ ] TrainingView/LossUnit删除且 invariants仍被 validator覆盖。
- [ ] structural metrics全部派生。
- [ ] context每个剩余 key是 behavior或显式 provenance。

## 13. References

- `vrl/rollouts/batch/core.py`
- `vrl/rollouts/batch/ops.py`
- `vrl/rollouts/collector/batch_builder.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/orchestration/types.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/rollouts/evaluators/trajectory.py`
- `vrl/rollouts/evaluators/denoise/sde_logprob.py`
- `vrl/rollouts/evaluators/token/token_logprob.py`
- `vrl/trajectory/types.py`
- `vrl/trajectory/views.py`
- `vrl/trajectory/resolver.py`
- `vrl/trajectory/validation.py`
- `vrl/trajectory/ops.py`
- `vrl/trajectory/builders.py`
- `vrl/models/steps/denoise/base.py`
- `vrl/trainers/online/trainer.py`
- `docs/sprints/done/SPRINT_rollout_wire_diet.md`
- `docs/sprints/done/SPRINT_trajectory_views_types_dead_fields_cleanup.md`
- `docs/sprints/reading/SPRINT_batch_context_dict_adjudication.md`
