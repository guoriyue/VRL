# SPRINT：Runtime payload smallest truth

状态：**planned（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：

- [Contract truthfulness and no-op inputs](../done/SPRINT_contract_truthfulness_and_noop_inputs.md)
- Config typed build result可并行落地；本 Sprint不改变 public config shape。

## 0. 结论先行

本 Sprint 清理的不是 protocol层，而是 protocol对象里的 dead/derived字段。判断标准：

- production控制流、runtime/config/Ray调用、可抛验证都算 behavior consumer；
- 只有 test reader或格式化/log reader仍算 dead；
- 可由同一对象 retained fields无歧义得到的值，不允许继续作为 constructor input；
- resolver完成 pin/download/validation后不再需要的值留在局部变量；
- 只删除冗余 assignment，不因一个 caller多传几项就删除整个公共 field。

每一 phase都应先改测试观察方式，再删 production字段，避免用测试需求反向扩大 runtime DTO。

## 1. T0 — Generation request/output diet

### 删除字段

| Field | 证据 | 改法 |
|---|---|---|
| `GenerationSampleRow.prompt_id` | 只有构造，无 `.prompt_id` / getattr / key reader | 删 stored field；局部 prompt id继续生成 group/sample IDs |
| `GenerationRequest.return_artifacts` | 唯一 writer固定 `{"output","trajectory"}`；零 reader | 删字段/参数/测试断言 |
| `GenerationRequest.priority` | 零 reader；Ray真实 priority来自 estimated cost | 删字段/参数 |
| `GenerationOutput.prompts` | gatherer只构造，零 output reader；sample rows已有 prompt | 删字段和 construction writes |

`GenerationRequest.prompts` property **KEEP**：family-agnostic execution真实读取它。

### 删除 argument

full-sequence denoise、chunk-AR denoise、token-AR 三个 local preview
`plan(request, sample_rows)` 都不读 `sample_rows`，且 `GenerationChunkExecutor` protocol没有该 method。
统一为 `plan(request)`；保留三个薄 facade以维持 cross-family quality-preview形状。

### 测试

- request normalization string → `GenerationInput`；
- sample/group/trajectory IDs稳定；
- output gather成功/错误两条路径；
- Ray job priority仍由 assignment cost决定；
- 三个 preview plan shape与修改前相同。

## 2. T1 — Model/math result 只保留 caller输出

### `DiffusionBackboneOutput.metrics`

唯一值：

```python
metrics={"transformer_calls": raw_calls}
```

只有测试读取；同一测试已经拥有 recording transformer。删除 field和 `raw_calls` production local，
测试直接断言 batched CFG调用一次、separate CFG调用两次。保留具名 output及三个 noise prediction。

### `RenoiseStepResult`

删除：

- `next_sample_mean`：test-only；
- `std_dev`：零 reader。

保留 `next_sample` / `log_prob` 和 result dataclass数学边界。测试自己计算 expected mean，而不是要求
production返回中间量。

### `CausVidResolvedArtifacts`

只保留：

```text
base_model_dir
checkpoint_file
```

删除 `source_root/source_revision/base_model_revision/checkpoint_revision` stored fields，并让三个
resolver只返回 `Path`。revision仍在 resolver内部验证/传给 Hub；source pin和SHA检查不能删。

### 回归

- CFG batched/separate true/false；
- renoise generated action与recorded action/replay两条路径；
- invalid sigma/shape失败；
- CausVid local/remote path成功，错误 revision、缺 pin、错误 SHA失败。

## 3. T2 — Registry capability由 recipe派生

### `runs_in_isolated_subprocess`

只有 Magi registry writer和两个 test reader，零 production行为。删除 capability和 writer：

- registry值测试删除；
- VAE source scan使用测试本地 Magi allow-list，不把测试分类塞回 production capability。

### `supports_policy_replay`

当前同时保存：

```text
GenerationRuntimeCapabilities.supports_policy_replay
DenoiseFamilyBuild replay recipe
```

Magi因 `DenoiseFamilyBuild.__post_init__` 强制 recipe，被迫注册一个 production registry路径永远不可达
的 raise-only replay builder。

修复：

1. `DenoiseFamilyBuild` 接受严格三选一：
   - generic recipe：`replay_cls + transformer_classname`；
   - custom recipe：`replay_runtime_builder`；
   - generation-only：无 recipe，但必须有非空 `replay_unavailable_reason`；
2. `ModelFamilyEntry.supports_policy_replay` property从 recipe派生；
3. 两个 production reader改用 entry property；
4. 删除 capability stored bool；
5. 删除 `build_magi_1_replay_runtime_bundle`、dotted path、export和直接函数测试；
6. `entry.build_replay()` 在 import/权重加载前用 registry reason失败。

必须 **KEEP** `Magi1SubprocessModel.replay_forward/disable_adapter` raise stubs：当前
`RuntimeModel` 继承 `ReplayModel`，这是统一 protocol seam，不是不可达 builder。

### 测试

- generic/custom recipe派生 true并正确 dispatch；
- Magi派生 false且在 import前报显式原因；
- partial generic、recipe+reason、三者全缺都在 registry构造期失败；
- all-family runtime contract覆盖 Magi；replay contract只参数化支持 replay的 family。

## 4. T3 — Ray/resource/context summaries改 property

### `ResolvedDistributedResources`

以下 constructor fields改同名 property：

- `rollout_num_gpus = len(rollout_devices)`；
- `colocated = bool(set(trainer_devices) & set(rollout_devices))`；
- `requires_trainer_reservation` 从 retained topology、worker/GPU与 cross-node决定。

删除 constructor writes，使 API无法传入矛盾 summary。

### `DistributedTrainingContext`

- `distributed = strategy != "single_process"`；
- `is_primary = rank == 0`。

改 property；保留 rank/local rank/world size/device/strategy，它们是 launcher facts。

### `BundleLayout` 与 placement helper

- 删除 test-only `trainer_bundle_indices`；真实 reservation由 `bundle_gpu_ids`和 trainer-only bundle
  token CPU表达；
- 删除 `_bundle_cpu(bundle_index, gpu_id)` 的 unused `gpu_id`；
- 若未来建立 topology inspection API，再用显式 provenance DTO，不预存字段。

### 测试

- colocated true/false；
- trainer reservation true/false；
- CPU rollout 0 GPU与N GPU；
- single process、rank0、rank1；
- trainer-only bundle仍获 token CPU；
- constructors不能接受 derived bool/count。

## 5. T4 — Trainer/collector/benchmark dead state

### `TrainState.total_reward/total_loss`

两字段只累加并 checkpoint roundtrip，无 metric、branch或report reader。删除：

- state fields；
- accumulation assignments；
- 新 checkpoint keys；
- restore writes。

旧 checkpoint含这两个 key时静默忽略；`step/global_step`继续严格恢复。

### Reward scoring

- `TrajectoryRolloutBatchBuilder.reward_scoring_input(metadata)` 删除重复 argument，读
  `self.context.metadata`；
- `RewardScoringInput.expected_count` 删除，真实 count来自 sample rows/outputs；
- `batch_size` 改 property；
- 保留 sample/output长度 mismatch validation。

### `RunMetrics.run_dir`

只删除 `vrl/scripts/perf/reward_overlap_benchmark.py` 的字段和一处 assignment；保留 metrics
dataclass与函数参数 `run_dir`，后者仍用于错误信息和文件读取。

### MultiSegment assignment

`MultiSegmentTokenGRPO` 的确定 callee是 `TokenGRPO.compute_loss`，inner `AlgorithmInput` 只传
`signals/advantages`。不要删除 public `AlgorithmInput.rewards/group_ids`：adapter在需要计算
advantages时真实使用。

Online metrics CSV 的独立 protocol cleanup由
[Online metrics IO contract](SPRINT_online_metrics_io_contract.md) 承接；本 Sprint不修改同一
header/row/format区域。

## 6. T5 — ALL_CAPS prompt table移到正确 owner

`kling_video_reward.py` 同时承载 load/inference/parser和大型 domain prompt table。移动到
`vrl/rewards/models/kling_prompt_templates.py`：

```text
_DIMENSION_DESCRIPTIONS
_SIMPLE_PROMPT
_VIDEOSCORE_QUERY_PROMPT
_DETAILED_PROMPT_WITH_SPECIAL_TOKEN
_DETAILED_PROMPT
_build_video_reward_prompt
```

这个薄文件是合理的独立 prompt protocol/taxonomy asset。

保持原位：

- `_DEFAULT_REWARD_MODEL`：checkpoint identifier；
- `_SPECIAL_TOKENS`：tokenizer/checkpoint protocol；
- `_SCORE_KEY_MAP`：output normalization；
- Kling private checkpoint dataclass：loader-local contract。

VideoScore2/Cosmos3/UnifiedReward/VideoCon的短 prompt/regex与 parser紧耦合，不拆。

测试所有 template type输出保持，并覆盖非法 template/dimension。

## 8. What changes / what stays

### 改变

- dead/test-only stored fields；
- summary fields改 property；
- fake capability与fake builder改 recipe derivation；
- prompt大表移到真实 protocol owner。

### 保持

- request/result/batch/registry/math result等具名边界；
- registry dotted import/lazy dispatch；
- RuntimeBundle、raw handle、scheduler；
- resource placement/lifecycle、rank/device ownership；
- reward request/artifact/result lineage与独立验证；
- model protocol stubs与跨 family一致方法形状。

## 9. Non-goals

- 不清理 trajectory/rollout mirrors；由独立 Sprint处理。
- 不清理 token scheduler future-only state；由独立 Sprint处理。
- 不删除 log/telemetry fields仅因为不控制训练；明确 display-only即可。
- 不把薄 protocol module内联到 caller。
- 不运行 CausVid/Magi模型或任何 GPU/Ray experiment。

## 10. Acceptance gates

- generation/model/math/registry/resource/collector/checkpoint/metrics定向 CPU tests；
- CausVid submodule若未初始化，只跑不依赖 upstream source的 resolver/mocked tests，并记录环境事实；
- load registry全部 dotted paths；
- dead symbol与字符串 key复扫；
- `ruff` touched files；
- `git diff --check`。

## 11. Definition of Done

- [ ] 每个删除字段零 production reader且测试改为观察真实边界。
- [ ] 每个 derived property无可传矛盾 constructor参数。
- [ ] Magi generation-only由 build recipe表达，无 fake replay builder。
- [ ] 旧 TrainState checkpoint兼容读取。
- [ ] Kling大型 prompt taxonomy离开 workflow module。

## 12. References

- `vrl/generation/types.py`
- `vrl/generation/execution/ids.py`
- `vrl/generation/ray/executor.py`
- `vrl/models/steps/denoise/common/backbone.py`
- `vrl/math/denoise/renoise.py`
- `vrl/models/families/causvid/model.py`
- `vrl/families/registry.py`
- `vrl/models/families/magi_1/runtime.py`
- `vrl/models/families/magi_1/model.py`
- `vrl/scripts/common/factory.py`
- `vrl/ray/resources.py`
- `vrl/ray/placement.py`
- `vrl/trainers/distributed.py`
- `vrl/trainers/core/types.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/collector/batch_builder.py`
- `vrl/rollouts/collector/rewards.py`
- `vrl/scripts/perf/reward_overlap_benchmark.py`
- `vrl/rewards/models/kling_video_reward.py`
