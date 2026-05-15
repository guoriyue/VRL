# SPRINT：统一 ReplayResult 到 TrajectorySignalBatch

状态：implemented。

实现结果：

- `ReplayModel.replay_forward(...)` 的 public contract 已收口为 `ReplayResult`。
- `RuntimeBundle.model` 现在是 shared runtime 主字段；`RuntimeBundle.policy`
  只保留为短期兼容 property。
- Janus、Janus-R1、NextStep、SD3/Wan/Cosmos diffusion replay 都返回
  `ReplayResult` / `ReplaySegmentResult`。
- evaluator 不再消费 loose replay dict；统一通过 `require_replay_segment(...)`
  和 `require_replay_value(...)` 读取 typed replay payload。
- old logprob、mask、distribution 仍由 `TrajectoryBatch` / `TrainingView`
  resolver 提供，`ReplayResult.values` 不作为训练 source of truth。

验证：

- `pytest tests/models/test_replay_model_contract.py tests/models/test_runtime_model_contract.py tests/rollouts/test_replay_result_signals.py tests/rollouts/test_replay_model_disable_adapter_evaluators.py tests/rollouts/test_multisegment_token_logprob.py tests/models/test_janus_replay.py tests/models/test_janus_r1_model.py tests/models/test_diffusion_model_base.py`
- `pytest tests/trainers/test_online.py tests/trainers/test_weight_sync.py tests/engine/generation/test_runtime_factory.py tests/rollouts tests/models`
- `pytest tests/config/test_load_all_experiments.py tests/config/test_janus_pro_r1_config.py tests/trainers/test_memory_guards.py tests/trainers/test_offline_dpo_timesteps.py`
- `python -m compileall vrl tests`
- `git diff --check`
- legacy replay dict scan 已无命中：
  `rg 'out\["logits"\]|out\["log_probs"\]|fwd\["noise_pred"\]|replay_forward\([^\n]*\).*\["(logits|log_probs|noise_pred)"' vrl tests`

## 核心结论

这轮要解决的问题不是 `Policy` 是否存在，而是：这个名字太大，
而且把 replay、weight sync、general inference 混成了一个概念。

```text
family model 的 replay_forward() 现在返回 loose dict。
Evaluator 还在按 family 拆 key：
  diffusion: fwd["noise_pred"]
  Janus: out["logits"]
  NextStep: out["log_probs"]
```

目标是让 `replay_forward()` 有统一 typed return，但这个 return 只表达
model replay 的原始结果，不表达完整训练信号：

```text
ReplayModel.replay_forward(...) -> ReplayResult
```

然后 evaluator 继续负责把 replay 结果转换成训练信号：

```text
ReplayResult + TrajectoryBatch old_log_prob/mask/ref
-> TrajectorySignalBatch
```

不要引入 `PolicyOutput` / `PolicySegmentOutput`。这两个名字太像
`SegmentSignal` / `TrajectorySignalBatch`，会让人误以为它们也是
algorithm-ready signal。这里真正表达的是 replay forward 的结果，所以统一叫：

```text
ReplayResult
ReplaySegmentResult
ReplayRequest
```

## 命名边界

`vrl/models/families/*/model.py` 里的 `*Model` 是 family 的 general
inference object。它可以同时拥有：

```text
generation helper:
  encode_prompt / prepare_sampling / forward_step / decode_latents
  sample_image_tokens / decode_image_tokens

trainer replay helper:
  replay_forward
  disable_adapter

trainable state sync helper:
  load_trainable_state
```

所以不要把具体 class 改名成 `ReplayModel`。`ReplayModel` 只是一个窄
Protocol，表示“这个 object 可以被 evaluator 用来 replay recorded
trajectory”。同一个 `JanusProModel` / `SD3_5Model` 可以结构化满足它，
但它本身仍然是 general inference model。

建议拆成两个 public interface：

```text
ReplayModel
  evaluator / trainer replay 消费
  replay_forward(...)
  disable_adapter(...)

RuntimeModel
  RuntimeBundle / Ray rollout worker 需要的最低公共 model 能力
  ReplayModel + load_trainable_state(...)
```

不要继续使用 `Policy` 作为 public interface 名字。`Policy` 在 RL 里太大，
容易被误解成完整 generation strategy；这里实际只是 trainer-facing replay
和 trainable-state sync 能力。

`RuntimeModel` 不等于“完整模型所有能力”。generation methods 仍然是
family/executor 私有能力，不进入 public interface。

## 为什么 `ReplayResult` 不能直接替代 `SegmentSignal`

`ReplayResult` 回答的是当前 model replay 的问题：

```text
当前 replay model 对这条 recorded trajectory 的 raw replay output 是什么？
```

`SegmentSignal` 回答的是 algorithm 训练的问题：

```text
new log_prob / old log_prob / mask / ref log_prob / distribution / KL aux 是否完整？
```

所以职责必须分开：

```text
TrajectoryBatch
  rollout record / source of truth
  action, old_log_prob, mask, reward, replay inputs

ReplayResult
  current replay result
  logits / log_probs / noise_pred / family replay extras

SegmentSignal
  algorithm-ready signal for one segment
  current log_prob + old log_prob + mask + optional ref log_prob

TrajectorySignalBatch
  algorithm-ready signal batch for all trainable segments
```

`ReplayResult` 不应该携带 reward、advantage、old logprob 的 source of truth，也不应该成为 algorithm input。algorithm 仍然消费 `TrajectorySignalBatch`。

## 目标数据流

最终链路：

```text
TrajectoryBatch
  -> ReplayModel.replay_forward(...)
  -> ReplayResult
  -> Evaluator
  -> SegmentSignal / TrajectorySignalBatch
  -> Algorithm
```

带 reference model 的链路：

```text
current_model.replay_forward(...)
  -> ReplayResult

ref_model.replay_forward(...)
or current_model.disable_adapter() + current_model.replay_forward(...)
  -> ReplayResult

Evaluator combines:
  current ReplayResult
  reference ReplayResult
  old_log_prob / mask resolved from TrajectoryBatch
  group_ids / context from TrajectoryBatch

-> TrajectorySignalBatch
```

## 目标 schema

新增到：

```text
vrl/models/interfaces/replay.py
```

### `ReplayRequest`

```python
@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Options for a recorded-trajectory replay forward."""

    segment_names: tuple[str, ...] | None = None
```

说明：

- `segment_names=None` 表示 replay training view 里的全部 trainable segments。
- R1 evaluator 可以传具体 segment。
- 不放 `need_ref` / `need_entropy` / `need_kl_intermediates`。这些属于 evaluator 的 `SignalRequest`，不是 model replay selection。

### `ReplaySegmentResult`

```python
@dataclass(slots=True)
class ReplaySegmentResult:
    """Current replay result for one trainable trajectory segment."""

    segment: str
    values: dict[str, Any] = field(default_factory=dict)
```

字段语义：

```text
segment:
  TrajectorySegment.name。必须能和 TrainingView.loss_units 对齐。

values:
  只放 current replay 产生的原始结果。
  examples:
    Janus: {"logits": logits, "token_ids": token_ids}
    NextStep: {"log_probs": log_probs, "tokens": tokens}
    diffusion: {"noise_pred": noise_pred}
  不能作为 old_log_prob/mask/reward source of truth。
```

### `ReplayResult`

```python
@dataclass(slots=True)
class ReplayResult:
    """Current replay result for a rollout batch."""

    segments: dict[str, ReplaySegmentResult]
    context: dict[str, Any] = field(default_factory=dict)
```

校验要求：

- `segments` 非空。
- dict key 必须等于 `ReplaySegmentResult.segment`。
- `ReplaySegmentResult` 不携带 distribution；distribution 的 source of truth 是
  `TrajectoryBatch.segments[segment].distribution`。
- `ReplayResult` 不携带 primary segment。primary 的 source of truth 是
  `TrainingView.primary_segment` / `TrajectorySignalBatch.primary_segment`。

### 新增 `ReplayModel`

当前 legacy interface：

```python
class Policy(Protocol):
    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]: ...
```

目标：

```python
class ReplayModel(Protocol):
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult: ...
    def disable_adapter(self) -> AbstractContextManager[None]: ...
```

`disable_adapter()` 仍属于 replay contract，因为 reference replay 需要它。

### 新增 `RuntimeModel`

新增到：

```text
vrl/models/interfaces/replay.py
```

```python
class RuntimeModel(ReplayModel, Protocol):
    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any: ...
```

`RuntimeBundle` 可以要求它的 family general inference model 满足这个最低公共
runtime contract：

```text
RuntimeBundle.model: RuntimeModel
```

短期为了迁移可以保留兼容 property：

```python
@property
def policy(self) -> RuntimeModel:
    return self.model
```

但新代码和文档不再使用 `policy` 命名。

使用边界：

```text
Evaluator:
  只依赖 ReplayModel。

Ray weight sync:
  从 RuntimeModel 上只调用 load_trainable_state(...)。

Executor:
  仍然可以依赖 concrete family model 的 generation methods，
  这些 methods 不进入 RuntimeModel interface。
```

## `ReplayResult` 到 `SegmentSignal` 的转换规则

不要新增一个“自动把 replay output 全量转 signal”的重型 helper。不同 family
从 raw replay result 到 current log_prob 的数学不同：

```text
categorical: logits -> log_softmax -> gather sampled token
gaussian: log_probs already computed by model replay
flow_matching: noise_pred -> sde_step_with_logprob(...)
```

新增轻量 helper 建议放到：

```text
vrl/rollouts/evaluators/replay_result.py
```

核心函数：

```python
def require_replay_segment(
    output: ReplayResult,
    segment_name: str,
) -> ReplaySegmentResult:
    ...

def require_replay_value(
    result: ReplaySegmentResult,
    key: str,
) -> Any:
    ...
```

evaluator 里的使用方式：

```python
current = require_replay_segment(output, "image_tokens")
logits = require_replay_value(current, "logits")
log_prob = compute_current_log_prob(logits, batch)
distribution = batch.trajectory.segments[current.segment].distribution

return segment_signal_from_batch(
    batch,
    segment_name=current.segment,
    log_prob=log_prob,
    ref_log_prob=ref_log_prob,
    distribution=distribution,
    timestep_idx=timestep_idx,
    aux=dict(current.values.get("aux", {})),
)
```

实现细节：

- old logprob 和 mask 必须从 `TrajectoryBatch` / `TrainingView` resolver 读，不从 `ReplayResult.values` 读。
- shape 校验沿用 `segment_signal_from_batch(...)` 和 `TrajectorySignalBatch.__post_init__`。
- `ReplayResult.segments` 缺 evaluator 需要的 segment 时必须 fail-fast。
- `ref_result.segments` 缺 current segment 时必须 fail-fast。
- evaluator 需要的 replay value 缺失时必须用 `require_replay_value(...)`
  fail-fast，不允许裸 `current.values["logits"]` 这种难定位的 `KeyError`。
- `ReplaySegmentResult.values` 只提供 current replay payload，不改变训练 source of truth。
- distribution 必须从 `TrajectoryBatch.segments[segment].distribution` 读，不从
  `ReplayResult` 读，避免重复 source of truth。

`require_replay_value(...)` 的错误信息需要包含 segment name、missing key 和已有 keys：

```python
raise KeyError(
    f"ReplaySegmentResult for segment {result.segment!r} "
    f"missing required key {key!r}; got {sorted(result.values)}"
)
```

## family 映射

### SD3 / Wan / Cosmos diffusion

目标输出：

```python
ReplaySegmentResult(
    segment="denoise",
    values={
        "noise_pred": noise_pred,
    },
)
```

变更含义：

- diffusion `replay_forward()` 不再返回 loose dict，而是返回 typed `ReplayResult`。
- `sde_step_with_logprob(...)` 暂时留在 `FlowMatchingEvaluator`，因为 scheduler / noise_level / sde_type 是 evaluator 当前拥有的信号计算语义。
- 本 sprint 不把 diffusion log_prob 计算下沉到 replay model。

需要特别处理：

- 现在 `FlowMatchingEvaluator` 持有 scheduler、noise_level、sde_type；迁移后也先保持这个边界。
- 如果后续单独决定把 diffusion log_prob 计算下沉，需要另开 sprint 处理 scheduler ownership。
- 不能让 `ReplayResult` 携带 old logprob / mask。

### Janus discrete AR

目标输出：

```python
ReplaySegmentResult(
    segment="image_tokens",
    values={"logits": logits, "token_ids": token_ids},
)
```

变更含义：

- `TokenLogProbEvaluator` 不再读 loose `out["logits"]`。
- `log_softmax + gather` 暂时保留在 evaluator，避免把 categorical signal math 下沉到 model contract。
- sampled action ids 从 `TrajectoryBatch` / `RolloutBatch.actions` 读取。
- logits 保留在 `values`，供 entropy 或 debug 使用。

### NextStep continuous AR

目标输出：

```python
ReplaySegmentResult(
    segment="image_tokens",
    values={"log_probs": log_probs, "tokens": tokens},
)
```

变更含义：

- NextStep 当前已经接近目标，因为它已经返回 `log_probs`。
- 先迁 NextStep 是最小风险路径。

### Janus-R1 multi-segment

目标输出：

```python
ReplayResult(
    segments={
        "initial_image": ReplaySegmentResult(...),
        "selfcheck_text": ReplaySegmentResult(...),
        "final_image": ReplaySegmentResult(...),
    },
)
```

变更含义：

- 当前 `MultiSegmentTokenLogProbEvaluator` 还调用 `replay_r1_segment(...)`。
- 目标是让 `JanusProModel.replay_forward(..., request=ReplayRequest(segment_names=...))` 返回多 segment `ReplayResult`。
- `replay_r1_segment(...)` 可以先保留为 model 内部 helper，但 evaluator 不再直接调用它。
- primary segment 继续由 `TrainingView.primary_segment` / `TrajectorySignalBatch.primary_segment`
  表达，不进入 `ReplayResult`。

## Phase 1：新增 schema 和 validation

编辑：

```text
vrl/models/interfaces/replay.py
vrl/models/interfaces/runtime.py
tests/models/test_replay_model_contract.py
tests/models/test_runtime_model_contract.py
```

新增：

```text
ReplayRequest
ReplaySegmentResult
ReplayResult
ReplayModel
RuntimeModel
```

更新：

```text
Policy -> ReplayModel / RuntimeModel
require_policy(...) -> require_replay_model(...)
require_runtime_model(...)
RuntimeBundle.policy -> RuntimeBundle.model
```

测试：

- `ReplayResult` 不能为空。
- segment key 必须等于 `ReplaySegmentResult.segment`。
- `ReplayResult` 不允许定义 `primary_segment`。
- `ReplayModel` runtime protocol 接受返回 `ReplayResult` 的 minimal class。
- `RuntimeModel` runtime protocol 接受 replay + `load_trainable_state(...)`。
- evaluator 只要求 `ReplayModel`，Ray weight sync 只调用 `RuntimeModel.load_trainable_state(...)`。

本 phase 可以先允许旧 dict 返回值通过 evaluator 兼容层，不一次性改完所有 family。

guard placement：

- `RuntimeBundle.model` 字段类型使用 `RuntimeModel`，但不要把所有场景都硬塞进
  `RuntimeBundle.__post_init__`。bundle 是装配对象，不应该替 executor 验证
  generation-only methods。
- common online recipe 取出 `bundle.model` 后调用 `require_runtime_model(...)`。
- evaluator 入口调用 `require_replay_model(...)`。
- Ray weight sync 调用 `require_runtime_model(...)` 后只使用 `load_trainable_state(...)`。
- executor 不调用 `require_replay_model(...)`；executor 依赖 concrete family model 的
  generation methods。

## Phase 2：新增 replay result helper

新增：

```text
vrl/rollouts/evaluators/replay_result.py
tests/rollouts/test_replay_result_signals.py
```

实现：

```text
require_replay_segment(...)
require_replay_value(...)
```

测试覆盖：

- single segment AR result lookup。
- single segment diffusion result lookup。
- multi segment R1 result lookup。
- missing segment fail-fast。
- missing replay value fail-fast with helpful key list。
- ref_result missing segment fail-fast。
- old_log_prob 和 mask 仍由 `segment_signal_from_batch(...)` 从 trajectory/training_view 取，不从 values 取。

## Phase 3：迁 NextStep

编辑：

```text
vrl/models/families/nextstep_1/model.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
tests/rollouts/test_replay_model_disable_adapter_evaluators.py
```

目标：

- `NextStep1Model.replay_forward()` 返回 `ReplayResult`。
- `ContinuousTokenLogProbEvaluator` 不再读 loose `out["log_probs"]`。
- evaluator 从 `ReplaySegmentResult.values["log_probs"]` 取 current log_prob，再用 `segment_signal_from_batch(...)` 组装 signal。

原因：

```text
NextStep replay 已经直接得到 per-token log_probs，迁移风险最低。
```

## Phase 4：迁 Janus discrete AR

编辑：

```text
vrl/models/families/janus_pro/model.py
vrl/rollouts/evaluators/ar/token_logprob.py
tests/models/test_janus_replay.py
tests/rollouts/test_replay_model_disable_adapter_evaluators.py
```

目标：

- `JanusProModel.replay_forward()` 返回 `ReplayResult`。
- `TokenLogProbEvaluator` 不再读 loose `out["logits"]`。
- logits 进入 `ReplaySegmentResult.values["logits"]`。
- `TokenLogProbEvaluator` 继续执行 `F.log_softmax(...).gather(...)` 生成 current log_prob。

要求：

- gathered log_prob shape 为 `[B, L]`。
- distribution 从 `TrajectoryBatch.segments["image_tokens"].distribution` 读取，
  expected value 是 `"categorical"`。
- signal axis 仍由 `segment_signal_from_batch(...)` / `TrainingView` 解析。

## Phase 5：迁 Janus-R1 multi-segment

编辑：

```text
vrl/models/families/janus_pro/model.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
tests/rollouts/test_multisegment_token_logprob.py
tests/models/test_janus_r1_model.py
```

目标：

- `ReplayRequest(segment_names=...)` 支持只 replay 指定 R1 segments。
- `JanusProModel.replay_forward()` 可以返回多个 segment。
- evaluator 不再直接调用 `replay_r1_segment(...)`。
- `replay_r1_segment(...)` 如果保留，只作为 model 内部 helper。

完成后：

```text
MultiSegmentTokenLogProbEvaluator:
  current_output = model.replay_forward(...)
  ref_output = ...
  compute per-segment log_probs from ReplaySegmentResult.values
  return TrajectorySignalBatch(...)
```

## Phase 6：迁 diffusion

编辑：

```text
vrl/models/diffusion/model_base.py
vrl/models/families/sd3_5/model.py
vrl/models/families/wan_2_1/model.py
vrl/models/families/cosmos/predict2/model.py
vrl/models/families/cosmos/predict2_5/model.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
tests/models/test_diffusion_model_base.py
```

目标：

- diffusion `replay_forward()` 返回 `ReplayResult`。
- `FlowMatchingEvaluator` 不再读 loose `fwd["noise_pred"]`。
- `FlowMatchingEvaluator` 从 `ReplaySegmentResult.values["noise_pred"]` 取 raw prediction。
- `sde_step_with_logprob(...)` 仍由 evaluator 调用。

需要先解决的设计点：

```text
FlowMatchingEvaluator 现在持有 scheduler/noise_level/sde_type。
本 sprint 不迁移这些 ownership；只迁 replay_forward 返回容器。
```

推荐做法：

- scheduler / noise_level / sde_type 继续由 `FlowMatchingEvaluator` 持有。
- 如果后续要让 replay model 直接输出 flow-matching log_prob，需要单独设计 scheduler ownership。
- 本 sprint 只消除 loose replay dict。

## Phase 7：删除旧 dict 兼容

清理：

```text
out["logits"]
out["log_probs"]
fwd["noise_pred"]
```

这些只能作为 `ReplaySegmentResult.values[...]` 中的 named payload，不能作为 loose replay dict 主路径。

扫描 gate：

```text
rg 'replay_forward\\([^\\n]*\\).*\\[\"(logits|log_probs|noise_pred)\"' vrl tests
rg 'out\\[\"logits\"\\]|out\\[\"log_probs\"\\]|fwd\\[\"noise_pred\"\\]' vrl tests
```

完成后应该无主路径命中。

## 不做的事

本 sprint 不做：

- 不改 executor generation API。
- 不改 `TrajectoryBatch` schema。
- 不把 algorithm 改成直接吃 `ReplayResult`。
- 不把 reward / advantage 放进 `ReplayResult`。
- 不新增 `DiffusionPolicy` / `ARPolicy` inheritance base。
- 不把 concrete `*Model` 改名成 `ReplayModel`。`model.py` 仍是 general inference object。
- 不做 `StagedPolicy`。stage orchestration 应该先留在 recipe/trainer 层。

## 完成标准

完成后应该满足：

- `ReplayModel.replay_forward()` public contract 返回 `ReplayResult`。
- `Policy` public interface 名称被移除或降级为兼容别名；新代码使用 `ReplayModel` / `RuntimeModel`。
- `RuntimeBundle.model` 是 family general inference object；`RuntimeBundle.policy` 只能作为短期兼容 property 存在。
- production evaluator 不再读取 replay dict key：
  - `fwd["noise_pred"]`
  - `out["logits"]`
  - `out["log_probs"]`
- evaluator 通过 `ReplayResult` / `ReplaySegmentResult` 读取 typed replay payload。
- evaluator 只依赖 `ReplayModel`。
- Ray weight sync 只调用 `RuntimeModel.load_trainable_state(...)`。
- `SegmentSignal.old_log_prob` 和 `SegmentSignal.mask` 只从 `TrajectoryBatch` / `TrainingView` resolver 获取。
- Janus / NextStep / diffusion / Janus-R1 至少各有一个测试覆盖 evaluator 使用 `ReplayResult` 组装 `TrajectorySignalBatch`。
- `TrajectorySignalBatch` 仍是 algorithm input。
- SD3.5 OCR baseline gate 不退化。

## 验证计划

至少运行：

```text
pytest tests/models/test_replay_model_contract.py
pytest tests/models/test_runtime_model_contract.py
pytest tests/rollouts/test_replay_result_signals.py
pytest tests/rollouts/test_replay_model_disable_adapter_evaluators.py
pytest tests/rollouts/test_multisegment_token_logprob.py
pytest tests/models/test_janus_replay.py tests/models/test_janus_r1_model.py
pytest tests/models/test_diffusion_model_base.py
pytest tests/trainers/test_online.py tests/trainers/test_weight_sync.py
python -m compileall vrl tests
git diff --check
```

最后跑 legacy scan：

```text
rg 'out\\[\"logits\"\\]|out\\[\"log_probs\"\\]|fwd\\[\"noise_pred\"\\]' vrl tests
```

如果还有命中，必须确认只是 `ReplaySegmentResult.values` 的测试或 debug，不是 loose dict evaluator 主路径。

## 关键文件参考

目标接口：

```text
vrl/models/interfaces/replay.py
vrl/models/interfaces/runtime.py
```

当前 signal schema：

```text
vrl/rollouts/evaluators/types.py
```

当前 evaluator 主路径：

```text
vrl/rollouts/evaluators/diffusion/flow_matching.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
```

当前 trajectory resolver / view：

```text
vrl/engine/trajectory/views.py
vrl/engine/trajectory/resolver.py
vrl/rollouts/evaluators/trajectory.py
```
