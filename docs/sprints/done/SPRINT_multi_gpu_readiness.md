# SPRINT: Multi-GPU Training Readiness

状态：已完成 / readiness landed。P1-P6 全部落在 main：schema+context d19faa2（TrainingSection.strategy=Literal[single_process|fsdp]、DistributedTrainingContext、resolve_training_context、assert_strategy_executable fail-fast），P2 资源校验 b0d7b57（_validate_trainer_device_count/_validate_fsdp_trainer_disjoint），P3 strategy seam 0d1b046/001ab41/1e6dc24（Strategy + SingleProcessStrategy，已 wire 进 trainer.py），P4 collect/train 拆分 f979cce（collect_training_batch/train_on_rollout_batch/TrainingBatch），P5 rollout state 去前缀 fea4ba9（flatten_trainable_module_state 剥离 _orig_mod/module.），P6 rank0<->Ray ownership spike e6facbd。78 个相关测试全绿；FSDP2 fully_shard 后续已由 `done/SPRINT_multi_gpu_training.md` 落地并完成真实多卡验收。
sprint：在还没有真实 multi-GPU 硬件时，把配置、资源解析、trainer 边界、checkpoint /
rollout weight sync 的契约先准备好。完成后，拿到多 GPU 时应该能直接进入 FSDP2 真实
运行验证，而不是先处理 schema、resource resolver、rank0/Ray ownership 这些基础问题。

关联：
- `docs/sprints/done/SPRINT_multi_gpu_training.md`
- `docs/sprints/done/SPRINT_global_ray_placement_owner.md`
- `docs/sprints/reading/SPRINT_reward_execution.md`

## 0. Core Decision

本 sprint **不实现 FSDP2**，也不宣称 VRL 已经支持 multi-GPU training。它只做一件事：
把当前 single-process trainer 变成一个可以安全接入 distributed training 的结构。

核心判断：

1. **multi-GPU training 不是先写 FSDP wrapper。** 当前代码在资源层、配置层、trainer
   step 边界、state export 边界都会先挡住或污染 FSDP 接入。先消这些确定风险。
2. **没有多卡也能完成 readiness。** schema、resolver validation、rank context、rank0
   gating、fake wrapped module export、checkpoint policy gate 都可以在 CPU / single GPU /
   monkeypatched env 下测试。
3. **真实 FSDP2 留给后续 sprint。** 这份 readiness 的输出是干净的接口和 fail-fast
   gate，不是半成品 FSDP。

完成后的状态：

```text
single_process behavior unchanged
distributed.training config accepted and validated
trainer multi-GPU resource requests are gated by training strategy
OnlineTrainer collect/train boundaries are separable
checkpoint and rollout sync have a strategy-aware export seam
rank0-only Ray ownership is specified and tested with fakes
FSDP unsupported paths fail-fast with actionable errors
```

## 1. Current Blockers

### B1. trainer 多卡现在会在 resource resolver 直接失败

当前 resolver 明确拒绝 trainer 拿多于 1 张 GPU：

```python
if len(trainer_devices) > 1:
    raise ValueError(
        "distributed.resources.trainer.devices currently supports only "
        f"0 or 1 GPU for the single-process trainer, got {trainer_devices}",
    )
```

这意味着后续 sprint 里的命令还没进入 FSDP 逻辑，就会在资源解析阶段失败：

```bash
torchrun --nproc-per-node=2 -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  distributed.training.strategy=fsdp \
  distributed.resources.trainer.num_gpus=2
```

Readiness 要先把这条 hard cap 改成 strategy-aware validation：

```text
single_process -> trainer 只能 0/1 GPU
fsdp           -> trainer 可以多 GPU，但必须匹配 training world size
```

### B2. `distributed.training` schema 现在不存在

当前 `DistributedSection` 只有：

```text
resources
rollout
reward
```

所以 `distributed.training.strategy=fsdp` 不是一个可验证的配置面。readiness 要先新增
schema，但默认必须保持 `single_process`，并且不允许 config 里出现投机值：

```text
allowed: single_process | fsdp
rejected: any other strategy value
```

不为非 FSDP strategy 预留 schema 空槽。这个 sprint 的目标是给 FSDP2 铺路，不提前扩展
一组暂时没有实现和测试承诺的 backend 名称。

### B3. `OnlineTrainer.step()` 现在把 collect 和 train 绑在一起

multi-GPU online GRPO 的真实边界是：

```text
rank0:
  collect rollout
  score rewards
  compute global advantages
  broadcast / provide training payload

all ranks:
  shard payload without breaking prompt groups
  replay forward/backward
  optimizer step

rank0:
  export trainable state
  push rollout weights
  log/checkpoint/eval
```

当前 `OnlineTrainer.step()` 在一个流程内完成 loss、backward、optimizer、EMA、
global_step、rollout sync：

```python
self._backward(loss)
_gn, _stepped = self._clip_and_step(optimizer)
after_optimizer_step(...)
sync_phase_times = await self.rollout_schedule.after_train_step()
```

Readiness 要先做无行为变化拆分。否则真正接 FSDP 时，会同时改变训练逻辑和分布式逻辑。

### B4. checkpoint / rollout sync 直接假设普通 `state_dict()`

rollout sync 当前从 `RuntimeBundle.trainable_modules` 展平普通 module state：

```python
module_state = state_dict()
state[f"{name}.{key}"] = value
```

checkpoint 当前也是直接保存普通 trainable module `state_dict()`：

```python
out[name] = to_cpu(state_dict())
```

FSDP2 / DTensor 下，trainer 侧内部 state 不能直接泄漏给 rollout policy。readiness 要先
定义 strategy-aware export seam，保证 rollout-facing state 永远是普通、unwrapped、
policy-facing key space。

## 2. Scope

本 sprint 覆盖：

- 新增 `distributed.training` schema 和 base config。
- 新增 distributed training context resolver，但只要求 `single_process` 完整可用。
- resource resolver 根据 training strategy 校验 trainer GPU 数。
- `OnlineTrainer.step()` 做 collect/train 的无行为变化拆分。
- 建立 strategy-aware state export/load seam，先接 `SingleProcessStrategy`。
- fake wrapper / fake rank tests 覆盖 wrapper key 不泄漏、rank0-only Ray ownership。
- 为 FSDP 未实现路径加 fail-fast gate。

本 sprint 不覆盖：

- 不实现 `torch.distributed.fsdp.fully_shard`。
- 不做真实 2-GPU run。
- 不加入 Megatron / DeepSpeed / Ray Train。
- 不新增 `vrl/distributed/` 顶层包。
- 不把 rollout worker 改成 FSDP shard consumer。
- 不重写 checkpoint 格式为 sharded checkpoint。

## 3. Target Config

新增 base config：

```text
configs/base/distributed/training_single_process.yaml
configs/base/distributed/training_fsdp.yaml
```

目标 schema：

```yaml
distributed:
  training:
    strategy: single_process     # single_process | fsdp
    launcher: none               # none | torchrun
    num_nodes: 1
    gpus_per_node: 1
    backend: nccl
    init_method: env

    fsdp:
      mesh: ["dp_shard"]
      precision_policy: actor
      reshard_after_forward: true
      activation_checkpointing: actor
      cpu_offload: false
      state_dict: full_rank0
```

Readiness 只要求：

- `single_process` 是默认值，现有 config 不需要改。
- `strategy=fsdp` 可以被 schema 接受，但启动时明确 fail-fast：

```text
distributed.training.strategy=fsdp is configured, but FSDP2 execution is not implemented yet.
Complete SPRINT_multi_gpu_training.md after this readiness sprint.
```

- `strategy=fsdp` 时必须能解析 torchrun env：

```text
RANK
LOCAL_RANK
WORLD_SIZE
```

缺失时给出明确错误，而不是在 CUDA device 或 Ray launch 阶段才失败。

## 4. Resource Resolver Contract

资源解析继续以 `vrl/ray/resources.py` 为 source of truth，不新增第二套 placement
planner。readiness 只把 trainer 多卡的规则从硬编码改成 strategy-aware。

规则：

```text
single_process:
  trainer_devices length must be 0 or 1
  trainer_torch_device() keeps returning cpu or cuda:<single-device>

fsdp:
  trainer_devices length must equal num_nodes * gpus_per_node
  torchrun WORLD_SIZE must equal num_nodes * gpus_per_node
  LOCAL_RANK must map into trainer_devices
```

rollout/reward 资源规则保持现在的 role-level resolver，不在本 sprint 重新设计：

```text
rollout/reward placement owner stays under vrl/ray/placement.py
reward GPU-pool resolution stays under vrl/ray/resources.py
reward transport/overlap contract stays under done/SPRINT_reward_service.md
cross-node Ray placement validation stays in existing Ray resource path
```

`strategy=fsdp` 下必须新增校验：

```text
trainer GPU set ∩ rollout GPU set = empty
trainer GPU set ∩ reward GPU set = empty
```

除非显式满足已有 overlap/release 语义。错误信息要打印 trainer/rollout/reward 三组设备。
`strategy=single_process` 保持现有 colocated debug 路径，不因为这条 FSDP 专属校验误伤
单卡 trainer+rollout 共享 GPU 的配置。

## 5. Training Context

新增一个窄的 context，不读取 Ray，不创建 process group，不 wrap model：

```text
vrl/trainers/distributed.py
```

它放在 `vrl/trainers/`，不是 `vrl/ray/`。这个 context 描述的是 torchrun/FSDP 训练进程
身份；Ray placement、Ray actor lifecycle 仍然留在 `vrl/ray/`。

接口形状：

```python
@dataclass(frozen=True, slots=True)
class DistributedTrainingContext:
    strategy: str
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_primary: bool
    device: torch.device
```

规则：

- train script 和 trainer 不直接读 `RANK` / `LOCAL_RANK` / `WORLD_SIZE`。
- `single_process` 返回 `rank=0, local_rank=0, world_size=1, is_primary=True`。
- `fsdp` 在 readiness 阶段只解析并校验 env，然后 fail-fast 到后续 sprint。
- context 不命名为 scheduler。它不调度任务，只描述当前训练进程身份。

## 6. Strategy Seam

新增 strategy protocol，但 readiness 只实现 `SingleProcessStrategy`。

```text
vrl/trainers/strategy.py
```

目标接口：

```python
class TrainingStrategy(Protocol):
    context: DistributedTrainingContext

    def backward(self, loss: torch.Tensor) -> None: ...
    # clip + grad-scaler-aware optimizer step + zero_grad; returns (grad_norm, stepped)
    def clip_and_step(self, optimizer: torch.optim.Optimizer) -> tuple[float, bool]: ...
    def export_trainable_state(self, bundle: RuntimeBundle) -> dict[str, dict[str, Any]]: ...
    def export_rollout_state(self, bundle: RuntimeBundle) -> dict[str, Any]: ...
    def load_trainable_state(self, bundle: RuntimeBundle, state: dict[str, Any]) -> None: ...
    def barrier(self) -> None: ...
```

`SingleProcessStrategy` 只是把当前行为搬进去：

```text
backward -> existing OnlineTrainer._backward semantics
clip_and_step -> existing _clip_and_step (clip + grad-scaler-aware optimizer step + zero_grad); returns (grad_norm, stepped)
export_trainable_state -> checkpointing.export_trainable_state
export_rollout_state -> weight_sync.flatten_trainable_module_state
load_trainable_state -> checkpointing.load_trainable_state
barrier -> no-op
```

`backward` 和 `clip_and_step` 都按 `self._grad_scaler` / `self.accelerator` 分派：`_backward`
有 grad_scaler / accelerator / 裸 `loss.backward()` 三分支（trainer.py:326），`_clip_and_step`
在 fp16 GradScaler 下走 `scaler.step/update` 并据缩放是否回退判定是否真的 stepped
（trainer.py:334）。所以 `SingleProcessStrategy` 必须持有这两个句柄，不能简化成
`loss.backward()` + `clip_grad_norm_`。返回的 `stepped` 是 load-bearing：trainer 只在 stepped
时跑 EMA / `after_optimizer_step` / 推进 global_step（trainer.py:813-822）；丢掉 `stepped`，
P3「输出和当前 helper 一致」就不再是无行为变化。

这看起来是薄层，但它是必要边界：

- protocol boundary：trainer 后续不直接知道 FSDP。
- public seam：checkpoint 和 rollout sync 都从这里导出 state。
- cross-family consistency：online/offline trainer 后续可以走同一 strategy 入口。

## 7. OnlineTrainer Decomposition

先做无行为变化拆分，目标不是改训练算法，而是把未来 rank split 的边界露出来。

目标形状：

```python
async def collect_training_batch(...) -> TrainingBatch:
    ...

async def train_on_rollout_batch(batch: TrainingBatch) -> TrainStepMetrics:
    ...
```

`OnlineTrainer.step()` 在 single-process 下继续串行调用：

```text
batch = await collect_training_batch(...)
metrics = await train_on_rollout_batch(batch)
await rollout_schedule.after_train_step()
return metrics
```

必须保持：

- metrics header 不变。
- `state.step` / `state.global_step` 更新语义不变。
- EMA 调用语义不变。
- `rollout_schedule.after_train_step()` 调用时机不变。
- continuous rollout 的 asyncio yield 行为不变。

`train_on_rollout_batch()` 必须保持 async。当前训练内循环靠 `await asyncio.sleep(0)` 在
per-timestep 计算之间让 continuous rollout producer 在同一个 asyncio loop 上继续推进；
如果把 train 段做成同步函数，readiness refactor 就已经改变了 rollout 交织行为。

测试用 fake collector / fake algorithm 锁住：

```text
single-process step metrics unchanged
global_step increments only after optimizer-stepped path
rollout weight sync still happens after train step
collector stats still merge into phase_times
```

## 8. State Export Contract

readiness 的关键产物是一个明确 state contract：

```text
checkpoint state:
  nested by trainable module name
  CPU tensors
  compatible with existing checkpoint format in single_process

rollout state:
  flat policy-facing keys
  no wrapper "module." prefix
  no torch.compile "_orig_mod." prefix
  no FSDP shard / DTensor internal key
  only trainable parameters unless a strategy explicitly opts into full state
```

新增测试：

- fake wrapped module 的 `state_dict()` 带 `module.` 前缀时，export seam 不能把前缀泄漏给 rollout。
- compiled-like `_orig_mod` wrapper 仍保持现有行为。
- empty trainable state 继续 fail-fast。
- checkpoint export/load 的 single-process 格式不变。

如果 `fsdp` 被配置但 readiness 尚未实现 FSDP full-state export，必须 fail-fast：

```text
distributed.training.strategy=fsdp requires strategy-aware full-state export.
This readiness sprint only installs the seam; run SPRINT_multi_gpu_training.md for FSDP2.
```

## 9. Rank0 ↔ Ray Ownership Spike

不需要真实 Ray cluster 或多 GPU；用 fake runtime 测协同规则。

约定：

```text
rank0:
  owns Ray client interactions
  calls rollout runtime generate/update_weights
  writes metrics/checkpoints/eval outputs

non-rank0:
  never calls Ray generate/update_weights
  waits at strategy barrier or consumes broadcast payload
  never writes duplicate metrics/checkpoints/eval outputs
```

测试：

- monkeypatch `RANK=0/1`, `LOCAL_RANK=0/1`, `WORLD_SIZE=2`。
- fake Ray runtime 记录调用次数。
- rank0 path 调用 generate。
- non-rank0 path 不调用 generate。
- weight sync 只在 primary rank 推送。

这不是完整 distributed integration test，只是把 ownership contract 固化，避免后续 FSDP
实现时出现 N 个 rank 同时连 Ray、重复采样、重复写 checkpoint。

## 10. Checkpoint / Resume Policy Gates

readiness 要先决定哪些组合在 FSDP 第一版前必须报错：

```text
fsdp + EMA enabled               -> fail-fast until DTensor-aware EMA is implemented
fsdp + optimizer resume          -> fail-fast until FSDP optimizer state export/load is implemented
fsdp + unsupported model family  -> fail-fast until trainable root is explicit
fsdp + missing torchrun env      -> fail-fast
```

这些 gate 要进测试。不要等真实多卡时才发现 silent partial support。

single-process checkpoint/resume 必须完全不变。

## 11. Implementation Phases

### P0. Baseline and characterization

完成条件：

- 记录一条 single-GPU SD3 OCR online training baseline：config、seed、reward curve、
  checkpoint/resume verdict。
- 增加 characterization tests，锁住当前 resource resolver、checkpoint export、rollout
  sync、OnlineTrainer step metrics。

### P1. Config schema and training context

完成条件：

- `distributed.training` 被 schema 接受。
- `single_process` 默认行为不变。
- `fsdp` env parsing 可测试，但执行路径明确 fail-fast。
- unknown strategy 报清楚错误。

### P2. Resource validation

完成条件：

- `single_process + trainer.num_gpus=2` 继续 fail-fast。
- `fsdp + trainer.num_gpus=2 + WORLD_SIZE=2` 通过资源校验，随后停在 FSDP not implemented gate。
- `fsdp + trainer.num_gpus != WORLD_SIZE` fail-fast。
- trainer/rollout/reward overlap 的错误信息包含三组设备。

### P3. SingleProcessStrategy seam

完成条件：

- trainer backward / clip+step / checkpoint export / rollout export 经过 strategy seam。
- `SingleProcessStrategy` 输出和当前 helper 输出一致（含 `_clip_and_step` 返回的 `stepped`）。
- 没有 FSDP wrapper 代码进入本阶段。

### P4. OnlineTrainer collect/train split

完成条件：

- `OnlineTrainer.step()` 单卡行为不变。
- collect 和 train 逻辑可分别测试。
- rank0-only output 的未来边界有测试覆盖。

### P5. State export and fake wrapper tests

完成条件：

- fake wrapped module 不泄漏 wrapper key。
- rollout-facing state 仍能被 policy `load_trainable_state()` 使用。
- checkpoint nested state 和 rollout flat state 的差异有独立测试。

### P6. torchrun/Ray ownership spike

完成条件：

- monkeypatched rank tests 证明只有 primary rank 调 Ray runtime。
- non-primary rank 不写 metrics/checkpoint/eval。
- barrier/broadcast seam 已存在，即使当前 single_process 是 no-op。

## 12. Acceptance Criteria

代码层：

- 现有 single-process configs 不需要改，行为不变。
- 新增 `distributed.training` 后，配置系统不会把合法 training keys 当 unknown。
- `vrl/ray/resources.py` 不再把 trainer 多卡 hard-code 成全局非法；它按 strategy 判定。
- trainer 不直接读取 distributed env。
- checkpoint 和 rollout sync 不直接依赖裸 module state as the only path；它们可走
  strategy-aware export seam。
- FSDP 未实现组合都有明确 fail-fast，不出现半支持。

测试层：

```bash
python -m pytest -q tests/config
python -m pytest -q tests/ray/test_resources.py
python -m pytest -q tests/trainers/test_distributed_training.py
python -m pytest -q tests/trainers/test_strategy.py
python -m pytest -q tests/trainers/test_checkpointing.py
python -m pytest -q tests/trainers/test_weight_sync.py
python -m pytest -q tests/trainers/test_online.py
```

手动验证：

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr
```

仍然走 single-process。

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  distributed.training.strategy=fsdp \
  distributed.resources.visible_devices=[0,1] \
  distributed.resources.trainer.num_gpus=2
```

应通过 schema / resource / context preflight，然后在 FSDP not implemented gate 明确停止。

## 13. Architecture Hygiene

应该改变：

- 新增 `distributed.training` typed config block；不要把 allowed strategy 写成散落的
  ALL_CAPS 手工集合，优先从 schema Literal / dataclass fields 派生。
- 新增 `vrl/trainers/distributed.py` 和 `vrl/trainers/strategy.py` 作为 protocol
  boundary。
- 让 checkpoint / rollout sync 走 strategy-aware export seam。
- 把 `OnlineTrainer.step()` 拆成 collect 和 train 两个可测试边界。

应该保持不变：

- `vrl/ray/resources.py` 继续是 role resource source of truth。
- `vrl/ray/placement.py` 继续只管 Ray placement owner，不承担 torchrun scheduler 职责。
- `OnlineTrainer` 仍然是 online GRPO orchestration owner；本 sprint 不把 trainer 变成
  Ray actor。
- `RayGenerationRuntime` / `RayRewardRuntime` 的 thin runtime boundary 保留。它们是
  framework adapter，不是可以为了减少文件数而压扁的 helper。

为什么这些薄层必要：

- `vrl/trainers/distributed.py` 是 process identity boundary。
- `strategy.py` 是 trainer 与 distributed backend 的 protocol boundary。
- Ray runtime / placement 文件是 framework adapter 和 lifecycle owner boundary。
- 保留这些薄层能让后续 FSDP、Ray rollout、checkpoint 三条线分别 grep 和测试。

Non-goals：

- 不为了 LOC 合并 `vrl/ray/resources.py`、`vrl/ray/placement.py`、
  `vrl/trainers/distributed.py`、`vrl/trainers/strategy.py`。
- 不维护一份会和 schema 腐烂的 `_ALLOWED_STRATEGIES = {...}`。
- 不新增泛化过度的 scheduler/manager 命名；名字按实际职责取。

## 14. References

当前代码切点：

- `/home/mingfeiguo/Desktop/wm-infra/vrl/ray/resources.py:108`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/ray/resources.py:114`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/ray/resources.py:253`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/config/schema.py:336`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py:783`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py:806`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py:876`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/weight_sync.py:85`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/weight_sync.py:101`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/checkpointing.py:250`

设计参考：

- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/done/SPRINT_multi_gpu_training.md`
- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/done/SPRINT_global_ray_placement_owner.md`
- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/reading/SPRINT_reward_execution.md`
- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/reading/cosmos-rl.md`
- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/reading/SPRINT_cosmos_rl_scaling_learnings.md`
