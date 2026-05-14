# SPRINT：通用 replay 防线与 SD3.5 host RAM OOM guard

状态：done。

## 核心结论

这次 OOM 的根因仍然是：

```text
trainer 进程加载完整 generation policy
Ray rollout worker 进程也加载完整 generation policy
```

在单 GPU colocated Ray 设置下，两份完整模型会同时占用 host RAM。旧 run 的直接症状是 Ray host memory 超过 0.95 kill threshold，然后杀掉正在初始化的 rollout worker。

但这份 sprint 不再实现“每个 family 怎么加载最小 replay module”。那部分会进入独立 sprint：

```text
SPRINT_family_minimal_replay_module_loading.md
```

本 sprint 只做通用基础设施：

```text
1. trainer / rollout weight sync 只能走 trainable-only payload
2. RuntimeBundle 显式声明当前 bundle 是否加载 full generation modules
3. common recipe 记录 host memory，并在 colocated Ray + full generation trainer bundle 时给出 guard
4. trainer replay policy contract 先定义清楚，但不落具体 family loader
```

所以这份 sprint 单独完成后，不应该 claim “SD3.5 trainer RSS 已经根治”。真正降低 trainer RSS 的动作是 family-specific minimal replay loader sprint。

## 保留目标

必须保留：

```text
单 GPU 仍然使用 Ray 起第二个完整 rollout worker 进程
```

不能通过这些方式假装修复：

```text
取消 Ray worker
单 GPU 改回 in-process rollout
降低 rollout.n / sample_batch_size
依赖 RAY_memory_usage_threshold=0.99
让 trainer 重新从 OutputBatch.extra 读 replay state
```

## 本 sprint 不做的事

不新增这些文件：

```text
vrl/models/families/sd3_5/train_policy.py
vrl/models/families/sd3_5/trainer_runtime.py
tests/models/test_sd3_train_policy.py
tests/models/test_sd3_trainer_runtime.py
```

不实现这些类型：

```text
SD3_5TrainPolicy
SD3_5ReplayPolicy
WanReplayPolicy
CosmosReplayPolicy
JanusReplayPolicy
NextStepReplayPolicy
```

原因是这些不是 shared infra。它们属于“每个 family 怎么最小加载 replay module”的模型侧实现，应该在独立 sprint 里按 family 分析并逐个落地。

## 通用架构边界

当前先把系统切成两层：

```text
shared trainer/runtime infra
  - ReplayPolicy protocol
  - RuntimeBundle metadata
  - trainable-only rollout sync payload
  - host memory logging / colocated guard

family-specific minimal replay loader
  - SD3.5 transformer-only trainer bundle
  - Wan transformer-only trainer bundle
  - Cosmos transformer-only trainer bundle
  - Janus language-model/image-head replay bundle
  - NextStep language-model/flow-head replay bundle
```

本 sprint 只实现第一层。

## 文件改动范围

### 新增文件

```text
vrl/models/interfaces/replay_policy.py
vrl/trainers/memory.py
tests/trainers/test_memory_guards.py
SPRINT_family_minimal_replay_module_loading.md
```

### 主要编辑文件

```text
vrl/models/interfaces/runtime.py
vrl/models/interfaces/__init__.py
vrl/models/__init__.py
vrl/models/interfaces/diffusion_policy.py
vrl/trainers/weight_sync.py
vrl/trainers/online.py
vrl/scripts/common/online.py
vrl/rollouts/runtime/backend.py
vrl/distributed/ray/rollout/worker.py
vrl/models/families/*/runtime.py
tests/trainers/test_weight_sync.py
tests/trainers/test_online.py
```

### 明确不改的文件

```text
configs/experiment/sd3_5_ocr_grpo.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
vrl/engine/trajectory/*
vrl/rollouts/packers/trajectory.py
```

原因：

- 当前 SD3.5 OCR recipe 已经能继续跑，不能靠改 batch/config 掩盖 host RAM 结构问题。
- strict trajectory 主路径已经成立，本 sprint 不能把 replay source of truth 拉回 legacy extras。
- single-GPU Ray preset 要保留二进程一致性。

## Phase 1：ReplayPolicy contract

新增：

```text
vrl/models/interfaces/replay_policy.py
```

最小 contract：

```text
replay_forward(batch, timestep_idx)
disable_adapter()
load_trainable_state(state_dict)
```

这不是 rollout policy contract。rollout policy 可以拥有 prompt encoder、VAE/VQ decoder、scheduler、sampling loop；replay policy 只表达 trainer replay 和 rollout weight sync 需要的最小能力。

完成标准：

- `ReplayPolicy` 从 `vrl.models.interfaces` 和 `vrl.models` 导出。
- 不要求现有 family 立刻实现新的 minimal replay loader。
- 不新增 family-specific replay policy 文件。

## Phase 2：RuntimeBundle metadata

给当前 full generation bundle 标记：

```text
runtime_role: full_generation_policy
loads_full_generation_modules: true
requires_minimal_replay_loader: true
```

这不是最终优化，只是让 shared infra 能看见风险。未来 minimal replay loader 应该返回：

```text
runtime_role: minimal_replay_policy
loads_full_generation_modules: false
requires_minimal_replay_loader: false
```

完成标准：

- SD3.5 / Wan / Cosmos / Janus / NextStep 当前 runtime bundle 都显式声明 full generation role。
- guard 不靠 family name 猜测行为。

## Phase 3：trainable-only weight sync

旧问题：

```python
await self.weight_syncer.push(self.model.state_dict())
```

这会把“model 全量 state_dict”变成默认 rollout sync 语义。对于大模型，这很容易把 frozen backbone / generation-only state 推进 Ray object store。

新路径：

```text
RuntimeBundle.trainable_modules
  -> flatten trainable parameters only
  -> RayRuntimeWeightSyncer.push(...)
  -> rollout policy.load_trainable_state(...)
```

完成标准：

- `OnlineTrainer` 有 weight sync 时必须传入显式 trainable-state getter。
- shared recipe 从 `RuntimeBundle.trainable_modules` 构造 sync payload。
- sync payload 只包含 `requires_grad=True` 参数。
- diffusion `load_trainable_state` 接受 trainable-only payload，不再要求完整 transformer state_dict。

## Phase 4：host memory instrumentation 与 colocated guard

新增：

```text
vrl/trainers/memory.py
```

记录点：

```text
before_bundle_build
after_bundle_build
before_rollout_backend_build
after_rollout_backend_build
ray_worker:<id>:before_load_policy
ray_worker:<id>:after_load_policy
```

guard 语义：

```text
if colocated Ray GPU rollout and driver bundle loads_full_generation_modules:
    warn by default
    raise when VRL_STRICT_REPLAY_MEMORY_GUARD=1
```

默认不直接 fail 的原因：

```text
当前还没有 family-specific minimal replay loader。
如果默认 fail，现有 SD3.5 OCR 真实训练会被阻断，但 trainer RSS 仍然不会下降。
```

严格模式的用途：

- CI 中防止未来新增 recipe 又把 full generation policy 放回 trainer。
- minimal replay loader 落地后，把对应 recipe 切到 strict guard。

## 验证

已通过：

```text
pytest tests/trainers/test_weight_sync.py tests/trainers/test_memory_guards.py tests/trainers/test_online.py
python -m compileall vrl
git diff --check
```

当前 sprint 完成后应该满足：

- trainer 不再默认同步 `model.state_dict()`。
- Ray weight sync payload 是 trainable-only。
- common recipe 能在日志里看到 trainer / Ray worker host memory。
- colocated Ray + full generation trainer bundle 有显式 warning / strict guard。
- 没有新增任何 SD3.5 专属 minimal replay loader。

当前 sprint 完成后仍然不能 claim：

```text
SD3.5 trainer 进程已经 transformer-only
SD3.5 host RAM OOM 已经架构级根治
所有 family 都已经 minimal replay loading
```

这些属于 `SPRINT_family_minimal_replay_module_loading.md`。
