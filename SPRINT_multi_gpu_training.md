# SPRINT: Multi-GPU Training

## 0. Core Decision

本 sprint 的目标是让训练侧支持真正的 multi-GPU training，而不是只把 rollout worker 分到多张卡。

先支持两类 PyTorch 原生策略：

```text
single_process
ddp
fsdp
```

`megatron` 只保留 schema 和设计边界，不在第一阶段实现。原因是 Megatron 不是简单的 wrapper；它要求 tensor/pipeline parallel 模型切分、Megatron checkpoint 格式、optimizer/state sharding、权重同步协议和 family model adapter 都重新定义。它应该是后续独立 backend，而不是混在 DDP/FSDP sprint 里。

本 sprint 依赖 `SPRINT_distributed_resource_config.md` 的资源解析结果，但职责不同：

- `SPRINT_distributed_resource_config.md`：决定 trainer / rollout 拿多少资源。
- 本 sprint：决定 trainer 如何在这些资源上创建 rank、wrap model、shard batch、同步梯度、保存 checkpoint、向 rollout workers 推送权重。

## 1. Current Code Reality

当前训练入口是 single-process：

```python
trainer = OnlineTrainer(
    model=sd3_5_model,
    ...
)
```

`OnlineTrainer` 直接从 `self.model.parameters()` 建 optimizer：

```python
trainable = [p for p in self.model.parameters() if p.requires_grad]
self._optimizer = _create_optimizer(trainable, self.config)
```

rollout weight sync 直接推 `self.model.state_dict()`：

```python
await self.weight_syncer.push(self.model.state_dict())
```

checkpoint 依赖 `RuntimeBundle.trainable_modules`：

```python
"trainable_modules": export_trainable_state(bundle)
```

所以 multi-GPU training 不能只是把 policy 外面套一层 DDP/FSDP。必须同时解决：

- trainer rank / local rank / world size。
- model wrapper 放在哪一层。
- rollout batch 如何广播 / 分片。
- checkpoint 如何 rank0-only 保存，FSDP 如何导出 full state。
- rollout workers 需要收到普通 `state_dict`，不能收到 FSDP shard。
- EMA、LoRA export、resume 必须继续正确。

## 2. Scope

本 sprint 覆盖：

- `torchrun` 启动的 single-node DDP。
- `torchrun` 启动的 single-node FSDP。
- 为 multi-node DDP/FSDP 保留 schema 和环境变量路径。
- rank0 负责 rollout collection、reward、eval、metrics、checkpoint。
- all ranks 负责 replay loss、backward、optimizer step。
- rank0 从 wrapped model 导出 rollout-compatible trainable state，并同步给 Ray rollout workers。

本 sprint 不覆盖：

- Megatron Tensor Parallel / Pipeline Parallel 真实实现。
- RayTrainGroup 主训练入口。
- ZeRO / DeepSpeed。
- 多节点 hostname/rack-aware placement。
- 让 rollout worker 也持有 FSDP/Megatron shard。
- 非 LoRA full-model 大 checkpoint 的高性能 shard 存储优化。

## 3. Target Config

新增 base config：

```text
configs/base/distributed/training_single_process.yaml
configs/base/distributed/training_ddp.yaml
configs/base/distributed/training_fsdp.yaml
```

目标 schema：

```yaml
distributed:
  training:
    strategy: single_process   # single_process | ddp | fsdp | megatron
    launcher: none             # none | torchrun | ray
    num_nodes: 1
    gpus_per_node: 1
    backend: nccl
    init_method: env
    find_unused_parameters: false

    fsdp:
      sharding_strategy: FULL_SHARD
      backward_prefetch: BACKWARD_PRE
      cpu_offload: false
      use_orig_params: true
      mixed_precision: actor
      activation_checkpointing: actor
      state_dict_type: full_rank0

    megatron:
      enabled: false
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
```

规则：

- `strategy=single_process` 时行为必须和当前 repo 一致。
- `strategy=ddp/fsdp` 时必须通过 `torchrun` 或等价环境启动。
- `strategy=ddp/fsdp` 且 `WORLD_SIZE` 缺失时 fail-fast。
- `strategy=megatron` 第一阶段必须 fail-fast，错误信息说明未实现。
- `distributed.resources.trainer.num_gpus` 必须等于 `training.num_nodes * training.gpus_per_node`，否则启动时报错。

## 4. Training Strategy Abstraction

新增：

```text
vrl/distributed/training/context.py
vrl/distributed/training/strategy.py
vrl/distributed/training/ddp.py
vrl/distributed/training/fsdp.py
tests/distributed/training/test_context.py
tests/distributed/training/test_strategy.py
```

目标接口：

```python
class TrainingStrategy(Protocol):
    context: DistributedTrainingContext

    def prepare_bundle(self, bundle: RuntimeBundle) -> RuntimeBundle:
        ...

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def clip_grad_norm(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        ...

    def export_trainable_state(self, bundle: RuntimeBundle) -> dict[str, dict[str, Any]]:
        ...

    def load_trainable_state(self, bundle: RuntimeBundle, state: dict[str, Any]) -> None:
        ...

    def barrier(self) -> None:
        ...
```

`DistributedTrainingContext`：

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

原则：

- train script 不直接读 `RANK` / `LOCAL_RANK`。
- trainer 不直接知道 DDP/FSDP 细节。
- checkpoint 和 rollout sync 通过 strategy 导出 unwrapped/full trainable state。

## 5. Model Wrapping Contract

不要 wrap 整个 policy。优先 wrap `RuntimeBundle.trainable_modules` 里的 trainable root。

Diffusion families 当前已经适合：

```python
trainable_modules={"transformer": transformer}
```

需要新增统一 helper：

```text
vrl/models/trainable.py
```

目标能力：

- `iter_trainable_modules(bundle)`
- `replace_trainable_module(bundle, name, wrapped_module)`
- `unwrap_trainable_module(module)`
- `export_unwrapped_state_dict(module)`

Diffusion policy 需要显式支持替换：

- SD3: `policy._set_transformer(wrapped)`
- Wan diffusers: `policy._set_transformer(wrapped)`
- Cosmos: `policy._set_transformer(wrapped)`

AR family 需要先收窄 trainable contract：

- Janus-Pro 当前是 `trainable_modules={"model": model}`，后续还可以改成更明确的 language model LoRA root。
- NextStep-1 当前是 `trainable_modules={"model": model}`，后续还可以改成 language model / image head 的明确 roots。

第一阶段只要求 SD3 / Wan diffusers / Cosmos。Janus-Pro / NextStep-1 可以 fail-fast：

```text
distributed.training.strategy=fsdp is not supported for family=janus_pro until trainable_modules exposes explicit trainable roots.
```

## 6. Online GRPO Loop Changes

当前 `OnlineTrainer.step()` 既 collect 又 train。multi-GPU 下要拆成两层：

```text
rank0:
  collect rollout batches
  compute rewards
  compute global advantages
  broadcast training batch payload

all ranks:
  shard training payload
  replay forward/backward
  all-reduce/reduce-scatter through DDP/FSDP
  optimizer step

rank0:
  export trainable state
  push rollout weights
  log metrics/checkpoint/eval
```

新增方法：

```python
async def collect_training_batch(...)
def train_on_rollout_batch(...)
```

`OnlineTrainer.step()` 在 single-process 下继续调用两者，保持旧行为。

DDP/FSDP 训练 shard 规则：

- GRPO group 必须完整保留，不能把同一个 prompt 的 `n` 个 samples 分散后再各自算 advantage。
- advantage 在 rank0 统一算好后再分片。
- 分片单位是 rollout sample 或 microbatch，但不能破坏 prompt group。
- 如果 batch 不能被 world size 整除，先支持 padding + mask，不 silent drop。

## 7. Offline DPO Loop Changes

Wan DPO 是 offline trainer，应该单独接 distributed dataloader：

- `DistributedSampler`
- rank-local batch
- DDP/FSDP backward/step
- rank0-only metrics/checkpoint

这条路径比 online GRPO 简单，可以作为 DDP/FSDP 的第二个验证目标。

## 8. Checkpoint and Resume

修改：

```text
vrl/trainers/checkpointing.py
vrl/trainers/online.py
vrl/trainers/offline_dpo.py
```

要求：

- single-process checkpoint 格式保持兼容。
- DDP checkpoint 只在 rank0 保存 unwrapped module state。
- FSDP checkpoint 第一版使用 rank0 full state dict，不做 shard checkpoint。
- resume 时所有 ranks 必须加载同一份 trainable state。
- optimizer state 第一版可以要求 full optimizer state rank0 保存 / broadcast；如果实现复杂，FSDP optimizer resume 可以先 fail-fast，但必须明确写入 DoD。
- EMA 第一版默认只支持 DDP；FSDP + EMA 先 fail-fast，除非实现完整参数 gather/update。

rollout sync 要改成：

```python
state = strategy.export_trainable_state(bundle)
await weight_syncer.push(flatten_for_policy_load(state))
```

不能继续直接使用：

```python
self.model.state_dict()
```

## 9. Rollout Weight Sync Contract

当前 Ray rollout worker 调用：

```python
policy.load_trainable_state(state_ref)
```

这个 contract 要保留，但 trainer 侧必须保证传过去的是 rollout policy 能加载的普通 unwrapped state。

Diffusion families 当前要求 `transformer.*` prefix：

```python
load_trainable_state only accepts trainable keys prefixed with "transformer."
```

因此 strategy 导出后要统一成 policy-facing key space，而不是 DDP/FSDP wrapper key space。

新增测试：

- DDP-wrapped transformer 导出的 key 不包含 `module.` 泄漏。
- FSDP full state 导出的 key 能被 fresh rollout policy `load_trainable_state()` 加载。
- LoRA-only sync 仍然只推 trainable adapter state，不推 frozen checkpoint。

## 10. Megatron Boundary

`strategy=megatron` 第一阶段只做配置校验和 fail-fast：

```text
distributed.training.strategy=megatron is reserved for a future Megatron backend and is not implemented in this sprint.
```

后续 Megatron sprint 需要独立解决：

- model family 是否有 Megatron-compatible module。
- tensor/pipeline parallel size。
- Megatron optimizer and scheduler。
- Megatron checkpoint import/export。
- rollout worker 如何接收 TP/PP shard 或 rank0 gathered LoRA state。
- diffusion transformer 是否值得 Megatron 化，还是只支持 AR LLM-like trunk。

## 11. Implementation Phases

### Phase 1: Config and Context

新增 distributed training config 和 context resolver。

完成条件：

- `single_process` 默认行为不变。
- `ddp/fsdp` 能从 `torchrun` 环境解析 rank/local_rank/world_size/device。
- `megatron` fail-fast。
- 配置校验覆盖 resources/training GPU 数一致性。

### Phase 2: Strategy Abstraction

新增 `SingleProcessStrategy`、`DDPStrategy`、`FSDPStrategy` skeleton。

完成条件：

- trainer 通过 strategy backward/clip/export/load。
- single-process tests 全部不变。
- DDP unit test 能证明 wrapper key export 正确。

### Phase 3: Diffusion Family Wrapping

先支持 SD3 / Wan diffusers / Cosmos。

完成条件：

- `prepare_bundle()` 可以 wrap transformer 并写回 policy/backend。
- `policy.forward_step()` 仍然走 wrapped transformer。
- `load_trainable_state()` 能加载 strategy 导出的 state。

### Phase 4: Online GRPO Rank Split

拆 `OnlineTrainer.step()`。

完成条件：

- rank0 collect/reward/advantage。
- all ranks train。
- rank0 log/checkpoint/sync。
- single-process 行为和当前 metrics header 兼容。

### Phase 5: Checkpoint and Rollout Sync

接入 strategy-aware checkpoint。

完成条件：

- DDP resume 正常。
- FSDP rank0 full checkpoint 正常。
- rollout workers 收到的 state 是 unwrapped policy-facing state。
- FSDP + EMA 如未完整实现必须 fail-fast。

### Phase 6: Real Runs

至少完成：

```bash
torchrun --nproc-per-node=2 -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.training.strategy=ddp \
  distributed.resources.trainer.num_gpus=2
```

以及：

```bash
torchrun --nproc-per-node=2 -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.training.strategy=fsdp \
  distributed.resources.trainer.num_gpus=2
```

真实 checkpoint DoD：

- DDP 2-GPU SD3 OCR 能跑至少 2 epoch。
- FSDP 2-GPU SD3 OCR 能跑至少 2 epoch，或明确 fail-fast 在不支持的配置上。
- rank0 metrics 不重复写。
- checkpoint 可以 resume。
- rollout worker policy version 正常递增。
- fixed eval 使用 rank0 model state，输出不重复。

## 12. Acceptance Criteria

代码层：

- 所有 train scripts 不再直接决定 distributed rank/device。
- `OnlineTrainer` 不再直接把 `self.model.state_dict()` 推给 rollout workers。
- checkpoint export/load 走 strategy。
- DDP/FSDP 不污染 rollout-facing state dict keys。
- unsupported family + unsupported strategy 有明确错误。

测试层：

```bash
python -m pytest -q tests/distributed/training
python -m pytest -q tests/trainers/test_online.py
python -m pytest -q tests/trainers/test_checkpointing.py
python -m pytest -q tests/distributed/ray
python -m pytest -q tests/config/test_load_all_experiments.py
```

手动真实运行：

- single-process SD3 OCR 仍然能跑。
- 2-GPU DDP SD3 OCR 能跑。
- 2-GPU FSDP SD3 OCR 能跑或在明确未支持项 fail-fast。

## 13. References

当前代码切点：

- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/sd3_5/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/wan_2_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/cosmos/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/janus_pro/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/nextstep_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/wan_2_1/train_dpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/offline_dpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/checkpointing.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/weight_sync.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/fsdp.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/interfaces/runtime.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/model_base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/sd3_5/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/wan_2_1/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/cosmos/predict2/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/janus_pro/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/nextstep_1/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/train/group.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/worker.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/weight_sync.py`

相关设计：

- `/home/mingfeiguo/Desktop/wm-infra/SPRINT_distributed_resource_config.md`
- `/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/placement_group.py`
- `/home/mingfeiguo/Desktop/miles/miles/utils/arguments.py`
