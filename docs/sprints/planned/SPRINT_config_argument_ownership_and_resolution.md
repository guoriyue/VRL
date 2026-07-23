# SPRINT：Config argument ownership and resolution

状态：**planned（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：[Contract truthfulness and no-op inputs](SPRINT_contract_truthfulness_and_noop_inputs.md)

## 0. 结论先行

OmegaConf、Pydantic、resolver/builder 和 runtime config 四层都保留。要删除的是层间重复决策：

```python
def validate_training_config(cfg):
    warn_unknown_keys(cfg)
    parse_config(cfg)                 # result discarded
    resolve_precision_policy(cfg)

validate_training_config(cfg)
precision = resolve_precision_policy(cfg)  # parsed again
```

目标是：public config parse一次；每个决策 resolve一次；builder返回具名 typed结果；runtime
dataclass只含对应 consumer真正读取的值。

本 Sprint 不以“一个 YAML section对应一个 dataclass”为目标。一个 public `trainer:` section可以
投影到 controller、trainer、checkpoint多个明确 runtime owner；反过来，也不能为了保持单一
`TrainerConfig`，把 generation和model-setup字段塞进去。

## 1. T0 — Typed validation/build result

### 当前问题

- `parse_config(cfg)` 的 `RootConfig` 被丢弃；
- precision在 validation和build各 resolve一次；
- reward validation/build再次解析；
- `build_configs()` 返回 `dict[str, Any]`；
- reward返回 `(weights, kwargs)`，caller使用 `[0]` / `[1]`。

### 目标

1. public parse返回并保留 typed Root结果；
2. precision、reward、distributed/resource decisions各 resolve一次；
3. `build_configs()` 返回 `BuiltConfigs`；
4. reward返回 `RewardRuntimeConfig(weights, kwargs)`；
5. factory、online recipe、DPO需要的 typed slice从同一 build结果获得；
6. `build_configs`、`build_trainer_config` 等 facade **KEEP**，改变返回形状而不是删除边界。

`BuiltConfigs` 是一次 build的具名输出，不是新的 config层。它应放在 `vrl/config/builders.py`
附近；不要新建只有 re-export 的 `manager`/`container` 模块。

### 验收

- typed字段访问替代所有 `built["..."]` 和 reward tuple下标；
- reward有/无、precision role split true/false；
- invalid reward在模型构造前失败；
- spy证明每个 resolver每次 build只调用一次。

## 2. T1 — 按 consumer 拆 `TrainerConfig`

### 当前问题

`TrainerConfig` 同时承担 OnlineTrainer、recipe controller、model setup、checkpoint和generation
bridge：

- `total_epochs/save_freq/seed` 只有 `run_online_recipe` 读；
- `resume_from` 只有 checkpoint loader读 raw cfg，dataclass副本零 reader；
- `gradient_checkpointing` 由 model setup helper读 raw cfg，OnlineTrainer不读；
- `samples_per_chunk` 由 generation读 rollout/request，trainer副本只校验/日志；
- prompts/group/GAS/microbatch同时被 controller和trainer读取；
- `microbatch_size` 与 GAS互相赋值后同时存储。

### 目标形态

#### `TrainerConfig`

- 只保存 `OnlineTrainer`、optimizer和trainer-owned validation实际读取的值；
- 从 `vrl/trainers/core/types.py` 移到 `vrl/trainers/online/config.py`；
- `vrl.trainers.TrainerConfig` public re-export保持，避免无关 import churn；
- 不保留 `resume_from`、`gradient_checkpointing`、`samples_per_chunk`。

这个新文件不是为了少几行而拆；它让 online-only type回到唯一 consumer package，属于真实
ownership boundary。

#### `OnlineBatchPlan`

在同一个 `vrl/trainers/online/config.py` 定义 frozen plan，构造一次并由 controller/trainer共享：

```text
prompts_per_batch
n_samples_per_prompt
gradient_accumulation_steps       # canonical stored value
replay_samples_per_chunk
host_memory_budget_fraction
microbatch_size                   # computed property
```

- public可以接受 size或count输入；
- runtime只存 canonical GAS；
- size-only、count-only都能构造同一 plan；
- 冲突和不整除 fail fast。

`host_memory_budget_fraction`、`microbatch_size` 从 `rollout` YAML owner移到 `actor`，因为它们控制
streaming replay/backward，不控制 generation。迁移所有 presets，不保留两个永久 alias。

#### Controller 与 checkpoint

- `total_epochs/save_freq/seed` 进入与 `OnlineRecipeRun` 同址的 `OnlineRunConfig`；
- `resume_from/resume_strict` 由 checkpoint module的 `TrainingResumeConfig` 一次解析；
- `output_dir/profile` 同时有 trainer与controller行为，可继续作为共享 resolved input，不能为了
  “每字段只被一个类读”强行复制。

### 明确保留

- public `rollout.samples_per_chunk` 和所有 generation consumer；
- public `actor.gradient_checkpointing` 与共享 resolver/helper；
- `replay_samples_per_chunk` 是训练 backward容量，和 generation chunk不是同一值；
- `ContinuousRolloutConfig → ContinuousRolloutSettings` 是 config到 frozen runtime snapshot。

## 3. T2 — Typed collector-local config 与正向 request projection

### 当前问题

`RolloutCollectorConfig` 把整个 `rollout` / `sampling` flatten，再依赖：

```python
_REQUEST_SAMPLING_EXCLUDES = {
    "host_memory_budget_fraction",
    "kl_reward_coef",
    "microbatch_size",
    "n_samples_per_prompt",
    "prompts_per_batch",
}
```

新增 local key时若漏加 blacklist，会静默泄漏到 generation request。

### 修复

- collector config显式拥有 `kl_reward_coef`、`trajectory_storage` 等 local字段；
- 单独保存 `request_sampling` mapping；
- `sampling.*` 作为明确 wire payload进入 request；
- `rollout.*` 仅从 public schema的 generation-owner metadata正向派生；
- `algorithm.train_segments`、由 KL coefficient派生的 `return_kl` 继续显式投影；
- `_REQUEST_SAMPLING_EXCLUDES` 等正向投影与测试原子落地后删除。

`RolloutCollectorConfig.get()` **KEEP**：`cfg_get()` 通过动态 `getattr(node, "get")` 间接调用，是
framework adapter，不是零 caller薄函数。

`algorithm.kl_reward_coef` **KEEP 原位置**：它是 objective决策、由 collector执行；执行点不等于
public ownership必须改名。

### 测试

- generation-owned field进入 request；
- trainer/collector-local field不进入；
- KL/storage仍能经 `cfg_get()` 读取；
- 新增一个标记为 generation-owned 的 fixture field会自动进入，证明无第二张名单；
- 未标 owner的 rollout scalar默认 fail closed。

## 4. T3 — Data loader 只派生一次

schema和 `load_prompt_examples_from_config()` 当前各自实现 format → loader。共享一个具体
`resolve_data_loader()`：

- `image_caption_jsonl` → `prompt_image_manifest`；
- ordinary text/jsonl → `prompt_manifest`；
- 显式合法 loader保持；
- 显式冲突/非法值失败。

这个薄函数 **KEEP**，因为它消除跨层真实重复。若 T0 后 runtime直接消费 typed Root result，也仍应
保留同一 derivation owner，不能在 Pydantic validator里 mutate一个随后被丢弃的副本。

## 5. T4 — Ray worker 与 distributed strategy defaults

### Ray

相同 worker值/default当前存在于：

1. `RolloutWorkerSection`；
2. `RayGenerationConfig`；
3. `RayGenerationConfig.from_cfg()` fallback literals；
4. 三个 base presets。

目标：

- public `RolloutWorkerSection` 是 defaults单一 source；
- runtime `RolloutWorkerConfig` 是无默认的 frozen snapshot；
- `RayGenerationConfig` 组合 `worker + ResolvedDistributedResources`，不复制字段/default；
- placement owner和launcher接收同一个 resolved worker object；
- 删除 base preset中纯默认重述；
- `cpus_per_worker: 4.0` 等真实 override、named topology显式 pin保留。

`RayGenerationConfig` **KEEP**：它是 launcher/runtime protocol，而不是 schema镜像。

### FSDP/DDP

- public `FSDPConfig/DDPConfig` 继续拥有 user-facing defaults；
- T0 的 typed parse result传给 `build_strategy`，删除 fallback literals；
- `FSDPStrategy` / `DDPStrategy` constructor config参数改 required，不再充当第二默认注册表；
- `apply_fsdp(..., reshard_after_forward=True, cpu_offload=False)` 默认 **KEEP**：它是可独立调用的
  低层 API boundary，不是 public training config副本。

### 测试

- omitted/default与explicit override；
- placement/launcher观察同一个 CPU/worker值；
- pipelined single-worker true、multi-worker false；
- FSDP actor/none precision、reshard true/false、offload true/false；
- DDP unused parameter true/false；
- 构造 API不能制造 schema与runtime矛盾值。

## 6. T5 — Schema key 派生，但不混淆 public 与 runtime

`TrainerConfig` field metadata已经记录 YAML owner。online runtime字段的 known-key registry从该
metadata派生，不再在 `TrainerSection/ActorSection` 手写第二份。

但以下 key不是 TrainerConfig字段，仍需由其真实 public owner显式声明：

- controller-only；
- checkpoint；
- model setup / activation checkpointing；
- offline DPO；
- entrypoint / dispatch。

`_OFFLINE_DPO_ACTOR_FIELDS` / `_OFFLINE_DPO_TRAINER_FIELDS` 当前 **KEEP**：它们是真实 entrypoint
allow-list，阻止 online-only key进入 DPO。等 offline public config有 typed source后，再从该类型
派生，不能提前删除。

测试应模拟增加一个 runtime config字段，known keys自动同步；同时证明 offline配置继续拒绝
online-only字段。

## 7. T6 — Supervisor default 与 health config owner

### Retry/process config

- argparse 的 `max_attempts/same_cause_limit` 默认从 `RunSupervisor` field派生；
- CLI使用 bounded type；
- `RunSupervisor.__post_init__` 验证负 attempt、零 same-cause、负 grace/backoff；
- `_child`、`_stop_requested`、`_health_gate` **KEEP**，它们是真实进程 owner state。

### Health config

supervisor已经 load resolved training config，因此：

- continuous metrics是否 required从 schedule mode派生，不再用
  `health.max_stale_policy_versions is not None` 当 enable sentinel；
- health stale override仍可存在，但只能比训练 schedule上限相同或更严格；
- CLI未显式设置 parity threshold时，从 `trainer.debug.max_abs_logprob_diff` 派生；
- 显式 operational override保留；
- `_REQUIRED_HEALTH_METRICS` / `_CONTINUOUS_HEALTH_METRICS` **KEEP**，是小型 CSV protocol subset。

这不把 supervisor并入 trainer；它仍是独立进程控制边界，只消费 resolved contract。

## 8. What changes / what stays

### 改变

- validation/build只解析一次；
- anonymous dict/tuple改 typed；
- runtime config按 consumer拆分；
- update knobs回到 actor owner；
- generation request使用正向投影；
- public defaults只有一处。

### 保持

- 四层 config architecture；
- builders的 public facade；
- public YAML section组合；
- offline allow-list；
- Ray/runtime config、strategy、checkpoint等真实边界；
- experiment-level明确 pin，不做无差别 YAML清扫。

## 9. Non-goals

- 不把 runtime config换成 Pydantic。
- 不从一个 mega dataclass生成所有 CLI/YAML/runtime对象。
- 不在本 Sprint迁 family model schema；由独立 child Sprint承接。
- 不用反射 FSDPStrategy constructor signature当默认 source。
- 不新增 `_manager` / `_handler` / `_component` 命名层。

## 10. Acceptance gates

- 全部 bundled experiments load + build；
- typed build/result tests；
- unknown-key、DPO、data、supervisor、Ray config、strategy CPU tests；
- placement使用 pure/mocked tests，不启动 Ray；
- config snapshot证明所有保留字段在迁移前后 resolved值相同；
- snapshot是一-shot artifact，验收后删除；
- `ruff` touched files、`git diff --check`。

## 11. Definition of Done

- [ ] Root/precision/reward每次 build只解析一次。
- [ ] `BuiltConfigs` / reward result无字符串或 tuple下标。
- [ ] TrainerConfig无 generation/model-setup/checkpoint死镜像。
- [ ] batch split只有一个 canonical stored value。
- [ ] request projection fail closed。
- [ ] Ray/FSDP/DDP/supervisor defaults无平行副本。
- [ ] public/runtime/offline key集合都从各自 source派生。

## 12. References

- `vrl/config/schema.py`
- `vrl/config/validation.py`
- `vrl/config/builders.py`
- `vrl/config/precision.py`
- `vrl/config/unknown_keys.py`
- `vrl/trainers/core/types.py`
- `vrl/trainers/online/trainer.py`
- `vrl/trainers/checkpointing.py`
- `vrl/trainers/data/prompts.py`
- `vrl/rollouts/collector/config.py`
- `vrl/rollouts/collector/requests.py`
- `vrl/generation/ray/config.py`
- `vrl/generation/ray/launcher.py`
- `vrl/ray/placement.py`
- `vrl/trainers/strategy.py`
- `vrl/trainers/fsdp.py`
- `vrl/scripts/common/online.py`
- `vrl/scripts/common/factory.py`
- `vrl/scripts/supervise.py`
- `docs/sprints/done/SPRINT_config_as_signatures.md`
