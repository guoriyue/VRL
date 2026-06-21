# SPRINT: rollout colocation 输入面收口到 gpu_pool 单一源 (done)

状态：done（2026-06-20）。P0–P3 全部落地：删 `RolloutResourceConfig.memory_fraction` 死镜像、
`DistributedResourceConfig.rollout_persistent_colocated_workers` 改 `resolve_distributed_resources`
顶部就地派生、删遗留 `colocate` 块语法（`_parse_colocate_block` + schema 字段 + 唯一在用 config 迁移到
`gpu_pool: trainer`）、所有指回 `rollout.colocate` 的报错（含 `vrl/generation/ray/config.py` 三处）改指
`distributed.resources.rollout.gpu_pool=trainer + memory_fraction`，旧 `colocate:` 块改 hard-reject。
验证：tests/ray+config+generation/ray 339 passed、ruff 全过、config lint 全过。
范围：把 `vrl/ray/resources.py` 里 rollout「共卡 + 显存帽」这一个意图的**输入侧三套存储/三套语法**——遗留 `distributed.rollout.colocate` 块、`RolloutResourceConfig.memory_fraction` 死镜像字段、`DistributedResourceConfig.rollout_persistent_colocated_workers` 派生缓存布尔——收成 `distributed.resources.rollout.{gpu_pool, memory_fraction}` 这一个权威语法 + 内部就地派生。**不碰** runtime lease 词汇、`allow_driver_gpu_overlap`、`RayLifecyclePlan` 派生规则（那几项已被 [[SPRINT_ray_phase_lifecycle_plan]] 与 [[SPRINT_colocation_config_simplification]] 刻意定案）。

## 0. Core Decision（先看这一段）

[[SPRINT_gpu_pool_grammar_unification]] 已经把 `distributed.resources.rollout.gpu_pool: trainer` + 同级 `memory_fraction` 立成 rollout 共卡的**新权威语法**（镜像 `reward.gpu_pool`）。但旧的 `distributed.rollout.colocate: {memory_fraction}` 块没删，于是现在**同一个意图有两套并存的公有语法**，而且内部还多存了两份冗余状态：

1. **遗留 `colocate` 块**仍被解析（`_parse_colocate_block`，`vrl/ray/resources.py:1207`），代码自己注释它是 "legacy / alternate surface"（`:1161`）。更糟的是 removed-key 报错（`:1290/1302/1307`）反过来把迁移者**指回** `rollout.colocate`，与 gpu_pool 文档说它 legacy 互相打架。全仓只剩一个 config 真在用它。
2. **`RolloutResourceConfig.memory_fraction`（`:40`）是死字段**：在 `:490` 被写入，但 resolver 全程只读扁平的 `config.rollout_gpu_memory_fraction`（`:293`），nested 这份**零读者**（vrl/ 与 tests/ 都没有）。
3. **`rollout_persistent_colocated_workers`（`:72`）是缓存派生布尔**：在 `:511-512` 被算成恰好 `gpu_pool == "trainer" and memory_fraction is not None`，再在 `:283/288/299/315` 反复对照它的两个输入重新判断。它读起来像独立开关，实则不能独立设置。

收口原则：**rollout 共卡只有一个权威输入面 `gpu_pool: trainer` + `memory_fraction`；其余两个内部字段从这两个值就地派生，不缓存、不镜像。** 这条线和 reward 侧 `gpu_pool` 已经划在同一处。

## 1. 现状实锤

### 1.1 遗留 `colocate` 块与新 gpu_pool 语法并存

`_parse_rollout_pool` 同时接受两套语法，并在两者都给时报错（`vrl/ray/resources.py:1166-1175`）：

```python
colocate_fraction = None if colocate is _MISSING else _parse_colocate_block(colocate)
new_pool = cfg_get(rollout_node, "gpu_pool", _MISSING)
new_fraction = cfg_get(rollout_node, "memory_fraction", _MISSING)
if colocate_fraction is not None and (new_pool is not _MISSING or new_fraction is not _MISSING):
    raise ValueError("set either the distributed.rollout.colocate block or the new "
                     "distributed.resources.rollout.{gpu_pool, memory_fraction}, not both")
if colocate_fraction is not None:
    return ("trainer", colocate_fraction)
```

docstring 自己点名它是遗留替代面（`:1161-1162`）：

```text
The ``distributed.rollout.colocate`` block is an alternate surface that maps to
``gpu_pool=trainer`` + its (required) memory_fraction.
```

但 removed-key 守卫的报错（`:1304-1307`）把迁移者**指回** colocate：

```python
"distributed.rollout.gpu_memory_fraction moved into "
"distributed.rollout.colocate: {memory_fraction: <0..1>}.",
```

公有 schema 仍把 `colocate` 注册为已知 key（`vrl/config/schema.py:440`）：

```python
colocate: Annotated[Any, ConfigBlock(("memory_fraction",))] = None
```

全仓真正在用遗留块的只有一个 config：`configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:41-42`：

```yaml
    colocate:
      memory_fraction: 0.55
```

base preset（`configs/base/distributed/ray_rollout_colocated_single_gpu.yaml:16-17`）与 cosmos DDP config（`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml:51`）都已用新的 `gpu_pool: trainer` 语法。

### 1.2 `RolloutResourceConfig.memory_fraction` 是死镜像

字段定义（`vrl/ray/resources.py:40`）：

```python
    memory_fraction: float | None = None
```

`from_cfg` 把同一个值同时写进 nested 与扁平两处（`:489-514`）：

```python
        gpu_pool=rollout_gpu_pool,
        memory_fraction=rollout_memory_fraction,    # :490 nested
        ...
        rollout_gpu_memory_fraction=rollout_memory_fraction,  # :514 flat
```

而 resolver 只读扁平那份（`:293`）：

```python
    gpu_memory_fraction = config.rollout_gpu_memory_fraction
```

grep `rollout.memory_fraction` 在 vrl/ 与 tests/ 对这个 nested 字段**零读者**（唯一同名命中是另一个结构体 `PipelineStageRuntimePolicy.memory_fraction`，`vrl/generation/pipeline/topology.py:39`，与本字段无关）。同级 `gpu_pool` 字段则有真实读者（`:208/227`），所以只有 `memory_fraction` 可删。

### 1.3 `rollout_persistent_colocated_workers` 是缓存派生布尔

`from_cfg` 把它算成两个兄弟字段的纯函数（`vrl/ray/resources.py:511-513`）：

```python
        rollout_persistent_colocated_workers=(
            rollout_gpu_pool == "trainer" and rollout_memory_fraction is not None
        ),
```

随后 resolver 在 4 个分支里读它，每处都在重新对照它本可即时算出的输入（`:283/288/299/315`）：

```python
    if config.rollout_persistent_colocated_workers and not colocated: ...
    if config.rollout_persistent_colocated_workers and reward_shared_with_rollout: ...
    if config.rollout_persistent_colocated_workers and gpu_memory_fraction is None: ...
    rollout_release_before_train = (
        colocated and not config.rollout_persistent_colocated_workers
    )
```

它挂在 `DistributedResourceConfig`（`:72`）上像个独立 user 开关，实则不能独立设置——公有侧的旧 `persistent_colocated_workers` key 早已 hard-reject（`:1299-1303`），它如今只是 from_cfg 内部缓存。

## 落地方案

### P0. 行为锁定测试（先锁派生结果）

在 `tests/ray/test_resources.py` 锁住「删字段/删语法后派生结果不变」：

```text
gpu_pool=trainer + memory_fraction=0.45  -> resident，rollout.mode=resident，gpu_memory_fraction 生效（= 今天 colocate 块派生）
gpu_pool=trainer 无 memory_fraction       -> on_demand，无 cap
旧 colocate 块出现                          -> ValueError 指向 gpu_pool: trainer（新增 rejection）
```

并迁移 `tests/config/test_load_all_experiments.py:312`（现断言 `cfg.distributed.rollout.colocate.memory_fraction == 0.55`）到断言派生出的 resolved/lifecycle 字段。

### P1. 删 nested `memory_fraction` 死字段

`vrl/ray/resources.py`：删字段定义（`:40`）与 `from_cfg` 的赋值（`:490`），保留同级 `gpu_pool`。扁平 `rollout_gpu_memory_fraction`（`:77`）是唯一被读的存储，保留不动。

### P2. 把 `rollout_persistent_colocated_workers` 改成就地派生

删 `DistributedResourceConfig` 上的字段（`:72`）与 `from_cfg` 的赋值（`:511-513`）。在 `resolve_distributed_resources` 顶部从单一源算一次本地变量：

```python
persistent = config.rollout.gpu_pool == "trainer" and config.rollout_gpu_memory_fraction is not None
```

把 `:283/288/299/315` 四处 `config.rollout_persistent_colocated_workers` 改读这个本地量。public 侧的 removed-key 守卫（`:1299-1303`）不动（它拒的是旧 YAML key，不是这个内部字段）。

### P3. 删遗留 `colocate` 块语法，统一指向 gpu_pool

迁移唯一在用方 `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:41-42`：

```yaml
    gpu_pool: trainer
    memory_fraction: 0.55
```

随后 `vrl/ray/resources.py`：删 `_parse_colocate_block`（`:1207-1244`）、`_parse_rollout_pool` 的 `colocate` 形参与映射分支（`:1154/1166-1175`）；`vrl/config/schema.py:440` 删 `colocate` 字段；把所有把迁移者**指回** `rollout.colocate` 的报错（`:217/692/1290/1302/1307`、`schema.py:425-427/468`）改指 `distributed.resources.rollout.gpu_pool=trainer` + `memory_fraction`。见到旧 `colocate:` 块改为 hard-reject 指向新语法（与 `:1286-1308` 同模式）。base preset 注释（`ray_rollout_colocated_single_gpu.yaml:25-26`）删掉「legacy colocate 仍可用」一段。

迁移 `tests/config/test_load_all_experiments.py:312` 与任何断言 `colocate` 块解析的用例。

## 验证（finishing criteria）

```text
1. grep colocate 在 vrl/ 与 configs/ 下不再出现 distributed.rollout.colocate 作为活语法（仅注释/报错文案里作为「已删」提示）。
2. RolloutResourceConfig 不再有 memory_fraction 字段；DistributedResourceConfig 不再有 rollout_persistent_colocated_workers 字段。
3. gpu_pool: trainer + memory_fraction 解析出的 ResolvedDistributedResources / RayLifecyclePlan 与今天 colocate 块完全一致（P0 锁测证明）。
4. 旧 colocate 块出现时 hard-reject，报错指向 gpu_pool: trainer（不再静默接受、也不再指回 colocate）。
5. tests/ray tests/config tests/generation/ray tests/rewards 全绿（环境缺失 skip 不计）；ruff 全过。
6. python -m vrl.config.lint（或现有 config-resolve 入口）对全部 in-repo configs 通过。
```

## 非目标 / Non-Goals

```text
不碰 runtime lease 词汇（with_release_after_collect / release_after_call / _RuntimeLease）——
  [[SPRINT_ray_phase_lifecycle_plan]] 已论证盲改破坏 API、release_after_score 是 per-reward runtime kwarg。
不改 allow_driver_gpu_overlap / is_colocated() / RolloutLifecycle.runtime_is_colocated 命名——
  [[SPRINT_colocation_config_simplification]] 已定它继续作内部 resolved 结果。
不改 RayLifecyclePlan / ActorLeasePolicy / PhaseHandoffPolicy 的派生规则。
不动 distributed.resources.allow_overlap 资源安全闸。
不动 reward.gpu_pool / reward.kwargs.<name>.release_after_score。
不引入 sleeping / offload。
```

## References

仓库代码（本 sprint 实际读取并核对）：

- `vrl/ray/resources.py:27-40`（`RolloutResourceConfig` 字段，含死 `memory_fraction`）、`:68-77`（`DistributedResourceConfig` 含派生缓存 `rollout_persistent_colocated_workers`）、`:200-208/226-234`（resolver 读 `gpu_pool`）、`:283-321`（`persistent_colocated_workers` 四处读 + release 派生）、`:475-525`（`from_cfg` 双写 memory_fraction + 缓存 persistent）、`:1151-1204`（`_parse_rollout_pool` 双语法）、`:1207-1244`（`_parse_colocate_block`）、`:1278-1308`（removed-key 守卫指回 colocate）
- `vrl/config/schema.py:421-440`（`RolloutWorkerSection`，`colocate` 仍注册为公有 key）
- `vrl/generation/pipeline/topology.py:39`（同名但无关的 `PipelineStageRuntimePolicy.memory_fraction`，排除误判）
- `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:41-42`（全仓唯一在用遗留 colocate 块）
- `configs/base/distributed/ray_rollout_colocated_single_gpu.yaml:16-26`、`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml:51`（已用新 gpu_pool 语法）
- `tests/config/test_load_all_experiments.py:312`、`tests/ray/test_resources.py:284-285/1261`（待迁移断言）

相关 sprint：

- [[SPRINT_gpu_pool_grammar_unification]]（立 gpu_pool 新权威语法；本 sprint 删掉它遗留下的并存 colocate 旧面）
- [[SPRINT_colocation_config_simplification]]（前序收口 public colocation surface；定 allow_driver_gpu_overlap / 兼容视图字段不动）
- [[SPRINT_ray_phase_lifecycle_plan]]（定 lifecycle / lease 词汇与 RayLifecyclePlan 派生规则不动）
- [[SPRINT_resolved_struct_field_audit]] / [[SPRINT_design_smell_audit]]（前序 dead/derived 字段审计，审的是 Resolved* 结构；本次清的是输入 config 结构上后引入的死/派生字段）
