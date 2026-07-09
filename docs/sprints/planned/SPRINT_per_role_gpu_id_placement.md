# SPRINT: 按角色显式指定 GPU id 的放置（per-role GPU-id placement）

状态：resolver T1–T4 已实现 + 测试（feat/reward-exec-cost-hint）；多卡 colocated **执行**运行时为后续 sprint（本轮 resolve-only）

## 实现状态（本轮落地）

| 任务 | 内容 | 状态 | 关键落点 |
|---|---|---|---|
| **T1** | per-pair overlap：全局 `allow_overlap` bool → `OverlapPolicy(trainer_rollout, trainer_reward)`，bool 是退化广播 | ✅ 已实现 | `vrl/ray/resources.py` `OverlapPolicy` + `_parse_allow_overlap` |
| **T2** | per-GPU memory fraction：`colocate_with_trainer.memory_fraction` 接受标量或 `{gpu_id: fraction}`，标量=均匀退化 | ✅ 已实现 | `_parse_memory_fraction` + worker `_worker_memory_fraction` |
| **T3** | fsdp trainer 共卡：放开，但只允许 resident colocate 路径（offload 是 single_process 专用），reward-on-trainer 仍 disjoint | ✅ 已实现 | `_validate_fsdp_trainer_overlap` |
| **T4** | per-GPU lease：`ActorLeasePolicy.by_gpu`，role `mode` 为保守汇总，单卡=N=1 退化 | ✅ 已实现 | `_per_gpu_lease` + lifecycle 构造 |
| 测试 | resolver 纯单测 | ✅ 58 passed | `tests/ray/test_resources.py` |
| 多卡 colocated **执行**（fsdp rank 与 rollout worker 共卡的 offload/常驻协调） | ❌ 未做（resolve-only） | 后续 sprint：需 per-rank 常驻 colocation 运行时 |

测试：`tests/ray tests/generation` 共 **171 passed**（resolver 58 + 消费端未回归）。

---

## 0. 结论

让用户**直接给每个角色（trainer / rollout / reward）写显式 GPU id**、允许任意可重叠集合、从 id 交集推导 colocated/disjoint——本轮把这个模型在 **resolver/config 层做成一等公民**，并把**单卡当成多卡的 N=1 退化特例**：

1. **重叠**不再是一个全局 bool，而是 per-pair 的 `OverlapPolicy`；`allow_overlap: true/false` 退化成"全开/全关"的广播。
2. **显存配额**不再是一个标量，而是 `{gpu_id: fraction}` 映射；单个 `memory_fraction` 退化成"对每张 colocated 卡均匀广播"。
3. **fsdp 多卡共卡**被放开（之前 single_process-only），单卡 colocation 成为它的 N=1 退化；但诚实地只允许 resident memory-fraction 路径（offload 路径是 single_process 专用），且**执行**为 resolve-only。
4. **lease** 从 per-role 二值变成 per-GPU；role 级 `mode` 退化成"任一卡需释放即 on_demand"的保守汇总。

**这是 VRL 独有的配置面**：slime（数量 + `--colocate`）与 cosmos-rl（`mode` + 并行维度）都只能表达全重叠或连续两段不相交，表达不了 `rollout=[0,1]/train=[2]/reward=[0,1]` 这种部分重叠，也都不暴露 id 级放置。

**载重警告（写进代码注释）**：显存够大能开重叠，但一张物理卡 SM 固定——共卡只 interleave 不并行。id 放置的收益要用实测 per-phase wall-time 论证（见 `SPRINT_slime_overlap_strategy.md` §4a：reward 仅 ~1% wall-time），不能用"显存放得下"论证。真正的并行加速来自不相交的卡。

底座不重建：`GlobalRayPlacementOwner` / `RayLifecyclePlan` / `BundleLayout` 仍由 `done/` sprint 拥有，本轮只给它们喂更丰富的 `ResolvedDistributedResources`。

---

## 1. T1 — per-pair OverlapPolicy（已实现）

全局单 bool 表达不了"允许 trainer↔rollout 共卡、但禁止 trainer↔reward"。改成：

```python
@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    trainer_rollout: bool = False
    trainer_reward: bool = False
```

- `allow_overlap: false` → `OverlapPolicy(False, False)`；`allow_overlap: true` → `OverlapPolicy(True, True)`（退化广播，旧行为 byte-compatible）。
- `allow_overlap: {trainer_rollout: true, trainer_reward: false}` → per-pair。
- rollout↔reward 共享仍由 `reward.share_with_rollout` 管（独立机制），不进 OverlapPolicy。
- 解析 `_parse_allow_overlap`（拒未知 pair key）；schema `DistributedSection.resources` 加了 `allow_overlap` 子块声明 `trainer_rollout/trainer_reward`。
- 各检查点喂对 pair 的 flag：rollout 用 `config.overlap.trainer_rollout`，reward 用 `config.overlap.trainer_reward`。

测试：`test_per_pair`（隐含在 fsdp/reward 测试里用 mapping 形式）+ 全部旧 bool 测试不变。

## 2. T2 — per-GPU memory fraction（已实现）

`colocate_with_trainer.memory_fraction` 现在接受标量或 `{gpu_id: fraction}`：

- 标量 `0.45` → 解析为 `float`（均匀退化，运行时路径不变）。
- 映射 `{0: 0.6, 1: 0.4}` → 解析为 `dict[int, float]`，键必须**恰好覆盖** colocated rollout GPU（`resolve_distributed_resources` 校验 `capped == set(rollout_devices)`）。
- 解析在 `_parse_memory_fraction` / `_validated_memory_fraction`（范围 (0,1] 在 parse 校验）。
- 消费端：`ResolvedDistributedResources.rollout_gpu_memory_fraction: float | Mapping[int,float] | None`；`RayGenerationConfig.gpu_memory_fraction` 同型；worker 侧 `vrl/generation/execution/worker.py:_worker_memory_fraction` 按 `current_gpu_ids()[0]` 查自己的物理卡（容忍序列化后 int/str 键），标量/None 直接透传（单卡路径零改动）。

测试：`test_colocate_with_trainer_*`（标量退化不变）+ fsdp colocation 用 map 形式。

## 3. T3 — fsdp trainer 共卡放开（已实现，诚实 gate）

`_validate_fsdp_trainer_disjoint`（无条件禁 fsdp 重叠）→ `_validate_fsdp_trainer_overlap`：

- 未声明的 trainer 重叠已被 OverlapPolicy（§1）在上游拒掉，到这里的重叠都是有意的。
- **trainer↔rollout** 重叠：仅允许走 resident `colocate_with_trainer`（memory-fraction 常驻共卡）；offload 路径是 single_process 专用（`lifecycle.py` 只 offload 一个 driver model，协调不了 N 个 fsdp rank）。无 colocate 的 fsdp 重叠 → 报错指向 colocate_with_trainer。
- **trainer↔reward** 重叠：fsdp 下无 resident reward-on-trainer 机制 → 仍 disjoint，报错。
- single_process trainer 仍限 0/1 卡（这是真实约束，不是单卡特判——多卡训练的路径**是** fsdp）。
- **诚实**：多卡 fsdp colocation 在 resolver 解析通过，但其 colocated **执行**（resident rollout worker 与 fsdp rank 共卡）尚未在硬件验证，是 resolve-only。代码注释 + 本文件已标注。

测试：`test_fsdp_trainer_rollout_overlap_requires_colocate_with_trainer`、`test_fsdp_trainer_rollout_colocation_resolves_with_memory_fraction`、`test_fsdp_trainer_reward_overlap_still_rejected`。

## 4. T4 — per-GPU lease（已实现）

```python
@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    mode: Literal["resident", "on_demand"]
    by_gpu: tuple[tuple[int, str], ...] = ()
```

- `mode` = 保守的整角色汇总（任一卡需释放即 on_demand），**当前运行时仍消费它**（`launcher.py` 读 `lifecycle.rollout.mode`，未改）。
- `by_gpu` = per-GPU 真相：rollout GPU 当且仅当与 trainer 共享（且 release_before_train）或与 reward 共享（且 release_before_reward）时为 on_demand。
- **信息增益**：rollout 跨"trainer 共享卡 + 空闲卡"时，`by_gpu = ((0,'on_demand'),(1,'resident'))`——role 级 `mode` 丢掉了"空闲卡可常驻"这条信息，per-GPU 保住了。
- 单卡 = 一个 entry，mode == 该 entry（N=1 退化）。derivation 在 `_per_gpu_lease`（2 调用者：rollout/reward）。
- 与 `mode` 一致性：所有 config-reachable 拓扑下 `mode==on_demand ⟺ by_gpu 含 on_demand`。

测试：`test_per_gpu_lease_keeps_dedicated_gpu_resident_while_shared_releases`、`test_per_gpu_lease_single_gpu_is_degenerate_of_role_mode`、`test_per_gpu_lease_shared_reward_pool_is_on_demand`。

---

## 5. Architecture Hygiene

**改了（都是去掉真实的单卡特判，不是投机泛化）**
- 全局 `allow_overlap` bool → per-pair（单 bool 已被迫 thread 进 6 处检查）。
- 单标量 fraction → per-GPU map（单卡形状是真实债务）。
- per-role 二值 lease → per-GPU（role mode 丢失异构信息是真实信息损失，§4 的 shared+spare 案例可证）。

**没改（保留）**
- `BundleLayout` coalescing：已能表达 per-GPU 重叠，不动。
- `colocated = set(trainer)&set(rollout)` 派生式：不回退成显式 flag。
- `ActorLeasePolicy.mode` 命名与运行时消费：保留，只**加** `by_gpu`，不换词、不改运行时。
- 无新薄文件：所有逻辑落在 `vrl/ray/resources.py`。

## 6. 非目标

- 不重建 `GlobalRayPlacementOwner` / `RayLifecyclePlan` 派生逻辑（`done/` sprint 拥有）。
- **不做多卡 colocated 执行运行时**（fsdp rank 让卡 / per-rank 常驻 colocation / per-GPU lease 的运行时消费）——这是下一个 sprint。本轮 resolver 解析通过即止。
- 不做真并行加速机制（MPS / MIG / 跨 stream）。
- 不复活已删的 release_* 公共 YAML key（`_reject_removed_distributed_keys` 仍硬拒）。

## 7. 验收

**最小（单测）✅ 已过**
- 三例（全不相交 / 全共享 / 部分重叠）resolve 成功且 `colocated` / `by_gpu` 与真值表一致。
- per-pair：允许 trainer_rollout 同时拒 trainer_reward；未知 pair key 报错。
- per-GPU fraction：map 键不覆盖 colocated rollout GPU → 报错；标量退化不变。
- fsdp colocation 走 colocate_with_trainer 解析通过；无 colocate 的 fsdp 重叠报错；reward-on-trainer fsdp 报错。
- 消费端无回归：`tests/ray tests/generation` 171 passed。

**真机（后续，需多卡）**
- `rollout=[0,1]/train=[2]/reward=[0,1]`：GPU2 train 与 GPU0/1 rollout 真并行；reward 蹭 0/1 不 OOM；附 per-phase wall-time。
- 大显存 `train=[0]/rollout=[0]` 常驻不 offload：确认 offload 未触发、weight-sync 走片内拷贝。
- 多卡 fsdp colocation 端到端（需先建 per-rank 常驻 colocation 运行时）。

## 8. 参考

**本仓库代码（本轮落点）**
- `vrl/ray/resources.py` — `OverlapPolicy`/`_parse_allow_overlap`(T1)、`_parse_memory_fraction`(T2)、`_validate_fsdp_trainer_overlap`(T3)、`ActorLeasePolicy.by_gpu`/`_per_gpu_lease`(T4)
- `vrl/config/schema.py` — `DistributedSection.resources` 的 `allow_overlap` 子块(T1)
- `vrl/generation/ray/config.py` — `RayGenerationConfig.gpu_memory_fraction` 同型 + 校验(T2)
- `vrl/generation/execution/worker.py` — `_worker_memory_fraction` per-GPU 解析(T2)
- `tests/ray/test_resources.py` — T1–T4 测试

**相邻 sprint**
- `planned/SPRINT_colocation_config_simplification.md` — 二值前身：`colocate_with_trainer` 是本 sprint 推广的退化特例
- `planned/SPRINT_slime_overlap_strategy.md` §4a — 吞吐论证（reward 忙 ~1%）
- `done/SPRINT_global_ray_placement_owner.md` §2.3/2.4 — 运行时底座（共卡已可表达；勿重建）
- `done/SPRINT_ray_phase_lifecycle_plan.md` §1 — overlap-derived-from-id-sets 真值表

**外部（证明 id 放置是 VRL 特有）**
- slime `slime/ray/placement_group.py`、`docs/en/get_started/usage.md` — 数量 + `--colocate`
- cosmos-rl `cosmos_rl/launcher/launch_all.py`、`policy/config/__init__.py`（`mode` 默认 `disaggregated`）— mode + 并行维度
