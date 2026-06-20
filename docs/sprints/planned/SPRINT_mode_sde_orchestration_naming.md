# SPRINT: 消歧 overloaded 的 `mode` / `sde` 配置 token (planned)

状态：planned（2026-06-20）。范围：把同一个单词在不同正交轴上各表一意的两个 user-facing 配置 token 改成自描述名——rollout 调度的 `mode` → `schedule_mode`，以及 diffusion 采样里 `sde` 既是 `denoise_mode` 的值又是 `sde.type` 的值的同形冲突。**只做命名消歧，不动行为、不动派生字段处置（那部分已有别的 sprint 拍板，见 §0）。**

## 0. Core Decision（先看这一段）

这个主题里**只有两条是新的、且没被前序 sprint 拍板过**，值得做；候选清单里另外三条已被 prune，原因如下，避免重复劳动：

- **`weight_sync_barrier` 派生字段无运行时读者** —— 已由 [[SPRINT_design_smell_loose_ends]] §1 接管（改 `init=False` derive-only），并在 [[SPRINT_design_smell_audit]] P2、[[SPRINT_resolved_struct_field_audit]] §9.1 共担。**不在本 sprint 重报。**
- **`mode` 合法集未从 `RolloutScheduleMode` enum 派生** —— 已在 [[SPRINT_design_smell_audit]] 显式**评估并否决**（audit:299：「派生要让 config 层 import rollout enum 跨同一边界；mode 字符串还驱动 3 个 mode-specific 分支，enum 只能去重 2 元素成员检查」）。已决，不重开。
- **`sde` 字段类型** —— `denoise_mode` / `sde.type` 的 `Literal` 收敛已由 [[SPRINT_config_string_settings]] 落地（typed allow-list + 保留 wire-boundary guard）。那次只做了 **typing**，**没碰同形词消歧**——本 sprint 接力做后者。

剩下两条真正动作：

1. **rollout 调度 `mode` → `schedule_mode`**：和 `lifecycle.rollout.mode`（lease：resident/on_demand）、`denoise_mode`（采样步）三处同名却是三个状态机。lease 那个已被 `ActorLeasePolicy` 封装得很清楚（[[SPRINT_ray_phase_lifecycle_plan]]，类名+docstring 自带消歧），`denoise_mode` 已限定词，**唯一裸的、且与 lease 同挂在 `rollout.*` 下最易混的就是调度 `mode`**。内部 metadata 早已发 `schedule_mode`（已有前例目标名），只需把 config 键和字段对齐到这个名。
2. **diffusion `sde` 同形词**：`denoise_mode: sde`（采样路径）与 `sde.type: sde`（SDE 数学变体，相对 `cps`）同字不同义、相差一层。给 `sde.type` 的 `sde` 变体起个具体名（如 `flow_grpo`），把 `sde` 这个词留给 denoise-mode 轴独用。**注意**：`sde`/`cps` 是 user-facing wire 值、且根植 diffusion-RL 文献，rename 是 config 迁移，需权衡可读性 vs prior-art——见 §非目标对 `cps` 的保留。

两条都是低 severity 的 readability 修，但都是 config-contract 改动；做就一次性把 schema/runtime guard/YAML/脚本全迁完，避免半截。

## 1. 现状实锤

### 1.1 `mode` 一词三义，且两处都挂在 `rollout.*` 下

调度 cadence（`vrl/rollouts/orchestration/types.py:13-17`）：

```python
class RolloutScheduleMode(str, Enum):
    STRICT_ON_POLICY = "strict_on_policy"
    CONTINUOUS = "continuous"
```

config 侧裸字段（`vrl/trainers/core/types.py:170`）：

```python
mode: str = field(default="strict_on_policy")
```

`build_rollout_schedule` 读它（`vrl/rollouts/orchestration/schedule.py:60-62`）：

```python
mode = RolloutScheduleMode(
    getattr(config, "mode", RolloutScheduleMode.STRICT_ON_POLICY.value),
)
```

lease lifecycle（完全不同的状态机，`vrl/ray/resources.py:94`）：

```python
mode: Literal["resident", "on_demand"]
```

被 launcher / config 读（`vrl/generation/ray/launcher.py:205`、`vrl/generation/ray/config.py:164`）：

```python
and resources.lifecycle.rollout.mode == "on_demand"
```

采样步（第三义，`vrl/config/schema.py:248`）：

```python
denoise_mode: Literal["native", "sde"] | None = None
```

**关键实锤**：目标名 `schedule_mode` **已经是内部 metadata 的既有拼写**（`vrl/rollouts/orchestration/types.py:78`）：

```python
"schedule_mode": mode.value,
```

也就是说 metadata 层早就叫 `schedule_mode`，只有 config 键 + dataclass 字段还叫裸 `mode`，对齐即可，不是凭空发明。

### 1.2 `sde` 既是 denoise_mode 的值，又是 sde.type 的值（相差一层）

base 配置里两者同时出现、且注释亲口承认 `sde` 这个值 = Flow-GRPO 随机采样（`configs/base/rollout/diffusion.yaml:7,13`）：

```yaml
  denoise_mode: sde             # 'sde' keeps Flow-GRPO stochastic sampling.
  ...
  sde:
    type: sde
```

schema 两个正交轴各自允许 `sde`（`vrl/config/schema.py:222` 与 `:248`）：

```python
class SdeConfig(ConfigBase):
    type: Literal["sde", "cps"]
...
    denoise_mode: Literal["native", "sde"] | None = None
```

runtime guard 也各查一遍（`vrl/generation/diffusion/layout.py:268-281`）：

```python
def _parse_sde_type(value: Any) -> str:
    sde_type = str(value)
    if sde_type not in {"sde", "cps"}:
        raise ValueError("sampling.sde_type must be 'sde' or 'cps'")
    return sde_type

def _parse_denoise_mode(value: Any) -> str:
    denoise_mode = str(value).strip().lower()
    if denoise_mode not in {"native", "sde"}:
        raise ValueError("sampling.denoise_mode must be 'native' or 'sde'")
    return denoise_mode
```

数学变体的真正分支（`vrl/math/diffusion/flow_matching.py:116`）——只有 `cps` 走特殊公式，`sde` 是「另一支」：

```python
if sde_type == "cps":
```

**正交性实锤**：`denoise_mode == "native"` 仍会进 `sde_step_with_logprob`（executor 在 `vrl/generation/diffusion/executor.py:710` 才分 native 路径），两轴独立。所以 `denoise_mode: sde` + `sde.type: sde` 并存时是两件事，不是冗余/typo——但读 config 的人极易误判。

`sde.type` 值的散布面（rename 必须全覆盖）：`vrl/generation/diffusion/layout.py:60,85,133`、`vrl/generation/diffusion/executor.py:60,311,324`、`vrl/scripts/common/factory.py:191-192`、`vrl/scripts/eval/cosmos_predict25_kling_eval.py:280,476`、`vrl/scripts/diffusion/cosmos/anima/generate.py:268`、`vrl/scripts/perf/*`，以及 ~14 个 YAML（`grep -rln "sde:" configs/`）。

## 落地方案

### A. rollout 调度 `mode` → `schedule_mode`（迁移面小，先做）

1. `vrl/trainers/core/types.py:170`：字段 `mode` → `schedule_mode`（保留 default `"strict_on_policy"`）；`__post_init__`（:178-192）里所有 `self.mode` 引用同步改名。
2. `vrl/rollouts/orchestration/schedule.py:60-62`：`getattr(config, "mode", ...)` → `getattr(config, "schedule_mode", ...)`。
3. YAML 迁移（仅 4 处）：`configs/base/rollout/orchestration/strict.yaml:6`、`configs/base/rollout/orchestration/continuous.yaml:32`、`configs/experiment/diffusion/sd3_5/online_grpo_ocr_crossnode_debug.yaml`、`.../online_grpo_ocr_single_gpu_async_debug.yaml` 里的 `mode:` → `schedule_mode:`（这几个文件 grep 自 `rollout_orchestration`）。
4. 测试 fake：`tests/rollouts/orchestration/test_orchestration.py:13`、`tests/rollouts/orchestration/continuous/test_schedule.py:102` 的 `SimpleNamespace(mode=...)` → `schedule_mode=...`；`tests/config/test_load_all_experiments.py` 相关断言对齐。
5. **不改** `lifecycle.rollout.mode`（`ActorLeasePolicy.mode`，`vrl/ray/resources.py:94`）：类名 `ActorLeasePolicy` + docstring(:88-91) 已自带消歧，且是 [[SPRINT_ray_phase_lifecycle_plan]] 刚稳定的派生 struct，不值得为消歧再翻它。**不改** `denoise_mode`：已限定词。

### B. diffusion `sde.type` 的 `sde` 变体改具体名（迁移面大，与文献权衡）

把 `sde.type` 的 `sde` 变体值改为具体名 `flow_grpo`（与 `cps` 并列，`cps` 保留——它本就是具体变体名），让 `sde` 这个词只留给 `denoise_mode` 轴：

1. schema `vrl/config/schema.py:222`：`type: Literal["sde", "cps"]` → `Literal["flow_grpo", "cps"]`。
2. runtime guard `vrl/generation/diffusion/layout.py:270`：成员集 `{"sde","cps"}` → `{"flow_grpo","cps"}`，错误文案同步。
3. 数学分支 `vrl/math/diffusion/flow_matching.py:116`：`if sde_type == "cps"` 不变（它判的是 `cps`，`sde` 变体走 else，无需改判断；但若有任何 `== "sde"` 的判定需一并改——grep 确认）。
4. 默认值：`layout.py:85`、`executor.py:60,311` 的 `sde_type: str = "sde"` → `= "flow_grpo"`。
5. 脚本默认：`factory.py:192`、`cosmos_predict25_kling_eval.py:280`、`anima/generate.py:268`、`perf/*` 里的字面 `"sde"` → `"flow_grpo"`。
6. YAML（~14 处）：所有 `sde:\n  type: sde` → `type: flow_grpo`（`type: cps` 不动）。
7. 在 `SdeConfig` docstring（`vrl/config/schema.py:217-220`）补一句：`denoise_mode=native` 仍走 `sde_step_with_logprob`，两轴正交——把现状里最反直觉的点写进文档。

> **若团队判断 B 的 config 迁移成本 > 文献一致性收益**：退一步只做文档化（步骤 7）+ 给 `denoise_mode` 的 `sde` 值改名而非 `sde.type`（让 denoise 轴让位、`sde.type` 沿用文献词）。两条路二选一由实施者按 grep 出的实际散布决定，但**必须二选一彻底做完**，不留半个轴改名。

## 验证（finishing criteria）

- `pytest tests/config/test_load_all_experiments.py tests/rollouts/orchestration/ -q` 全绿（覆盖 schedule_mode 改名 + config 加载）。
- `pytest tests/` 全量绿（sde 变体改名波及 layout/executor/flow_matching/factory）。
- config-resolve 冒烟：对每个改过的 experiment YAML 跑一次配置解析（项目既有的 load_all_experiments 即覆盖），确认没有「unknown key `mode`」或「sde_type must be ...」误报。
- `grep -rn '"sde"' vrl/ configs/`（B 路径完成后）只剩 `denoise_mode` 轴的 `sde`，`sde.type` 轴不再出现裸 `"sde"`。
- `grep -rn '\bmode:' configs/base/rollout/orchestration/` 只剩 `schedule_mode:`。

## 非目标 / Non-Goals

- **不动 `weight_sync_barrier` 派生字段**：归 [[SPRINT_design_smell_loose_ends]]。
- **不把 `mode` 合法集改为从 `RolloutScheduleMode` enum 派生**：[[SPRINT_design_smell_audit]] 已否决（跨边界 import 不划算）。
- **不重做 `denoise_mode`/`sde.type` 的 typing**：[[SPRINT_config_string_settings]] 已做 `Literal` 收敛。
- **不改 `ActorLeasePolicy.mode`、不改 `denoise_mode` 名字**：前者已被类名消歧、后者已限定词。
- **不改 `cps` 变体名**：它已是具体变体名（consistency policy step），rename 它纯亏。
- **不动任何 wire-boundary runtime guard 的存在性**：guard 留着（rollout 请求过 Ray 是 plain dict，schema Literal 管不到），本 sprint 只改 guard 里的字符串集合内容，不删 guard。

## References

- `vrl/trainers/core/types.py:170,174,178-192` — `RolloutOrchestrationConfig.mode` 裸字段 + 派生/校验
- `vrl/rollouts/orchestration/types.py:13-17,78` — `RolloutScheduleMode` enum + metadata 已用 `schedule_mode`
- `vrl/rollouts/orchestration/schedule.py:60-62` — `build_rollout_schedule` 读 `config.mode`
- `vrl/ray/resources.py:85-94,337` — `ActorLeasePolicy.mode`（resident/on_demand），派生于 topology
- `vrl/generation/ray/launcher.py:205`、`vrl/generation/ray/config.py:164` — lease mode 消费点
- `vrl/config/schema.py:216-222,248` — `SdeConfig.type` + `denoise_mode` 两个正交 Literal 轴
- `vrl/generation/diffusion/layout.py:60,85,133,268-281` — `sde_type` 字段/默认/guard + `denoise_mode` guard
- `vrl/generation/diffusion/executor.py:60,311,324,710` — `sde_type` 默认 + `denoise_mode=="native"` 分支
- `vrl/math/diffusion/flow_matching.py:116` — `if sde_type == "cps"` 唯一数学变体分支
- `vrl/scripts/common/factory.py:191-192`、`vrl/scripts/eval/cosmos_predict25_kling_eval.py:280,476`、`vrl/scripts/diffusion/cosmos/anima/generate.py:268` — 脚本侧 `sde_type` 默认/读取
- `configs/base/rollout/diffusion.yaml:7,13` — `denoise_mode: sde` 与 `sde.type: sde` 并存（含确认注释）
- `configs/base/rollout/orchestration/{strict,continuous}.yaml:6,32` — 调度 `mode:` YAML
- `tests/rollouts/orchestration/test_orchestration.py:11-17`、`tests/rollouts/orchestration/continuous/test_schedule.py:101-107` — config fake 持有 `mode`/`weight_sync_barrier`
- 前序拍板：[[SPRINT_design_smell_loose_ends]]、[[SPRINT_design_smell_audit]]、[[SPRINT_resolved_struct_field_audit]]、[[SPRINT_config_string_settings]]、[[SPRINT_ray_phase_lifecycle_plan]]
- 相关命名 sprint：[[SPRINT_precision_naming_unification]]（同类 token 同形消歧主题）