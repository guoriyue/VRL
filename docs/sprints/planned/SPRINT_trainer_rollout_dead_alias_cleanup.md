# SPRINT: Prune trainer/rollout/trajectory dead aliases and duplicated single-source-of-truth logic (planned)

状态：全部落地（2026-06-20）。#1–#9 全部完成。
范围：清理 trainer / rollout / trajectory / scripts 四层里残留的死别名（legacy alias、back-compat shim）以及重复实现的“单一真相”逻辑（同一概念多处复制、同一标识符存三份）。不动外部 wire 协议，不重做精度命名（已由 [[SPRINT_precision_naming_unification]] 覆盖）。

## 落地状态（2026-06-20，分支 sprint/trainer-rollout-dead-alias）

- ✅ **#1 video_reward 死别名** → 删三处 fallback + 模块别名（零 config 命中、零调用）。
- ✅ **#5 SegmentSignal 三份标识符** → 删 `segment` 字段，`__post_init__` 改校验 `name == dict key`（7 个构造点 `name==segment`，等价）。
- ✅ **#6 phase_times back-compat 属性** → 删 `RolloutIteration` / `ContinuousRolloutItem` 两处属性，~7 处测试改读 `stats.as_phase_dict()`。
- ✅ **#2 `_train_timestep_indices` SSOT** → full-batch 路径（`train_on_rollout_batch`）原样抄了一遍的 count/step-size/range 算式 → 改调方法。
- ✅ **#3 DPO legacy `step` key** → `state_dict()` 只写 `global_step`，删 `state.get("step", 0)` 死 fallback。
- ✅ **#7 `role_tensor` 双份** → resolver 方法保留 segment-name 解析（`_fail` 未知段），role 匹配委托给 views 的唯一实现。
- ✅ **#4 每-prompt 采样数收敛** → 删 vestigial `default_group_size`（对 diffusion 默认值是错的且从不被用）+ `group_size` 提为 `collect`/`collect_unscored` 的 keyword-only 必填参数。`samples_per_prompt`（GenerationRequest 字段，生成域深嵌 20+ 处）与 `group_size`（GRPO 域）**各自是其领域的准确名字，不是重复 → 不抹平**（按 doc §D「keep + 注释」）；在 `requests.py` 边界加注释点明三个域名指同一数；public key `n_samples_per_prompt` / `eval.samples_per_prompt` 按设计保留（需独立弃用路径）。`EvalConfig.samples_per_prompt` 已有 eval-only 区分注释。
- ✅ **#9 AR 入口去 `ocr` 词** → `train_janus_pro_ocr_grpo` / `train_janus_pro_r1_ocr_grpo` / `train_nextstep_1_ocr_grpo` → `train_*_grpo`（函数名 + `__all__` + docstring），同步迁移 4 个 experiment config 的 `trainer.entrypoint`，**不留旧别名**（repo cleanup，直接迁移 shipped config）。run 输出目录名 `outputs/*_ocr_grpo` 保持不动（描述真实 OCR 运行，改了会孤立已有 run）。lint + 142 tests passed。
- ✅ **#8 轴名 `"timestep"`→`"denoise"`** → diffusion 轨迹的去噪步轴名与其 kind `"denoise_step"` / 段名 `"denoise"` 对齐。**外科式重命名**：只改轴名引用（`builders.py` 轴定义 + `("sample","timestep")` 张量轴元组 + `axis_lengths` + `capabilities.py` 的 `AxisCapability`/`ExecutionStageCapability` + `base.py` replay axis= + `planner.py` `_axis_length` 判断 + 测试），**保留**张量名 `"timesteps"`（复数）、`tensor_ref` 段名、以及所有 transformer-input 标量值键 `"timestep": <value>`（cosmos model / cfg.py）与 `gather.py` byte_values 的 wire 键。trajectory/generation/diffusion/algorithms/trainers 508 tests + config lint passed。

## 0. Core Decision（先看这一段）

仓库里目前并存一批“同一个东西有两套名字/两份实现”的痕迹，它们都不是真正的外部边界，而是迁移/复制残留。本 sprint 把它们收敛到唯一真相：

1. `video_reward` 是 `kling_video_reward` 的**无人使用**短拼写别名，三条 `OmegaConf.select(...) or OmegaConf.select(...)` fallback + 一个 `validate_production_video_reward_config` 模块别名，全仓库零调用、零 config 命中 → 删。
2. 去噪 timestep 索引展开的算式被标注“single source of truth”的 `_train_timestep_indices` 持有，但 `train_on_rollout_batch` 又**原样抄了一遍** → 改成调用方法。
3. `OfflineDPOTrainer.load_state_dict` 读一个自己 `state_dict()` 从不写出的旧 key `step` → 删（或显式标注 legacy-format）。
4. “每 prompt 采样数”这一个数沿 trainer→rollout 链被改名 4 次（`n_samples_per_prompt`→`group_size`→`default_group_size`→`samples_per_prompt`），其中 `samples_per_prompt` 还和 eval-only 字段同名 → 收敛内部命名。
5. `SegmentSignal` 把同一段标识符存了三份（dict key + `name` + `segment`）并用 `__post_init__` 自校验三者相等 → 砍到一份。
6. `RolloutIteration.phase_times` / `ContinuousRolloutItem.phase_times` 是显式标注 “Read-only back-compat view” 的死属性，生产只读 `.stats`，仅测试还在读属性 → 删属性，测试改读 `stats.as_phase_dict()`。
7. `role_tensor` 同名同逻辑存在两处（`views.py` 自由函数 + `resolver.py` 方法），错误串已经开始漂移 → 让方法委托给函数。
8. diffusion 轴名 `"timestep"` 与其 kind `"denoise_step"` 不一致，且与同轴张量名 `"timesteps"` 撞词 → 轴名改为与 kind/段名一致的 `"denoise"`。
9. AR 训练入口名里硬编码了 `ocr`（`train_janus_pro_ocr_grpo` 等），但入口是 reward-agnostic 的（diffusion 侧就是 `train_sd3_5_grpo`），导致 aesthetic recipe 指向 `..._ocr_grpo` 函数 → 去掉 reward 词。

这些改动彼此独立，可分批落地；除“每 prompt 采样数”涉及 public YAML key（需保留别名/弃用路径）外，其余均为内部清理。

## 1. 现状实锤

### 1.1 `video_reward` 死别名（config 层，三处 fallback + 一个模块别名）

`vrl/config/validation.py:131-134` reward kwargs 处接受短拼写：

```python
vr_kwargs = (
    OmegaConf.select(cfg, "reward.kwargs.kling_video_reward")
    or OmegaConf.select(cfg, "reward.kwargs.video_reward")
    or {}
)
```

`vrl/config/validation.py:343-344` 与 `vrl/scripts/common/online.py:994-995` 各有一处 gate fallback：

```python
OmegaConf.select(cfg, "production.kling_video_reward.enabled", default=False)
or OmegaConf.select(cfg, "production.video_reward.enabled", default=False)
```

模块级别名 `vrl/config/validation.py:349` + `__all__` 项 `vrl/config/validation.py:359`：

```python
validate_production_video_reward_config = validate_production_kling_video_reward_config
```

证据：`grep "video_reward" configs/ | grep -v kling_video_reward` 零命中（无任何 shipped config 用短拼写）；`grep "validate_production_video_reward_config" --include=*.py .` 仅命中定义与 `__all__`，**零调用**。仓库内唯一调用点 `vrl/config/validation.py:346` 用的是 canonical 名 `validate_production_kling_video_reward_config(cfg)`。

### 1.2 去噪 timestep 索引：方法是 SSOT，但 full-batch 路径又抄了一遍

`vrl/trainers/online/trainer.py:573-579` 是被显式标注的单一真相：

```python
@staticmethod
def _train_timestep_indices(num_timesteps: int, timestep_fraction: float) -> list[int]:
    """Denoise timesteps that receive loss (single source of truth)."""
    train_timestep_count = max(1, int(num_timesteps * timestep_fraction))
    if train_timestep_count < num_timesteps:
        step_size = num_timesteps / train_timestep_count
        return [int(i * step_size) for i in range(train_timestep_count)]
    return list(range(num_timesteps))
```

streaming 路径 `vrl/trainers/online/trainer.py:628` 已正确调用它。但 `train_on_rollout_batch` 在 `vrl/trainers/online/trainer.py:854-859` 内联复制了同一算式：

```python
train_timestep_count = max(1, int(num_timesteps * cfg.timestep_fraction))
if train_timestep_count < num_timesteps:
    step_size = num_timesteps / train_timestep_count
    train_indices = [int(i * step_size) for i in range(train_timestep_count)]
else:
    train_indices = list(range(num_timesteps))
```

两份逐字相同；任何一边改 rounding 都会让 streaming 与 full-batch 在同一 config 下训练不同的 timestep 集合。

### 1.3 DPO checkpoint 读一个自己从不写的 `step` key

`vrl/trainers/offline/dpo.py:401-407` 的 `state_dict()` 只写 `global_step`：

```python
return {
    "global_step": self.global_step,
    "optimizer": self._optimizer.state_dict(),
}
```

但 `vrl/trainers/offline/dpo.py:414` 读时多兜了一层 `step`：

```python
self.global_step = int(state.get("global_step", state.get("step", 0)))
```

`state.get("step", 0)` 分支只能命中旧格式 ckpt，当前代码不产出该 key。

### 1.4 “每 prompt 采样数”一个数被改名 4 次

- trainer 配置字段：`vrl/trainers/core/types.py:228` `n_samples_per_prompt`
- 进入 rollout 层即改名：`vrl/trainers/online/trainer.py:491` `group_size=cfg.n_samples_per_prompt`
- collector 又叫 `default_group_size`：`vrl/rollouts/collector/core.py:58,68`，并在 `core.py:261` 从 `config.require("n_samples_per_prompt")` 灌入
- request builder 再改回 `samples_per_prompt`：`vrl/rollouts/collector/requests.py:68` `samples_per_prompt=group_size`

而 `vrl/trainers/core/types.py:76` 的 `EvalConfig.samples_per_prompt` 是**互不相干**的 eval-only 字段，与 request 层的训练组大小同名，加重歧义。

### 1.5 `SegmentSignal` 把同一标识符存三份并自校验相等

`vrl/rollouts/evaluators/types.py:13-15` 同时有 `name` 与 `segment`：

```python
name: str
segment: str
axis: str
```

`vrl/rollouts/evaluators/types.py:51-54` 在 batch 的 `__post_init__` 里强制三者相等（dict key == name == segment）：

```python
if segment.segment != name:
    raise ValueError(
        f"trajectory signal key {name!r} must match segment={segment.segment!r}",
    )
```

构造侧 `vrl/rollouts/evaluators/trajectory.py:132` 永远传相同值 `SegmentSignal(name=name, segment=name, ...)`。`grep` 证实 `.segment` 字段唯一读者就是上面这条自校验（无任何控制流读 `.segment`）。三份拷贝里两份是纯腐烂风险。

### 1.6 `phase_times` 是死的 back-compat 属性

`vrl/rollouts/orchestration/types.py:49-50` 与 `vrl/rollouts/orchestration/continuous/types.py:67-68` 各有一个属性，docstring 自承 “Read-only back-compat view”：

```python
@property
def phase_times(self) -> dict[str, float]:
    """Read-only back-compat view of this iteration's phase timings."""
```

`vrl/utils/stats.py:4` 的迁移说明已点名它取代了手工穿线的 `phase_times` dict。`grep "\.phase_times"` 证实生产只读 `.stats`（`strict_on_policy.py` / `lifecycle.py` 里的 `phase_times` 是本地 dict 变量，与属性无关），唯一读属性的是 `tests/rollouts/orchestration/continuous/test_contracts.py:133-134,393-395` 与 `tests/rollouts/orchestration/continuous/test_schedule.py:203`。

### 1.7 `role_tensor` 同名同逻辑两份实现

自由函数 `vrl/trajectory/views.py:131-138`（被广泛 import，`vrl/trajectory/__init__.py:46,79` 导出，`models/ar/janus_pro/model.py`、`rollouts/evaluators/trajectory.py` 等使用）：

```python
def role_tensor(segment: TrajectorySegment, role: str) -> TrajectoryTensor:
    matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"segment {segment.name!r} requires exactly one role {role!r}, "
            f"found {len(matches)}",
        )
    return matches[0]
```

方法 `vrl/trajectory/resolver.py:76-88` 逻辑相同，只是先按名查 segment，且用 `_fail` 而非 `RuntimeError`，错误串已经和函数版略有差异。

### 1.8 轴名 `"timestep"` 与 kind `"denoise_step"` 及张量 `"timesteps"` 三词打架

taxonomy `vrl/trajectory/types.py:23` 标准化为 `"denoise_step"`。但 builder `vrl/trajectory/builders.py:104` 造出实例名为 `"timestep"` 的轴：

```python
"timestep": TrajectoryAxis("timestep", "denoise_step", timestep_count),
```

同轴上还有一个张量字面叫 `"timesteps"`（`vrl/trajectory/builders.py:67-70`）。对照 token 段 `vrl/trajectory/builders.py:166` `TrajectoryAxis("token", "discrete_token", ...)`——轴名与段名一致；diffusion 段名是 `"denoise"`（`builders.py:107-108`），轴却叫 `"timestep"`，唯独这里不对齐。

### 1.9 AR 入口名硬编码 `ocr`，但入口 reward-agnostic

diffusion 侧入口不带 reward 词：`vrl/scripts/diffusion/sd3_5/train.py:19` `train_sd3_5_grpo`、`vrl/scripts/diffusion/wan_2_1/train.py:20` `train_wan_2_1_grpo`。AR 侧却把 `ocr` 写进名字：`vrl/scripts/ar/janus_pro/train.py:16,22`（`train_janus_pro_ocr_grpo` / `train_janus_pro_r1_ocr_grpo`）、`vrl/scripts/ar/nextstep_1/train.py:16`（`train_nextstep_1_ocr_grpo`）。reward 来自 config，入口只绑 `family + task_variant`，所以名字里的 `ocr` 对非 OCR recipe 是谎言——`configs/experiment/ar/janus_pro/online_r1_grpo_aesthetic.yaml:36` 就把 aesthetic recipe 接到了 `train_janus_pro_r1_ocr_grpo`，读起来像复制粘贴错误。

全部引用（rename 时需原子更新）：

```
configs/experiment/ar/nextstep_1/online_grpo_ocr.yaml:24   -> train_nextstep_1_ocr_grpo
configs/experiment/ar/janus_pro/online_r1_grpo_ocr.yaml:14 -> train_janus_pro_r1_ocr_grpo
configs/experiment/ar/janus_pro/online_grpo_ocr.yaml:29    -> train_janus_pro_ocr_grpo
configs/experiment/ar/janus_pro/online_r1_grpo_aesthetic.yaml:36 -> train_janus_pro_r1_ocr_grpo
vrl/scripts/ar/nextstep_1/train.py:52  __all__
vrl/scripts/ar/janus_pro/train.py:81-82 __all__
```

## 落地方案

### A. 删 `video_reward` 死别名
- `vrl/config/validation.py:132-134`：去掉 `or OmegaConf.select(cfg, "reward.kwargs.video_reward")`，只留 `kling_video_reward`。
- `vrl/config/validation.py:343-344` 与 `vrl/scripts/common/online.py:994-995`：删 `or ... production.video_reward.enabled ...` 分支。
- `vrl/config/validation.py:349` 别名行 + `vrl/config/validation.py:359` 的 `__all__` 项一并删除，只保留 `validate_production_kling_video_reward_config`。

### B. timestep 索引收敛到方法
- `vrl/trainers/online/trainer.py:854-859` 整块替换为 `train_indices = self._train_timestep_indices(num_timesteps, cfg.timestep_fraction)`，与 streaming 路径 `:628` 对齐。

### C. DPO `step` legacy key
- `vrl/trainers/offline/dpo.py:414` 改为 `self.global_step = int(state.get("global_step", 0))`；若团队确认仍需加载 rename 前的旧 ckpt，则保留兜底但加注释标 `legacy-format alias`，不让它被误读为活 key。

### D. “每 prompt 采样数”内部命名收敛
- rollout 层主流拼写是 `group_size`，把内部链路统一到它：`vrl/rollouts/collector/core.py:58,68`（`default_group_size`→`group_size`）与 `vrl/rollouts/collector/requests.py:68`（`samples_per_prompt=group_size`→ 直接 `group_size=group_size` 或保留 request 字段名但加注释指明=训练组大小）。
- public YAML key `n_samples_per_prompt`（`config/schema.py`、`vrl/trainers/core/types.py:228`）与 `eval.samples_per_prompt`（`types.py:76`）**保持不变**（user-config surface，需别名/弃用路径，本 sprint 不动），仅在 `EvalConfig.samples_per_prompt` 注释里写明它是 eval-only、与训练组大小不同。

### E. `SegmentSignal` 砍到一份标识符
- dict key（`TrajectorySignalBatch.segments`）是真相。删 `vrl/rollouts/evaluators/types.py:14` 的 `segment` 字段，删 `:51-54` 的 `segment.segment != name` 自校验；`name` 留作自描述字段，由 batch 的 `__post_init__` 改为“key 不一致则纠正/赋值”而非断言三者相等。
- 更新构造点 `vrl/rollouts/evaluators/trajectory.py:132`、`vrl/scripts/perf/fp8_rollout_drift_probe.py` 及相关 tests，去掉 `segment=...` 传参。

### F. 删 `phase_times` 死属性
- 删 `vrl/rollouts/orchestration/types.py:49-50` 与 `vrl/rollouts/orchestration/continuous/types.py:67-68` 两个属性。
- 同一改动里把 `tests/rollouts/orchestration/continuous/test_contracts.py:133-134,393-395` 与 `tests/rollouts/orchestration/continuous/test_schedule.py:203` 改成 `iteration.stats.as_phase_dict()[...]` / `item.stats.as_phase_dict()[...]`。

### G. `role_tensor` 单一实现
- 让 `vrl/trajectory/resolver.py:76-88` 在解析出 segment 后委托给 `vrl/trajectory/views.py:131` 的自由函数，保留方法自身的“unknown segment”错误路径（`:80-81`），唯一性判定与错误串只留函数版一处。

### H. 轴名对齐
- `vrl/trajectory/builders.py:104` 轴名 `"timestep"` 改为 `"denoise"`（与段名/kind 对齐），同步更新 `builders.py:46,52,58,64,70,76` 所有 `("sample", "timestep")` 轴元组、`:132` `axis_lengths` 的 `"timestep"` key，以及任何按名读取 `"timestep"` 轴的 diffusion evaluator / `_loss_axis`（`views.py:141`）/ sde_logprob 消费方。replay 张量仍叫 `"timesteps"`（消除轴名/张量名撞词）。

### I. AR 入口去 reward 词
- 重命名：`train_janus_pro_ocr_grpo`→`train_janus_pro_grpo`、`train_janus_pro_r1_ocr_grpo`→`train_janus_pro_r1_grpo`、`train_nextstep_1_ocr_grpo`→`train_nextstep_1_grpo`，更新各自 `__all__`。
- 原子更新 1.9 列出的 4 个 config `entrypoint:` 行。

## 验证（finishing criteria）

- `grep -rn "video_reward" configs/ vrl/ | grep -v kling_video_reward` 零命中；`grep "validate_production_video_reward_config"` 零命中。
- `grep -n "train_timestep_count" vrl/trainers/online/trainer.py` 只剩方法 `_train_timestep_indices` 内一处。
- `grep -rn "\.phase_times" vrl/ tests/` 不再命中属性定义；测试改读 `as_phase_dict()` 后 `pytest tests/rollouts/orchestration/continuous/` 全绿。
- `grep -rn "ocr_grpo" vrl/scripts/ configs/` 仅命中 diffusion/sd3_5/wan 的 OCR **recipe 文件名**，不再命中任何 AR 入口函数名或 `entrypoint:` 值；对 4 个 AR config 跑 config-resolve（dry import entrypoint）成功。
- `grep -rn "\.segment\b" vrl/rollouts/evaluators/` 不再有 `SegmentSignal.segment`；GRPO 多段/token/continuous 算法 `pytest tests/algorithms/grpo/` 全绿。
- `pytest tests/trajectory tests/trainers/offline -q` 全绿（role_tensor 委托、DPO load_state_dict）。
- 全量 `pytest -q` 与现有 config 解析无回归。

## 非目标 / Non-Goals

- 不重做精度命名（`mixed_precision`/`bf16`/`'no'` 三拼写、`fsdp.mixed_precision` 同形词）——已由 [[SPRINT_precision_naming_unification]] 覆盖；本 sprint 不碰 `TrainerConfig.bf16`。
- 不改 public YAML key `n_samples_per_prompt` / `eval.samples_per_prompt`（user-config surface，需独立弃用路径）。
- 不动任何外部 wire / 协议边界；不为已删的 `video_reward`/`step`/`phase_times` 引入新的兼容层（确认无外部依赖后直接删）。
- 不扩展到其他 sprint 历史产物的清理。

## References

- `vrl/config/validation.py:108,131-146,156,163,343-346,349,357,359`
- `vrl/scripts/common/online.py:618,990,994-995,998,1001,1004-1005`
- `vrl/trainers/online/trainer.py:491,509,563,573-579,628,719,768,800,842,854-859,1210`
- `vrl/trainers/offline/dpo.py:401-407,414-418`
- `vrl/trainers/core/types.py:75-87,228`
- `vrl/rollouts/collector/core.py:58,68,123-124,260-261`
- `vrl/rollouts/collector/requests.py:44,68`
- `vrl/rollouts/evaluators/types.py:9-67`
- `vrl/rollouts/evaluators/trajectory.py:11,132`
- `vrl/rollouts/orchestration/types.py:49-50`
- `vrl/rollouts/orchestration/continuous/types.py:67-68`
- `vrl/utils/stats.py:4`
- `vrl/trajectory/views.py:105-107,131-138,141-145,148-156`
- `vrl/trajectory/resolver.py:76-93`
- `vrl/trajectory/__init__.py:46,79`
- `vrl/trajectory/types.py:19-23,50-62`
- `vrl/trajectory/builders.py:46-76,89-92,103-108,132,165-166,368-369`
- `vrl/scripts/ar/janus_pro/train.py:16,22,81-82`
- `vrl/scripts/ar/nextstep_1/train.py:16,52`
- `vrl/scripts/diffusion/sd3_5/train.py:19`、`vrl/scripts/diffusion/wan_2_1/train.py:20`
- `configs/experiment/ar/nextstep_1/online_grpo_ocr.yaml:24`
- `configs/experiment/ar/janus_pro/online_grpo_ocr.yaml:29`
- `configs/experiment/ar/janus_pro/online_r1_grpo_ocr.yaml:14`
- `configs/experiment/ar/janus_pro/online_r1_grpo_aesthetic.yaml:36`
- `tests/rollouts/orchestration/continuous/test_contracts.py:133-134,393-395`、`tests/rollouts/orchestration/continuous/test_schedule.py:203`
- 关联：[[SPRINT_precision_naming_unification]]、[[SPRINT_design_smell_audit]]、[[SPRINT_helper_passthrough_hygiene]]