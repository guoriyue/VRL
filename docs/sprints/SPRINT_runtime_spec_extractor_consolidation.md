# SPRINT: Runtime Spec Extractor Consolidation

状态：proposed。

## 0. Core Decision

`RuntimeBuildSpec` 这一层（whole RL cfg → 运行时窄切片 → builder）保留。问题在 cfg→spec 的解析：

- **旧现实**：每个 family 各写一份 `extract_<family>_runtime_spec`，约 85% 逐字拷贝。
- **Codex 的过度矫正**：抽成一个 `ConfigField(path, spec_key, cast, ...)` DSL + 字符串路径遍历，把 copy-paste 换成了反射式框架——失去 `cfg.model.lora.rank` 这种直接可 grep 的访问，复杂家（anima/janus/nextstep）还是 10–13 行 `ConfigField`，且 AR 的采样字段被映射进 diffusion 的具名 `scheduler_config` 槽，制造槽语义错配。

**本 sprint 的方向（最简）：不裁剪字段，整块驮。** extractor 把运行时相关的 config 块（`model` + `sampling`）原样深转成 plain dict 挂在 spec 上，**所有 family 共用同一个 extractor**。每个模型在 `from_spec` 里各取所需；用不到的字段惰性躺着、零代价。

底层逻辑：spec 是数据载体，不是 schema。**不需要按 family 声明"我用哪些字段"——驮全部，模型自己挑。** 这样：
- 没有 per-family 字段裁剪（消灭 5–7 份拷贝 / ConfigField 列表）。
- 没有 typed 槽语义错配（AR 的 `cfg_weight` 就在 `sampling_config` 里，diffusion 无视它）。
- AR 与 diffusion 统一，不需要"AR 要不要分开"的纠结——双方各驮各的字段，互相无视。

## 1. Current Code Reality

`extract_*_runtime_spec` 跨 5 个 diffusion family（wan / sd3 / cosmos-predict2 / cosmos-predict2.5 / anima）+ 2 个 AR family（janus / nextstep）重复同一形状。当前树里 Codex 已用 `vrl/models/runtime_config.py` 的 `ConfigField` DSL 统一过一版（green），但它是反射式框架 + 字符串路径，不符合 AGENTS.md 的「concrete, greppable」命名与「不过度抽象」原则。

YAML 本来就是统一形状（`model.*` / `sampling.*` 每家一致），所以解析可以统一——但**统一的正确方式是整块驮，不是逐字段映射**。

## 2. Target Shape

### 2.1 Spec 驮原始块

`RuntimeBuildSpec` 用两个 plain-dict 块取代当前一堆 curated typed 字段（`lora_config` / `scheduler_config` / `memory_config` / `offload_config` / `extra` / `native_backend_config` / `diffusers_backend_config`）：

```python
@dataclass
class RuntimeBuildSpec:
    model_name_or_path: str
    device: Any
    dtype: Any
    task_variant: str
    backend_preference: tuple[str, ...]
    model_config: dict[str, Any] | None = None      # 整个 cfg.model，深转 plain
    sampling_config: dict[str, Any] | None = None    # 整个 cfg.sampling，深转 plain
```

`model_config` 里就有 `path` / `lora` / `memory` / `torch_compile`，以及 anima 的 checkpoint 路径、AR 的 `freeze_vq` 等——**全部原样驮着**，谁用谁读。

### 2.2 一个统一 extractor，全家共用

```python
# vrl/models/runtime_config.py（替换 ConfigField DSL）
def extract_runtime_spec(
    cfg: Any,
    device: Any,
    dtype: Any,
    *,
    task_variant: str,
    backend_preference: tuple[str, ...],
) -> RuntimeBuildSpec:
    return RuntimeBuildSpec(
        model_name_or_path=str(cfg.model.path),
        device=device,
        dtype=dtype,
        task_variant=task_variant,
        backend_preference=backend_preference,
        model_config=plain_mapping(cfg.model, field_name="model"),
        sampling_config=_optional_block(cfg, "sampling"),
    )
```

`task_variant` 和 `backend_preference` 都在 registry `RolloutFamilyEntry` 里（`entry.task` + backend）。所以更彻底的版本是**registry 直接驱动**，连每家 wrapper 都不需要——launcher 用 `entry.task` 调统一 extractor。第一版可保留 family 薄 wrapper（改动面最小、registry 字符串引用不变）。

### 2.3 消费侧在 from_spec 里读

模型从 `spec.model_config` / `spec.sampling_config` 取自己要的：

```python
# wan from_spec
num_steps = spec.sampling_config["num_steps"]
lora = spec.model_config.get("lora")
vae_decode = (spec.model_config.get("memory") or {}).get("vae_decode")

# janus from_spec
cfg_weight = spec.sampling_config.get("cfg_weight", 5.0)
freeze_vq = spec.model_config.get("freeze_vq", True)
```

diffusion 不读 `cfg_weight`，AR 不读 `num_steps`——各取所需，互不干扰。

## 3. Plan

### 3.1 删 ConfigField DSL

删除 `vrl/models/runtime_config.py` 里的 `ConfigField` / `config_value` 字符串路径遍历 / `_config_get` 三重 fallback，替换为 §2.2 的整块驮 extractor。

### 3.2 重塑 RuntimeBuildSpec

curated 字段 → `model_config` + `sampling_config` 两个原始块（§2.1）。保留 universal typed 顶层字段（`model_name_or_path` / `device` / `dtype` / `task_variant` / `backend_preference`）因为它们每家都要、且部分是运行时注入。

### 3.3 每家 extract 收成 wrapper（或 registry 驱动）

`extract_<family>_runtime_spec` 只剩调用统一 extractor + 传 `task_variant`（取 registry `entry.task`）+ `backend_preference`。无字段裁剪。

### 3.4 消费侧迁移

所有 `from_spec` 改读 `spec.model_config` / `spec.sampling_config`（替换原来读 `spec.lora_config` / `spec.scheduler_config` / `spec.extra`）。memory 仍走 `apply_vae_decode_memory(memory_config=spec.model_config.get("memory"))`。

### 3.5 replay 变体

`extract_<family>_replay_runtime_spec` 原来 trim `spec.extra`。整块驮后，replay 与 full 的差异退化为"读不读某些块"——若 replay 确实需要排除某些字段，让 family 声明一个小的排除集，trim 逻辑共享一个 helper。多数情况整块驮后可能根本不需要 trim（replay 模型只读它要的，多余字段惰性躺着）。逐家核实后再决定是否保留 trim。

## 4. 接受的取舍（你已确认）

- **整块驮 ⇒ 顶层 config typo 静默忽略**。`model.lroa.rank`（打错）不会报错，只是 `lora` 块缺字段。这和别的 config 一个待遇（`cfg.model.lroa` 也是用到才炸），是「load all, ignore unused」的代价，已接受。
- **子块级校验仍在 consumer**：memory 的 `vae_decode` key 校验仍在 `vae_decode_memory_from_config`（derived keys），lora rank 在 `int()` 转换处。真正会炸的地方仍然炸。

## 5. What Should Stay Unchanged

- 只驮**运行时相关块（model + sampling）**，不驮 reward / algorithm / trainer / dataset——`RuntimeBuildSpec` 窄化的初衷（不让 builder 依赖整个 RL cfg）保留。
- `device` / `dtype` 继续运行时注入，不进 YAML。
- family 真正独有的**加载逻辑**（anima 从独立文件读 transformer/vae、AR 的 freeze 行为）留在各家 `from_spec`；它们读 `model_config` 里的原始字段，不在 extractor 里特判。
- registry 仍是 family 接线的真相源（`entry.task` 供 `task_variant`）。

## 6. Thin Functions / Constants Hygiene

### Should keep
- 一个 `extract_runtime_spec`：移除真重复的共享抽象，直接属性访问 + `plain_mapping` 整块深转，无字符串路径、无反射框架。
- `plain_mapping`（已在 `vrl/utils/config.py`）：唯一的 OmegaConf→plain 深转，复用。

### Should not add / 红线
- 不要 `ConfigField` / `config_value(path_string)` 这类字符串路径 DSL——直接 `cfg.model.x` 更可 grep、写错当场炸（AGENTS.md「greppable」「concrete naming」）。
- 不要按 family 复制 extractor；解析路径只有一份。
- 不要把 reward/algorithm 也驮上 spec——窄化边界保留。
- 校验 key 集一律 derived（AGENTS.md「派生，不要手维护大写常量集合」）。

## 7. Non-Goals

- 不重排 YAML 布局、不上 Pydantic 全量 schema。
- 不在本 sprint 处理 build 侧 `*_from_cfg` 包装（`extract→build` compose 重复，另议）。
- 不改 reward / algorithm / dataset 的 config 路径。

## 8. Verification

```bash
python -m ruff check vrl/ tests/
python -m pytest -q -p no:cacheprovider     # 期望维持 538 passed / 8 skipped
```

行为不变断言：7 家 `extract_*_runtime_spec(cfg, device, dtype)` 产出的 `RuntimeBuildSpec`，其 `model_config` / `sampling_config` 深转后包含原 curated 字段的等价值（参数化测试：每家跑 canonical cfg，断言 from_spec 读出的关键值与重构前一致）。

## 9. Critical Files

核心实现：
```text
vrl/models/runtime_config.py            # 删 ConfigField DSL，换整块驮 extractor
vrl/models/interfaces/runtime.py        # RuntimeBuildSpec: model_config + sampling_config
vrl/utils/config.py                     # plain_mapping（复用）
```

family 接线（收成 wrapper）+ 消费侧（from_spec 改读块）：
```text
vrl/models/diffusion/{wan_2_1,sd3_5,cosmos/predict2,cosmos/predict2_5,cosmos/anima}/{runtime,model}.py
vrl/models/ar/{janus_pro,nextstep_1}/{runtime,model}.py
```

registry / launcher（registry 驱动版需动）：
```text
vrl/rollouts/families/registry.py
vrl/generation/ray/launcher.py
```

tests：
```text
tests/rollouts/test_runtime_inputs.py
tests/config/test_load_all_experiments.py
```
