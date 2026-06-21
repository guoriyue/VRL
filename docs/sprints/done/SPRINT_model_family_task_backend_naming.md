# SPRINT: 消除 model/nn 层 family / task_variant / backend 身份串的同名歧义 (done)

状态：done（2026-06-20）

## 落地结果（2026-06-20）

七项全部收敛，`pytest tests/models tests/nn tests/rollouts tests/generation
tests/trainers tests/config tests/trajectory` 全绿（751 passed）。grep 净化：
`"model_family"` / `cfg.model,"task"` / `backend_handle` / model 类 `.family`·
`.model_family` 定义 / nn 层裸 `"backend"` metric 键，在 `vrl/` 全部 0 命中。

- **A** 拆 `task_variant`：新增 `RuntimeBuildSpec.ar_task`，AR（janus/nextstep）改读写
  `ar_task`，diffusion 独占 `task_variant`。`ar_task` 随 `asdict` 走 Ray launch 契约
  序列化往返（worker 端 `RuntimeBuildSpec(**payload)` 不白名单字段，自动透传）。
- **B**（**改为删除，非改名**）：model 类的 `.family`（diffusion，"wan-diffusers-i2v"
  等）/ `model_family`（janus）属性在 **D** 删掉其唯一消费者（context 死写）后已**无任何
  读者**——按 AGENTS.md derived-struct 规则「既不被行为消费、也非 display-only 的字段即死字段，
  删除」，故直接删 8 处定义（base + 6 个 diffusion override + 2 个 janus），而不是改名为
  `model_label` 再留一个谁都不读的字段。sprint 目标「收敛 family 取值空间」由删除更彻底地达成。
- **C** 删 Wan `model.task` 死读分支。
- **D** 删 9 处 `"model_family"` context 死写 + 改 `test_batch.py` 用中性键测透传。
- **E** 修活 checkpoint family 守卫：14 个 builder 的 `RuntimeBundle.metadata` 写
  `"family"`，值取 **variant-aware 的 `family_capability.family`**（= 注册表 canonical key，
  含 `janus_pro_r1` / `wan_2_1_i2v`，与 checkpoint payload 的 `entry.family` 同空间，避免
  variant 误报）。新增 `test_restore_training_checkpoint_rejects_family_mismatch` 回归。
- **F** `backend_handle` → `raw_handle`（全仓 40 处，含 model property + bundle 字段 +
  `train_dpo` 读 + wiring 测试）；metric 键 `"backend"` 拆 `attention_backend`（per-family
  label）vs `attention_kernels`（kernel 类串）+ 同步 2 处测试；docstring「backend classes」→
  「family model classes」。保留 `attention_backend` kernel 选择器名。
- **G** family-aware model schema：本就已并行落地，确认无需改动。


范围：把 model+nn 层里"哪个 model family / 哪个 task 变体 / 哪个 backend"三个身份串的命名与读写歧义收敛——同一字段不再扛两套枚举，死读/死写/死守卫清掉，并给"backend"这个被 AGENTS.md 点名的过载词划清边界。不动 reward / algorithm / trainer 算法逻辑。

## 0. Core Decision（先看这一段）

model+nn 层有五处**身份串命名歧义/死代码**经逐文件复核全部成立且安全，应在一个 sprint 内一并收敛：

1. **`RuntimeBuildSpec.task_variant` 是同名异义字段**：diffusion 侧塞 `t2v`/`i2v`，AR 侧塞 `ar_t2i`/`ar_t2i_r1`——同一个共享 interface 字段扛两套互不相交的枚举，类型层（都是 `str`）看不出冲突。
2. **`family` 一词散落四个互不兼容的取值空间**：`cfg.model.family`（`"wan"`/`"janus_pro"`）、`DiffusionModelBase.family`（`"wan-diffusers-i2v"`）、`JanusProModel.model_family`（`"janus-pro-t2i"`），三个都叫 family/model_family 却永不相等。
3. **Wan 的 `cfg.model.task` fallback 读的是 schema 里不存在的键**——死的"容忍拼写"分支。
4. **`model_family` 这个 trajectory/rollout context 键每个 family 都写、生产侧零读**——只是把 `self.family` 换个键名再编码一遍的死写。
5. **checkpoint 的 family-mismatch 守卫永久失效**：`restore_training_checkpoint` 读 `bundle.metadata.get("family")`，但所有 family 的 `RuntimeBundle.metadata` 从不写 `"family"` 键 → `bundle_family` 恒为 `None` → 守卫永远短路不触发。写方（checkpoint payload）和读方（bundle）对键名不一致。
6. **"backend" 一词在 model/nn 层过载四义**（`backend_handle` 原始 pipeline 对象 / `attention_backend` kernel 选择器 / metric 键 `"backend"` 两套取值空间 / docstring 里"backend classes" 指 family model 类）——AGENTS.md naming 规则明确点名 backend 已是"训练/推理引擎"义。
7. **`ModelConfig` 扁平 key 注册表不分 family，家族专属键泄漏全局命名空间**：`model.*` 的合法键全部平铺在一个 `Any`-typed `ModelConfig` 里，其中 `boundary_ratio` / `trainable_transformers`（Wan 2.2 双专家专属）、两个 `*_cpu_offload`（diffusers pipeline 专属）与通用键（`dtype`/`path`/`use_lora`）同处一个全局白名单。后果：在 SD3.5 config 上写 `boundary_ratio`，schema 不报错（对所有 family 都"合法"），而 SD3.5 loader 只 `.get()` 自己要的 → 该键被**静默忽略 = no-op 旋钮**（AGENTS.md 明确：user-facing 的 no-op key 是最坏情形）。

**已决定不做（从候选中剔除）**：`backend_label` 派生默认在 4 处重抄——这条 [[SPRINT_design_smell_audit]] 第二轮已复核并**撤回不做**（消费者 `.get(..., fallback)` 不是 dead，stub 测试走该路径），本 sprint 不再 revisit；#6 只处理 `backend_handle` / metric 键 / docstring 三处真歧义，不碰 `backend_label`。

核心原则（AGENTS.md naming）：**按角色而非通用词命名，一个字段永不扛两套枚举，写方/读方对同一键名达成一致；死读/死写/死守卫直接删**。

## 1. 现状实锤

### 1.1 `task_variant` 同名异义（diffusion t2v/i2v vs AR ar_t2i/ar_t2i_r1）

共享 interface 字段定义（`vrl/models/interfaces/runtime.py:59`）：

```python
task_variant: str | None = None
```

diffusion 侧把它规整成 `t2v`/`i2v`（`vrl/models/diffusion/wan_2_1/runtime.py:366`）：

```python
if text in {"image_to_video", "image-to-video", "i2v"}:
    return "i2v"
```

AR 侧把同一字段当成 `ar_t2i`/`ar_t2i_r1` 枚举用——`extract_janus_pro_runtime_spec` 写 `task_variant="ar_t2i"`（`vrl/models/ar/janus_pro/runtime.py:118-119`），训练脚本再覆写成 `ar_t2i_r1`（`vrl/scripts/ar/janus_pro/train.py:58, 76`）：

```python
spec.task_variant = "ar_t2i_r1"
```

replay builder 直接按这个 AR 枚举分支（`vrl/models/ar/janus_pro/runtime.py:81-85`）：

```python
family_capability = (
    JANUS_PRO_R1_FAMILY_CAPABILITY
    if spec.task_variant == "ar_t2i_r1"
    else JANUS_PRO_FAMILY_CAPABILITY
)
```

→ 同一 `task_variant` 字段，diffusion 读者学到的是"t2v/i2v"，AR 读者读到的是"ar_t2i_r1"，两套枚举无交集且类型层不可见。

### 1.2 `family` 一词的四个取值空间

| 站点 | 文件:行 | 取值样例 |
|---|---|---|
| 用户配置选择器 | `vrl/models/diffusion/wan_2_1/runtime.py:354` | `"wan"` / `"janus_pro"`（来自 `cfg.model.family`）|
| diffusion model 类属性 | `vrl/models/diffusion/wan_2_1/model.py:569` | `"wan-diffusers-i2v"` |
| AR model 类属性 | `vrl/models/ar/janus_pro/model.py:155` | `"janus-pro-t2i"` |

```python
# wan_2_1/runtime.py:354
family = str(cfg_get(cfg.model, "family", ""))
# wan_2_1/model.py:569
family = "wan-diffusers-i2v"
# janus_pro/model.py:155
model_family: str = "janus-pro-t2i"
```

`cfg.model.family` 经 `normalize_rollout_family`（`vrl/rollouts/families/registry.py:312`）只做别名映射、保持 `"wan"`/`"janus_pro"` 取值空间，被 `vrl/config/schema.py:530, 538-539` 的 capability gating 消费；而类 `.family` / `.model_family` 进的是别的 metric/context payload。三个都叫 family 却永不相等，读者必须 trace producer 才知道指哪个。

### 1.3 Wan 读不存在的 `cfg.model.task`

`_task_variant_from_cfg` 为一个概念读三个键（`vrl/models/diffusion/wan_2_1/runtime.py:347-357`）：

```python
explicit = cfg_get(cfg.model, "task_variant", None)
if explicit:
    return _normalize_task_variant(str(explicit))
task = cfg_get(cfg.model, "task", None)        # <- 死分支
if task:
    return _normalize_task_variant(str(task))
family = str(cfg_get(cfg.model, "family", ""))
```

但 `ModelSection` schema 只声明 `task_variant`、无 `task`（`vrl/config/schema.py:313`）：

```python
task_variant: Any = None
```

实测：`grep '^\s*task:' configs/` 零命中，`task_variant` 仅在 `configs/model/diffusion/wan_2_1/i2v_14b.yaml` 等处使用。`model.task` 是无配置、无 schema 的死读——用户若设 `model.task` 会被 unknown-key lint 拒，但代码暗示它被尊重。

### 1.4 `model_family` context 键：每家写、生产零读

每个 diffusion family 把 `"model_family": self.family` 写进 export context（`vrl/models/diffusion/cosmos/predict2/model.py:395`、`vrl/models/diffusion/wan_2_1/model.py:440`），AR 写进 trajectory context（`vrl/models/ar/janus_pro/runtime.py:398, 520`）：

```python
"model_family": getattr(self.model, "model_family", "janus_pro"),
```

全仓 grep `model_family` 键的读者：生产代码零命中。唯一引用是 `tests/rollouts/collector/test_batch.py:112`：

```python
assert combined.context["model_family"] == "cosmos"
```

——但该测试自己构造 `context={"guidance_scale": 7.0, "model_family": "cosmos"}` 再断言 `stack_batches` 把 context 透传，测的是**通用 context 合并**而非该键的生产语义。即：`model_family` 是把 `self.family` 换键名重编码一遍的死写，给 context payload 平添第三种 family 拼法。

### 1.5 checkpoint family 守卫永久死亡

写方——checkpoint payload 带 `"family"`（`vrl/trainers/checkpointing.py:122`），其值来自 `stack.family`（`vrl/scripts/common/online.py:563`），即 `normalize_rollout_family(cfg.model.family)`（`vrl/scripts/common/factory.py:62`）→ `"janus_pro"`/`"wan"` 取值空间。

读方——`restore_training_checkpoint` 拿 bundle 侧的同名键对比（`vrl/trainers/checkpointing.py:249-255`）：

```python
checkpoint_family = checkpoint.payload.get("family")
bundle_family = getattr(bundle, "metadata", {}).get("family")
if strict and checkpoint_family and bundle_family and str(checkpoint_family) != str(bundle_family):
    raise ValueError(...)
```

但所有 family 的 `RuntimeBundle.metadata` 只写 `model_path`/`task_variant`/`use_lora`，从不写 `"family"`（`vrl/models/ar/janus_pro/runtime.py:67-72, 95-100`、`vrl/models/diffusion/cosmos/predict2/runtime.py:114` 同形）：

```python
metadata={
    "model_path": spec.model_name_or_path,
    "task_variant": spec.task_variant,
    "use_lora": spec.use_lora,
    ...
}
```

→ `bundle_family` 恒 `None` → `and bundle_family` 恒短路 → 守卫永不触发。读者以为 resume 防住了"把 Wan checkpoint 灌进 Janus bundle"，实际没防。这是 dual-read 的最坏状态：守一个没人写的键。

### 1.6 "backend" 过载四义

| 含义 | 文件:行 | 取值 |
|---|---|---|
| 原始 pipeline / AR wrapper 对象 | `vrl/models/interfaces/runtime.py:162` | `backend_handle: Any` |
| attention kernel 选择器 | `vrl/nn/modules/ar_attention_backends.py:34` | `"vllm_paged"`/`"torch_native"` |
| metric 键（per-family label） | `vrl/nn/modules/ar_decoder.py:97` | `"backend": self.backend_label`（`"janus_pro_vllm_paged_attention"`）|
| metric 键（固定 kernel 类串） | `vrl/nn/kernels/attention/vllm_paged.py:279` | `"backend": "vllm_paged_attention_kernels"` |

同一个 metric 键 `"backend"` 在 `ar_decoder` 里是 per-family label、在 `vllm_paged` 里是固定 kernel 类串——两套取值空间；`backend_handle`（原始对象）和 `attention_backend`（kernel 选择器）是毫不相关的两个概念却共用 backend 一词。AGENTS.md naming 规则正点名 backend 在 RL 世界已是"训练/推理引擎"义。

`backend_handle` 的消费者（确认是真 public 契约，须随改）：`vrl/scripts/diffusion/wan_2_1/train_dpo.py:150`（`pipeline = bundle.backend_handle`）+ 7 个 family runtime 的 `backend_handle=model.backend_handle` 写入 + 多个 wiring 测试断言 `bundle.backend_handle is None`。

### 1.7 ModelConfig 扁平 key 注册表，家族专属键泄漏全局

> 状态更新（2026-06-20）：本节已在当前 patch 采用 family-aware model schema 落地。
> `ModelConfig` 只保留 shared keys；Wan / Cosmos / Janus / NextStep 各自有
> family-specific schema；unknown-key walker 通过 `model.family` 选择对应 key
> set。Wan 的两个互斥 CPU offload bool 同步收敛成
> `model.offload_mode: none|model|sequential`，旧 bool 在 Wan runtime 直接报错。

迁移前，`ModelConfig` 是个全 `Any`-typed 的扁平 key 白名单，注释自承"key registry, 值由 family loader 校验"（`vrl/config/schema.py:288-294`）：

```python
class ModelConfig(ConfigBase):
    family: str | None = None
    # Key registry: consumed by family runtime loaders.
    boundary_ratio: Any = None          # Wan 2.2 双专家专属
    dtype: Any = None
    enable_model_cpu_offload: Any = None        # diffusers pipeline 专属（已迁移）
    enable_sequential_cpu_offload: Any = None   # diffusers pipeline 专属（已迁移）
    ...
    trainable_transformers: Any = None   # Wan 2.2 双专家专属（schema.py:317）
```

family-专属键的真实消费面（确认只 Wan 2.2 读）：`boundary_ratio` 仅 `vrl/models/diffusion/wan_2_1/runtime.py:158-181` + eval 用 `getattr(pipe.config,"boundary_ratio",None) is None` 判是不是双专家（`vrl/scripts/eval/wan_i2v_base_sample.py:112`，2.1 单塔无此键）；`trainable_transformers` 仅 `runtime.py:168` + `vrl/models/diffusion/wan_2_1/model.py:101-113`；旧的两个 `*_cpu_offload` 已收敛为 `model.offload_mode`，由 Wan runtime 统一分派到 Diffusers 的互斥 accelerate API。

原问题：这些键对**所有** family 都"合法"，但只有部分 family 读。错放（如 SD3.5 config 设 `boundary_ratio`）走到对应 loader 的 `.get()` 时被静默丢弃——既不报错也不生效。且每加一个 family 专属旋钮，都得手改这个中央注册表。设计意图（key 白名单 + family 校验 value）本身合理，缺的是**按 family 分域**。

### 1.8 落地取舍（需拍板）

三条路，按收益/churn 排序：

1. **family loader fail-loud（推荐）**：保持扁平白名单过 schema，但让每个 family 的 runtime loader 在见到它不消费的 `model.*` 键时**报错**而非静默 `.get` 忽略——把"错放 = no-op"变成"错放 = 启动即拒"。最小改动消除最坏情形。
2. **per-family 子注册表**：`ModelConfig` 把 family-专属键收进 family-scoped 子块（如 `model.wan` / `model.diffusers_offload`），schema 层即可拒绝跨 family 错放。最干净但 churn 大（动所有 wan config 的键路径）。
3. **仅注释钉死**：给 family-专属键加 `# Wan 2.2 dual-expert only` / `# diffusers pipeline only` 注释（最低成本，不改行为，只降认知坑）。

## 落地方案

### A. 拆 `task_variant` 同名异义（§1.1）

- AR 侧改用独立字段。在 `RuntimeBuildSpec` 增 `ar_task: str | None = None`（`vrl/models/interfaces/runtime.py:59` 邻位），把 `task_variant` 留给 diffusion 的 t2v/i2v 轴。
- 同步迁移 AR 写入点：`vrl/models/ar/janus_pro/runtime.py:118-119`（`task_variant="ar_t2i"` → `ar_task="ar_t2i"`）、`vrl/scripts/ar/janus_pro/train.py:58, 76`（`spec.task_variant = "ar_t2i_r1"` → `spec.ar_task`）、replay 分支 `vrl/models/ar/janus_pro/runtime.py:83`（`spec.task_variant == "ar_t2i_r1"` → `spec.ar_task`）。
- bundle metadata 里 AR 写的 `"task_variant": spec.task_variant`（`runtime.py:69, 98`）随之改键，nextstep_1 同形点一并迁移。
- 注意：`task_variant` 是 Ray 序列化的 launch 契约一部分，新增字段须保证序列化往返；改名须 AR 三处（spec 写、train 覆写、replay 读）+ metadata writer 同步落地。

### B. 收敛 `family` 取值空间（§1.2）

- 保持 `cfg.model.family` 为唯一用户面选择器（YAML 兼容，不改键名），它已被 schema capability gating 与 Wan i2v 探测消费。
- 把 model 类属性 `.family`（`wan_2_1/model.py:569` 等）/ `model_family`（`janus_pro/model.py:155`）按其真实角色——"display/metric label"——更名为 `model_label`，并在定义处注释为 display/provenance-only（呼应 AGENTS.md derived-struct 规则）。统一所有 diffusion / AR family 同步改，保持跨 family 形状一致。
- 落地后只剩两个 family 概念：用户面 `cfg.model.family` 选择器 + 派生的 `model_label` 展示标签。

### C. 删 Wan `model.task` 死读（§1.3）

- 删 `vrl/models/diffusion/wan_2_1/runtime.py:351-353` 的 `task = cfg_get(cfg.model, "task", None)` 分支，只留 `task_variant` 读 + family 派生默认。无配置/无 schema/无测试引用，直接删是 clean fix。

### D. 删 `model_family` context 死写（§1.4）

- 从所有 family 的 export/rollout context 删 `"model_family"` 键：`predict2/model.py:395`、`wan_2_1/model.py:440`、`janus_pro/runtime.py:398, 520`，以及其余 diffusion family 同形点（grep `"model_family":` 全删）。
- `tests/rollouts/collector/test_batch.py:108-112` 改用一个中性键（如 `guidance_scale` 已有，可换任意非身份键）测 context 透传，避免再引用已删的生产键名。若未来需要该字段，读时从 `model_label` 派生，不再每家重写。

### E. 修活 checkpoint family 守卫（§1.5）

- 选定"修活"而非"删守卫"：让 family builders 在 `RuntimeBundle.metadata` 写 `"family"`，值取 `cfg.model.family` 取值空间（与 checkpoint payload 同空间，守卫才有意义）。改 `vrl/models/ar/janus_pro/runtime.py:67-72, 95-100` 及各 diffusion family 的 metadata dict，统一经 `vrl/models/replay_loading.py` 的共享 metadata 构造器加该键，保证跨 family 一致。
- 注意：payload 侧是 `normalize_rollout_family(cfg.model.family)`，bundle 侧须写同一规整后的值，否则把守卫从"恒不触发"变成"恒误报"。spec 已携带 family 信息或可由 builder 注入。

### F. 给 "backend" 划边界（§1.6，不含 backend_label）

- `RuntimeBundle.backend_handle` → `raw_handle`（或 `pipeline_handle`）。同步改：interface 定义 `vrl/models/interfaces/runtime.py:162` + docstring `:133`、7 个 family runtime 的写入、`train_dpo.py:150` 读取、wiring 测试断言。
- metric 键去歧义：`ar_decoder.py:97` 的 `"backend"`（per-family label）与 `vllm_paged.py:279` 的 `"backend"`（kernel 类串）改用不同键，如 `"attention_backend"` vs `"attention_kernels"`，避免一个键扛两套取值空间；`torch_attention.py:70, 82` 同步。
- `RuntimeBuildSpec` docstring 里"backend classes"措辞改"family model classes"（`vrl/models/interfaces/runtime.py:4` 区域）。
- 保留 `attention_backend` 作为 kernel 选择器名（对齐 vLLM/SGLang 用法），不改。
- 注意：metric 键可能被外部 dashboard 抓取——改键名属语义破坏，须在 sprint 内确认无内部 dashboard 依赖；若有，记一行 migration 注释。

### G. ModelConfig 家族键分域（§1.7-1.8，先拍板再动）

- 已落地：采用 family-aware schema，而不是 loader-local fail-loud。`RootConfig.model`
  的 unknown-key block 通过 `model.family` 选择 family-specific key set；静态 lint
  通过 variants 识别 family-owned code reads。
- 已落地：`boundary_ratio` / `trainable_transformers` 只在 Wan schema 合法；
  `freeze_vae` 只在 NextStep schema 合法；`reference_image` 只在 Wan/Cosmos Predict2
  schema 合法。
- 已落地：两个 Wan CPU offload bool 合并为 `offload_mode` enum；旧 bool 在 Wan loader
  层 fail loud，避免绕过 schema 时变成 no-op。
- 非目标：不拆 per-checkpoint config class；不把 family runtime 逻辑搬进 schema；
  `RuntimeBuildSpec.model_config` 继续作为 plain dict 传给 family runtime。

## 验证（finishing criteria）

- `pytest tests/models/ tests/nn/ tests/rollouts/collector/test_batch.py tests/generation/ray/ -q` 全绿（含改写后的 context 透传测试、backend_handle wiring 测试）。
- 配置解析：对每个改动 family 跑一次 `cfg.model.family` 选择 + bundle 构建，断言 `RuntimeBundle.metadata["family"]` 现在非空且等于 `normalize_rollout_family(cfg.model.family)`（E 修活的直接证据）。
- 新增一条 resume 守卫回归测试：构造 checkpoint payload family="janus_pro" + bundle metadata family="wan"，`restore_training_checkpoint(strict=True)` 须 raise `ValueError`（证明守卫从"恒不触发"变成"真触发"）。
- grep 净化：`grep -rn '"model_family"' vrl/` 在生产代码零命中；`grep -rn 'cfg.model, "task"' vrl/` 零命中；`grep -rn 'backend_handle' vrl/` 零命中（全部改名）。
- AR 路径冒烟：janus_pro 与 janus_pro_r1 各跑一次 spec 构建，断言 `ar_task` 取值正确分流 capability（`JANUS_PRO_R1_FAMILY_CAPABILITY` vs `JANUS_PRO_FAMILY_CAPABILITY`）。

## 非目标 / Non-Goals

- 不碰 `backend_label` 的派生默认重抄——[[SPRINT_design_smell_audit]] 已复核撤回不做（消费者 fallback 非 dead，stub 测试走该路径）。
- 不改 `cfg.model.family` 的 YAML 键名（用户面契约稳定）。
- 不动 `attention_backend` kernel 选择器命名（已对齐 vLLM/SGLang）。
- 不重构 reward / algorithm / trainer 算法逻辑，不 flatten `FAMILY_REGISTRY` 等跨 family 注册表（护栏要保的并行结构）。
- 不引入精度命名相关改动——见 [[SPRINT_precision_naming_unification]]（mixed_precision/bf16/'no' 三拼法已单列）。

## References

- `vrl/models/interfaces/runtime.py:4, 59, 133, 162`（RuntimeBuildSpec.task_variant、RuntimeBundle.backend_handle、metadata 契约）
- `vrl/models/diffusion/wan_2_1/runtime.py:347-368`（_task_variant_from_cfg 死读 + i2v 规整）
- `vrl/models/diffusion/wan_2_1/model.py:440, 569`（.family 类属性 + model_family context 写）
- `vrl/models/diffusion/cosmos/predict2/model.py:395`（model_family context 写）
- `vrl/models/ar/janus_pro/model.py:155`（model_family 类属性）
- `vrl/models/ar/janus_pro/runtime.py:67-72, 81-85, 95-100, 118-119, 398, 520`（metadata 无 family、ar_t2i 写入、ar_t2i_r1 分支、model_family context 写）
- `vrl/scripts/ar/janus_pro/train.py:58, 76`（spec.task_variant = "ar_t2i_r1"）
- `vrl/config/schema.py:288-294, 313, 317, 530, 538-539`（ModelConfig 扁平 key 注册表 + family-专属键、task_variant 有/task 无、family capability gating）
- `vrl/models/diffusion/wan_2_1/runtime.py:158-181`、`model.py:101-113, 603-606`、`vrl/scripts/eval/wan_i2v_base_sample.py:112`（boundary_ratio / trainable_transformers / *_cpu_offload 真实消费面，均 Wan 专属）
- `vrl/trainers/checkpointing.py:122, 249-255`（payload family 写 + 死守卫读）
- `vrl/scripts/common/online.py:563`、`vrl/scripts/common/factory.py:62`（stack.family = normalize_rollout_family(cfg.model.family)）
- `vrl/rollouts/families/registry.py:312`（normalize_rollout_family 取值空间）
- `vrl/nn/modules/ar_attention_backends.py:34, 84, 104`（attention_backend 选择器 + backend_label 派生）
- `vrl/nn/modules/ar_decoder.py:62, 97`（backend metric 键 = per-family label）
- `vrl/nn/modules/torch_attention.py:60, 70, 82`（backend metric 键，torch_native 变体）
- `vrl/nn/kernels/attention/vllm_paged.py:279`（backend metric 键 = 固定 kernel 类串）
- `vrl/nn/layers/attention/paged.py:22-38`（ARAttentionConfig 字段）
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py:150`（bundle.backend_handle 读取）
- `tests/rollouts/collector/test_batch.py:108-112`（model_family context 仅通用透传测试）
- `docs/sprints/done/SPRINT_design_smell_audit.md:175-179, 133`（backend_label 撤回不做 + ARAttentionConfig.dtype/device 已删）

相关 sprint：[[SPRINT_design_smell_audit]]、[[SPRINT_config_string_settings]]、[[SPRINT_precision_naming_unification]]、[[SPRINT_resolved_struct_field_audit]]
