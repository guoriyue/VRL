# SPRINT：Plug-and-play rollout optimization layer

状态：**done（2026-08-16）**。P0–P4 全部落地，`make verify` 绿。

执行中有三处证据推翻了计划里的假设，已按证据修正（见 §8 执行记录）：
TeaCache 不能做成 build-time pass；drift guard 对量化已自动生效，真正的缺口
只在 TeaCache；magi_1 需要显式声明"无 policy core"。

## 0. 结论先行

VRL 的 rollout 优化**内核层已经是 model-agnostic 的**（quantization 遍历
`nn.Linear`、TeaCache 用 latents 的 relative-L1），不 plug-and-play 的是**装配层**：
每个优化都要模型基类自己写一个方法、loader 再按 format 字符串分发一次。

本 sprint 不重写任何 kernel。它做一件事：**把"哪里是 policy core"从 N 个方法名
里抽出来，变成一个声明**，让优化层自己完成挂载。

三个可验证的产出：

1. 新增一个量化方案，模型侧零改动（今天要改 3 处）。
2. 新增一个 family，量化/compile 自动生效，不可能"忘记接线"。
3. **修掉一个已确认的静默缺陷**：Wan 双专家只有一半被量化（见 §1.3）。

第 3 项本身就足以证明当前设计的代价 —— 这不是假想的可维护性问题，是已经在
生产路径上的正确性问题。

## 1. 当前代码现实

### 1.1 内核层：已经 model-agnostic

`vrl/nn/quantization/base.py:89-102` 的 `swap_linears` 对模型结构一无所知：

```python
for parent_path, parent in root.named_modules():
    for child_name, child in list(parent.named_children()):
        if not isinstance(child, nn.Linear):
            continue
        path = f"{parent_path}.{child_name}" if parent_path else child_name
        if any(token in path for token in exclude):
            continue
        if not matches_linear_target(path, profile):
            continue
        ...
        setattr(parent, child_name, cls(child, **init_kwargs))
```

它只需要三个输入：**root module**、**exclude 元组**、**target profile**。
TeaCache 同理（`vrl/generation/steps/denoise/teacache.py`），信号是 latents 的
relative-L1，不认识任何 family。

**这一层不需要改。** 它已经是本 sprint 想要的形态。

### 1.2 装配层：身份被写进了方法名

模型基类各写一份，唯一的差异是 root 和 exclude：

```python
# vrl/models/steps/denoise/base.py:403-418
def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
    from vrl.nn.quantization import Fp8Linear
    return Fp8Linear.swap_linears(self.transformer, recipe=recipe)

# vrl/models/steps/token/base.py:105-121
def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
    from vrl.nn.quantization import LM_EXCLUDE, Fp8Linear
    return Fp8Linear.swap_linears(
        self.language_model, recipe=recipe, exclude=LM_EXCLUDE,
    )
```

然后 loader 再按字符串分发一次（`vrl/models/loader.py:154-159`）：

```python
if format_name == "fp8":
    swapped = model.quantize_rollout_fp8(recipe=recipe or "rowwise")
elif format_name == "nvfp4":
    swapped = model.quantize_rollout_nvfp4()
else:
    raise NotImplementedError(...)
```

代价：**新增一个 scheme 要改 3 处**（scheme 类 + 2 个基类的方法 + loader 分支），
其中只有第 1 处是新信息。这正是 AGENTS.md「一个构造点」规则要消除的形态 ——
`self.transformer` / `self.language_model` + `LM_EXCLUDE` 这三个事实被复制进了
每个 `quantize_rollout_*` 方法，而它们本该只声明一次。

### 1.3 已确认缺陷：Wan 双专家只量化一半

Wan 有两个 transformer：

```python
# vrl/models/families/wan_2_1/model.py:219-223
def _wan_transformers(self) -> dict[str, Any]:
    modules = {"transformer": self.transformer}
    if self.transformer_2 is not None:
        modules["transformer_2"] = self.transformer_2
    return modules
```

Wan **重写了** `torch_compile_transformer` 来编译两个
（`wan_2_1/model.py:323-328`，遍历 `trainable_modules`），但
**没有重写** `quantize_rollout_fp8` / `quantize_rollout_nvfp4`
（grep 确认：`vrl/` 下只有 2 处 `def quantize_rollout`，都在基类）。

因此 Wan 走继承的基类实现，只量化 `self.transformer`，`transformer_2`
静默保持 base dtype。而 loader 的兜底守卫也发现不了 —— 它只检查
「count == 0」（`vrl/models/loader.py:213-227`），量化了一半照样通过。

**这是"每个模型自己写一遍装配"必然产生的失败模式**：compile 的作者记得处理
双专家，quantization 的作者没有，而类型系统和守卫都不会提醒。

后果不只是慢一半：dual-stage 采样中两个专家走不同精度，rollout-vs-replay
的 drift 在 boundary 前后不一致，drift guard 的假设被破坏。

### 1.4 兜底守卫是设计缺陷的信号

`vrl/models/loader.py:229-238` 专门写了个 runtime guard 防止 family 作者忘记接线：

> "The family runtime builder did not apply the rollout quantization swap
> (e.g. apply_rollout_quantization), so {format_name} would silently run at
> the rollout base dtype."

**需要一个运行时守卫来防止忘记接线，本身说明接线方式有问题。** 正确的形态是
结构上不可能忘记。本 sprint 完成后这个守卫可以删掉（§4 P4）。

### 1.5 装配顺序是真实约束，不能丢

`vrl/models/steps/denoise/build.py:52-74` 的顺序有真实理由，重构必须保留：

```
LoRA attach  →  quantization swap  →  device move  →  torch.compile  →  offload hooks
```

- LoRA 必须在量化前：PEFT 只能包 plain `nn.Linear`（`build.py:52-54`）
- 量化必须在 compile 前：inductor 要看到最终的 quantized module
  （`denoise/base.py:413`）
- offload hooks 必须最后：Accelerate 要看到最终 module tree（`build.py:76-78`）
- blockwise fp8 与 compile 互斥：vLLM triton kernel graph-break，
  compiled forward 慢 ~10x（`loader.py:127-146`）

这些约束属于**装配层**，重构后应该集中在一处表达，而不是散在两个 build 文件里。

## 2. 目标形态

```
config  (precision.rollout.quantization / torch_compile / teacache)
  │
  ▼
OptimizationPass 注册表          ← 新增 scheme 只在这里加一项
  │   quantization / compile / teacache
  ▼
apply_rollout_optimizations(model, build)   ← 唯一装配点，拥有顺序约束
  │
  ▼
PolicyCore 声明                  ← family 只回答：我的 policy root 有哪些、排除什么
  │   roots: dict[str, nn.Module]
  │   exclude: tuple[str, ...]
  ▼
swap_linears / torch.compile / TeaCacheState   ← 已有内核，零改动
```

**family 只声明两件事，不再拥有任何优化方法。**

## 3. 设计决定

### 3.1 声明用 `policy_cores`（复数），不是 `policy_core`

Wan 的双专家证明单数是错的。声明形态：

```python
# vrl/models/steps/denoise/base.py（基类默认）
@property
def policy_cores(self) -> dict[str, nn.Module]:
    """优化 pass 作用的 policy root。键名进日志/错误信息。"""
    return {"transformer": self._require_transformer()}

quantization_exclude: tuple[str, ...] = DEFAULT_EXCLUDE
```

```python
# vrl/models/steps/token/base.py
@property
def policy_cores(self) -> dict[str, nn.Module]:
    return {"language_model": self.language_model}

quantization_exclude: tuple[str, ...] = LM_EXCLUDE
```

Wan 只需重写 `policy_cores` 返回 `self._wan_transformers()` —— **一处声明同时
修好量化和 compile**，且以后任何新 pass 自动覆盖双专家。这是 §1.3 缺陷的根治，
不是补丁。

**为什么不直接复用 `trainable_modules`**：语义不同。`trainable_modules` 是
「trainer 优化 + weight sync 的目标」，Wan 会按 `_trainable_transformer_names`
过滤（`wan_2_1/model.py:331-333`）—— 一个只训练 `transformer` 的配置下
`trainable_modules` 不含 `transformer_2`，但 rollout 时**两个专家都要跑、
都该量化**。混用会重新引入 §1.3 的 bug，方向相反。两个概念必须分开命名。

按 AGENTS.md Rule 3：`policy_cores` 是基类默认（denoise/token 各一个默认值），
family 忘记重写的后果是「少优化一个 root」，不是崩溃 —— 所以用 base default，
不用 abstract method。但 §4 P3 的一致性测试会抓出「有多个 transformer 属性却
没重写 `policy_cores`」的 family。

### 3.2 `OptimizationPass` 用协议，不用继承

```python
# vrl/nn/optimization/passes.py（新增）
class OptimizationPass(Protocol):
    name: str
    def enabled(self, build: ModelBuild) -> bool: ...
    def apply(self, model, build) -> PassResult: ...
```

按 AGENTS.md：consumer-facing `Protocol` 是结构契约，concrete pass 结构化满足即可，
不继承。

> **执行修正**：计划原本要做三个 pass（含 `TeaCachePass`）。读代码后不成立 ——
> TeaCache 是 request-scoped 且不碰 module tree，没有 build-time seam
> （证据见 §8.1 ①）。最终只有两个 build-time pass：`QuantizationPass`、
> `CompilePass`；TeaCache 通过 `REQUEST_SCOPED_DRIFT_SOURCES` 参与同一套
> drift 账目，保住了「drift 来源单一可枚举」这个真正目标。

### 3.3 scheme 注册表取代 if/elif

```python
# vrl/nn/quantization/__init__.py
QUANTIZATION_SCHEMES: dict[str, type[QuantizedLinear]] = {
    cls.quantization_scheme: cls for cls in (Fp8Linear, Fp4Linear)
}
```

从 `quantization_scheme` 类属性派生，不手写 —— AGENTS.md 明令禁止手维护
重复 typed structure 的 ALL_CAPS 常量。新增 int8 只需把类加进这个元组。

`loader.py` 的 `if format_name == "fp8" / elif nvfp4 / else raise` 整段删除。

### 3.4 RL 特有约束：drift 来源必须可枚举

这是 VRL 与 sglang-omni 的根本差异，plug-and-play 不能牺牲它。
每个 pass 声明自己是否引入 rollout-vs-replay 数值偏差：

```python
@dataclass(frozen=True, slots=True)
class PassResult:
    name: str
    applied: bool
    detail: str                      # 进日志
    introduces_replay_drift: bool    # RL 语义，非日志
    touched: tuple[str, ...] = ()    # 被修改的 policy core 键名
```

`introduces_replay_drift` 必须有非日志消费者，否则按 AGENTS.md 就是 dead field。
它的消费者是 `OptimizationReport.drift_sources`。

`touched` 的消费者是 P3 的一致性断言（多 root family 必须每个 root 都被 pass 覆盖），
这正是 §1.3 缺陷的自动化捕获。

> **执行修正**：计划原本要对「所有 drift 源」做前置检查。读代码后发现**量化那半
> 已经有了** —— rollout 与 training precision 不同会让 `stages_match` 变 False，
> trainer 自动装上 TIS correction + `mode="auto"` 的 drift guard（解析成
> `"fail"`）。真正的缺口只在 TeaCache：它改 `noise_pred` 而不改 precision label，
> 所有自动纠正全部不生效（证据见 §8.1 ②）。因此 `validate_guarded_rollout_drift`
> 只拦 request-scoped 那一类，并保留与 precision-split 相同的 expert 逃生舱。

### 3.5 命名

按 AGENTS.md「用已有系统的名字」：`pass` / `apply` 是编译器与
torch.fx / inductor 的既有词汇，读者能直接映射。不发明 `optimizer`
（与 `torch.optim` 冲突）、不用 `manager` / `handler` 这类装饰性后缀。

模块放 `vrl/nn/optimization/` —— 与 `vrl/nn/quantization/` 平级，
不放 `vrl/utils/`（那里保留给真正跨包的原语）。

## 4. 执行顺序

每一段的退出条件是下一段的入口条件。P0 独立可发布，P1-P4 依次依赖。

### P0 — 修 Wan 双专家量化缺陷（独立，先发）

**为什么先做**：这是已在生产路径上的正确性问题，不该等重构。
且它给后面的重构提供了回归基线。

- Wan 模型加 `quantize_rollout_fp8` / `quantize_rollout_nvfp4` 重写，
  遍历 `_wan_transformers()`。丑，但**故意是临时形态** —— P2 会把它替换成
  一行 `policy_cores` 声明并删掉这两个方法。
- 测试：dual-stage Wan（`boundary_ratio` 非 None）量化后，
  两个 transformer 下都有 `QuantizedLinear`。
- 退出条件：新测试在 P0 前 fail、P0 后 pass。

**这一步的临时代码必须在 P2 删除**，否则它就变成 §1.2 的又一个副本。
在方法上写明 `# TODO(P2): replaced by policy_cores declaration` 并在 P2 的
checklist 里点名删除。

### P1 — 内核层：scheme 注册表

- 新增 `QUANTIZATION_SCHEMES`（派生，§3.3）。
- `loader.py` 的 if/elif 分发改读注册表。
- 模型基类的 `quantize_rollout_*` 方法**暂不动**。
- 退出条件：`tests/nn/quantization/` 全绿；
  `test_apply_rollout_quantization_dispatches_by_scheme` 与
  `test_apply_rollout_quantization_dispatches_nvfp4` 不改断言即通过。

这一步纯内部替换，无行为变化，是安全的第一刀。

### P2 — 声明层：`policy_cores`

- denoise/token 基类各加 `policy_cores` + `quantization_exclude`（§3.1）。
- Wan 重写 `policy_cores` 返回 `_wan_transformers()`，
  **删除 P0 的临时方法 + 删除它的 `torch_compile_transformer` 重写**
  （后者由通用 compile pass 遍历 `policy_cores` 覆盖）。
- 删除两个基类的 4 个 `quantize_rollout_*` 方法。
- `apply_rollout_quantization` 改为遍历 `model.policy_cores`，
  用 `model.quantization_exclude`，直接调 `swap_linears`。
- 退出条件：P0 的 Wan 测试仍绿（现在由声明而非重写保证）；
  `tests/models/steps/token/test_rollout_quantization.py` 全绿。

**风险**：`assert_rollout_quantization_applied` 里的
`getattr(model, "transformer") or getattr(model, "language_model")` 探测
（`loader.py:203-208`）要同步改成读 `policy_cores`，否则守卫与装配读不同的
真值来源。这是本 sprint 最容易漏的一处。

### P3 — 装配层：pass 注册表 + 单一装配点

- 新增 `vrl/nn/optimization/passes.py`：`OptimizationPass` 协议 +
  `PassResult` + 三个 concrete pass（§3.2）。
- 新增 `apply_rollout_optimizations(model, build) -> tuple[PassResult, ...]`，
  §1.5 的顺序约束和 blockwise×compile 互斥检查**集中在这里**。
- `denoise/build.py` 与 `token/build.py` 各自的优化调用替换为一次调用。
  注意 denoise 的 LoRA 分支（`build.py:55-69`）里量化位置不同 ——
  装配点必须接受「LoRA 已附加」作为前置状态，不能把 LoRA 也吞进去
  （LoRA 不是优化 pass，它是训练配置）。
- 新增一致性测试：对每个注册 family 的 `policy_cores`，
  断言 pass 的 `touched` 覆盖全部 root。**这条测试是 §1.3 缺陷的自动化捕获**，
  以后任何多 root family 漏配都会红。
- 退出条件：`make verify` 全绿；新一致性测试覆盖全部 registry family。

### P4 — 删守卫 + drift guard 前置检查

- 删除 `assert_rollout_quantization_applied` 及其在
  `vrl/generation/execution/worker.py:727-729` 的调用
  —— 结构上不再可能忘记接线（§1.4）。
  **前提**：P3 的一致性测试已覆盖全部 family，否则不许删。
- 新增：`introduces_replay_drift=True` 的 pass 生效但 drift guard 未 arm 时
  fail loud（§3.4）。这把 TeaCache docstring 里的文字约定变成代码约束。
- 退出条件：`tests/quality/preview.py:140` 与
  `tests/models/steps/token/test_rollout_quantization.py` 中依赖该守卫的
  用例改为断言新行为；`make verify` 全绿。

## 5. 非目标

明确不做，避免 scope 蔓延：

- **不碰 kernel**。`swap_linears`、fp8/fp4 数学、TeaCache 的
  `rel_l1` 一行不改。本 sprint 是装配层重构。
- **不加新优化技术**（CUDA graph、continuous batching、paged KV 扩展）。
  加不加是 profile 决定的独立问题；本 sprint 只让「加」这件事变便宜。
- **不动 `vrl/nn/kernels/attention/vllm_paged.py`**。它明确不是
  model-agnostic 的，且自己在 docstring 里说明了原因（family runner 仍需
  patch 自己的 attention 层）。把它 model-agnostic 化是 native transformer
  executor sprint 的范围，那个 sprint 是 profile-gated 且当前 parked。
- **不引入外部 engine**（SGLang / vLLM 作为顶层）。
  `SPRINT_native_generation_engine_program.md` §1 已论证替换成本不对称，
  本 sprint 不重开该结论。
- **不追求与 serving 框架的吞吐可比性**。RL rollout 的约束是
  rollout-vs-replay 数值一致性，serving 框架没有这个约束，
  两者的优化空间不可比。

## 6. 验证

每段结束跑：

```bash
make verify
```

Ruff 只跑改动文件（AGENTS.md）：

```bash
ruff check --fix <files> && ruff format <files>
ruff check <files> && ruff format --check <files>
```

GPU 侧回归（P2、P3 结束各一次，需真实 CUDA）：

```bash
# 量化 A/B 未回归
python -m vrl.scripts.perf.quantized_linear_benchmark
# rollout-vs-replay drift 未变大
python -m vrl.scripts.perf.quantized_rollout_drift_probe
```

**P0 的 Wan 测试是全程的回归基线** —— 它在 P2 从「重写方法保证」变成
「声明保证」，断言不变。断言不变而实现改变，正是重构正确的证据。

## 7. 完成后的收益

| | 今天 | 完成后 |
|---|---|---|
| 新增量化 scheme | 改 3 处（scheme + 2 基类 + loader 分支） | 改 1 处（scheme 类加进元组） |
| 新增 family | 手工接线，靠 runtime 守卫兜底 | 声明 `policy_cores`，pass 自动挂载 |
| 多 root family | 每个优化各自记得遍历（compile 记得、量化忘了） | 一处声明，全部 pass 覆盖 |
| drift 来源 | 散在量化 config 与 sampling config 两处 | 单一可枚举，未 arm guard 直接 fail |
| 「忘记接线」 | runtime 守卫 + 只查 count==0（半量化漏网） | 结构上不可能 + 一致性测试 |

## 8. 执行记录（2026-08-16）

### 8.1 计划被证据修正的三处

**① TeaCache 不能做成 build-time pass（§3.2 的设想错了）**

计划想把 TeaCache 和量化/compile 一起放进 pass 注册表。读代码后不成立：

```python
teacache=TeaCacheConfig.from_sampling(sampling.get("teacache")),
```
`vrl/generation/bindings/full_sequence_denoise/layout.py:138`

TeaCache 在 **每个 sampling request** 解析，在 denoise loop 内部跳过 forward，
**不碰 module tree**，因此没有 build-time seam 可以安装。强行塞进
module-mutation 注册表会是"为了统一而统一"。

实际做法：pass 注册表只收 build-time 的两个（量化、compile）；TeaCache 通过
`REQUEST_SCOPED_DRIFT_SOURCES` 声明它的 drift，与 pass 共享同一套 drift 账目。
这保住了计划的真正目标（drift 来源单一可枚举），没有强行统一两种不同生命周期。

**② drift guard 对量化已自动生效，真正的缺口只在 TeaCache（§3.4 需要收窄）**

计划说"开了 drift 源却没 arm drift guard 直接 fail"。读代码后发现量化那半
已经有了：

```python
if not precision.stages_match:
    correction, guard = build_precision_split_safety_configs()
```
`vrl/trainers/online/config.py:333-335`

`PrecisionDriftGuardConfig.mode` 默认 `"auto"`，`resolve_guard_mode` 在
rollout≠training precision 时解析成 `"fail"`。量化会改 rollout precision label，
所以它**必然**翻转这个开关 —— 再加一道检查是重复。

真正的缺口是 TeaCache：它改的是 `noise_pred` 而不是 precision label，
`stages_match` 保持 True，所有自动纠正**全部不生效**。grep 确认
`vrl/trainers/`、`vrl/config/`、`vrl/algorithms/` 里没有任何 `teacache` 引用。

实际做法：`validate_guarded_rollout_drift` 只拦 request-scoped 的那一类，并保留
与 precision-split 路径相同的 expert 逃生舱（显式 `trainer.precision_*` 块）。

**③ magi_1 需要显式声明"无 policy core"**

新增的跨 family 一致性测试立刻抓出 `magi_1` 没有 `policy_cores`。读代码后确认
这是**正确的**——它是 subprocess facade，权重只存在于 MAGI CLI 的子进程里
（`vrl/models/families/magi_1/model.py:236-240`）。但"碰巧没有"和"明确声明没有"
不同：前者会让一个请求量化的 magi 配置静默无效。

实际做法：给它显式的空 `policy_cores` + 注释说明原因，这样请求量化时会在
`apply_rollout_optimizations` 里 fail loud。

### 8.2 与计划的其他偏差

- **P0 已在 `8b328fc5` 提交**（本次会话前完成），P2 按计划删掉了它的临时方法，
  断言不变 —— 这正是重构正确的证据。
- **device move seam**：计划没预见到 compile 必须在 device move **之后**
  （`apply_full_finetune` 的 `requires_grad_`/`.to()` 不能作用在 compiled
  wrapper 上）。用 `before_compile` 回调表达这个 seam，而不是给装配点加一个
  `stage` 字符串参数（后者是"参数只用来标识调用者"的反模式）。
- **`assert_rollout_quantization_applied` 已删除**（P4）。前提验证过：只有
  `magi_1` 有自定义 rollout builder 且它无 policy core，其余 24 个 family 全部
  经过唯一装配点。
- **未采纳 scheme 类上的 `takes_recipe`/`default_recipe`**：recipe 词汇表已由
  `_QUANTIZATION_FORMAT_RULES`（`vrl/config/precision.py:45-52`）拥有，且
  `QuantizationPolicy` 已完成归一化。装配点直接读 `recipe is not None`，
  不新建第二张表。

### 8.3 验证

```
make verify  →  ruff / config lint / pytest 全绿
pytest -m "not e2e and not slow_test"  →  4022 passed
tests/nn/optimization/  →  63 passed（覆盖全部 25 个 family）
tests/config/test_rollout_drift_guard.py  →  9 passed
```

两个已知无关失败（不在本 sprint 范围）：
`tests/architecture/test_generation_rollout_boundaries.py` 的两个 reward 测试，
来自未提交的 animereward 工作（registry 注册了 `animereward_quality`，文件名是
`animereward.py`）。`tests/generation/execution/test_batch_memory_shadow.py`
有一个 random-order flake（`fake_cuda` fixture），连跑 4 次全绿。

## 9. 后续：trainer 侧的另一半（2026-08-16）

上面的 sprint 只处理了 rollout 侧。同一个概念在 trainer 侧还有一份，形态更差。

### 9.1 字符串拼私有方法名

```python
writer = getattr(model, f"_set_{name}", None)
```
`vrl/trainers/strategy.py`（改前）

FSDP/DDP 包装完 module 后要写回，写回口靠**拼私有方法名**找。代价：Wan 必须
额外保留 `_set_transformer` / `_set_transformer_2`，且 rollout / replay 两组类
各写一遍，方法体全是一行赋值 —— 存在的唯一理由是让字符串拼接能解析。

而 Wan **已经有**正确形状的写回口 `_set_wan_transformer(name, transformer)`，
只是 `strategy.py` 用不了。

### 9.2 做法：一个方法

```python
def set_module_root(self, name: str, module: Any) -> None:
```
`vrl/models/steps/denoise/base.py`

Wan 只重写这一处，rollout compile 与 FSDP/DDP 同时被服务。

**一次走错的设计（已改正）**：中途曾在这个方法上再包两个 typed 入口
（`set_policy_core` / `set_trainable_root`），理由是「rollout 和 trainer 的
key 集合不同」。这是错的 —— 两个包装函数体完全相同，既不收窄类型也不校验，
纯粹是 pass-through。**集合不同的是 `policy_cores` / `trainable_modules`
这两个 property，不是写回操作本身**：把一个 root 写回去，跟是谁请求的无关。
一个概念一个方法，已删掉那两层。

两个 property 的区别仍然要钉住（Wan 的 dual-stage 只训一个、采样两个，
合并会静默收窄 rollout 覆盖），由
`tests/models/steps/denoise/test_module_root_writeback.py::test_the_two_views_are_not_the_same_key_set`
负责。

### 9.3 顺带修掉的潜伏债

```python
transformer = getattr(model, "transformer", None)   # 改前
```
`vrl/generation/execution/worker.py`（sequence-parallel 安装）

和 §1.3 完全同型的写法。今天不出错（只有 SD3.5 声明 installer，单 transformer），
但多 root 家族接上后只会装一半且不报错。改为遍历 `policy_cores`。

### 9.4 一处逐字重复的校验

`get_args(PipelineOffloadMode)` 的词汇表校验在两处逐字重复：
`vrl/models/interfaces/runtime.py` 的 `__post_init__`，以及
`vrl/generation/execution/memory_parking.py` 从 Ray wire mapping 里重新挖字符串
再校验一遍。提取为 `require_pipeline_offload_mode()`，由拥有该字段的类导出，
两处共用。

wire mapping 本身**保留** —— launch contract 必须是可序列化的 primitive，
且它在 typed `ModelBuild` 重建之前就要读这个字段。重复的是校验，不是读取。

### 9.5 查过但明确不动的

| 位置 | 为什么不动 |
|---|---|
| `vrl/rewards/` 8.6k 行 | 已经是本 sprint 的形态：`ClassVar` 能力声明 + registry + 明确类阶梯 |
| registry 的 10+ 处 `isinstance(family_build, ...)` | 收窄到只存在于一个变体的字段，是合法类型收窄 |
| `memory_parking.py` 的 11 处 offload 探测 | `uses_pipeline_cpu_offload` / `pipeline_cpu_offload_healthy` / `reset_pipeline_cpu_offload` **只有 Wan 定义**，任何基类都没有 —— 真接缝，`getattr(..., False)` 默认值对其他家族承重 |
| `memory_parking.py` 的 `move_frozen_components` 探测 | 一次调查曾判定为「死防御」（它在 denoise 基类上是具体方法）。**这个判定是错的**：AR 基类没有它，而 6 个 token family 都会走 worker parking。守卫是承重的 |
| `rewards/runtime.py` 的 `getattr(model, "score_batch")` | 真正的可选协议（批量打分快路径） |

### 9.6 验证

```
pytest -m "not e2e and not slow_test"  →  4061 passed
tests/models/steps/denoise/test_module_root_writeback.py  →  4 passed
ruff check vrl/ tests/  →  All checks passed
```

## 10. 参考

- `vrl/nn/quantization/base.py:59-104` — `swap_linears` 通用遍历（内核层，不改）
- `vrl/nn/quantization/targeting.py` — exclude 与 target profile
- `vrl/models/loader.py:97-238` — 量化装配、if/elif 分发、兜底守卫
- `vrl/models/steps/denoise/base.py:396-431` — denoise 侧 compile/量化方法
- `vrl/models/steps/token/base.py:105-138` — token 侧量化方法
- `vrl/models/steps/denoise/build.py:52-94` — 装配顺序约束
- `vrl/models/steps/token/build.py:103-106` — token 侧装配
- `vrl/models/families/wan_2_1/model.py:219-223,323-328` — 双专家 + compile 重写
- `vrl/models/families/registry.py:60-86` — `GenerationRuntimeCapabilities`（声明先例）
- `vrl/generation/steps/denoise/teacache.py` — TeaCache 与 RL drift 警告
- `vrl/generation/steps/denoise/loop.py:135-161` — TeaCache 生产接线
- `docs/sprints/SPRINT_native_generation_engine_program.md` — engine ownership 边界
