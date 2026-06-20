# SPRINT: 精度命名统一 —— 一个概念三套拼写 + 一个撞名 (planned)

状态：planned（2026-06-20）。范围：收口「compute 用什么精度」这一个概念在仓库里的**三套拼写 + 一个同名异义**，让 `precision:` 成为唯一可填真源，删冗余派生布尔、改撞名的 FSDP key。不动统一 `precision` policy 的语义、不动 fp8/fp4 rollout 劈叉路径。

关联：
- [[SPRINT_fullparam_and_fp8_precision]] —— 统一 `precision` policy（`forward`/`rollout`/`math`/`frozen` 四轴）与 fp8 rollout 劈叉的归属 sprint；本 sprint 是它的命名收尾。
- [[SPRINT_design_smell_audit]] / [[SPRINT_resolved_struct_field_audit]]（均 `done/`）—— 派生/影子字段的处置先例（"派生布尔不该手维护"、"resolved struct 每个字段要有非日志消费方"）。本 sprint 的 `bf16` 冗余判定沿用同一判据。

## 0. Core Decision（先看这一段）

「compute 用什么精度」一个概念，仓库里有三层拼写，加一个倒霉撞名的无关概念：

1. **`precision:`（`fp32`/`bf16`/`fp16`，顶层）** —— 唯一真源、用户唯一该设的。`vrl/config/precision.py`。
2. **`mixed_precision` + `bf16`（`TrainerConfig` 派生字段）** —— 从 #1 自动算出来的内部桥接字段，`metadata yaml="bridged"`（不可在 YAML 填）。`bf16` 是纯冗余布尔（= `compute=="bf16"`）。
3. **`'no'`/`'fp16'`/`'bf16'`（HF Accelerate 边界拼写）** —— `vrl/trainers/precision.py` 的 `normalize_mixed_precision` 把 `fp32`→`'no'`、`float16`→`fp16`，只为喂 HF Accelerate 的 `Accelerator(mixed_precision=...)`（当前仅 Wan DPO 一条路要）。
4. **`distributed...fsdp.mixed_precision: actor|none`（撞名异义）** —— 这根本不是 dtype，是 **FSDP 的参数/梯度规约策略**（`actor` = bf16 参数 + fp32 reduce）。只是不幸跟 #2 撞了同一个词。

核心结论：#1 是对的、保留；#2 的 `bf16` 布尔删掉（派生冗余）；#4 改名消撞；#3 评估是否能随 Wan DPO 收掉。落地后用户面只剩一个 `precision:`，resolved config 里的 `mixed_precision` 是派生输出（已有消费方），不再有第二个并行可填字段。

## 1. 现状实锤（evidence-first）

### 1.1 真源：统一 precision policy

```python
# vrl/config/precision.py
# - forward : generation rollout + trainer replay transformer forward dtype
# - math    : 非 transformer 的敏感算子（SDE step / logprob / loss 规约）
# - rollout : 实验性劈叉（fp8/fp4 rollout vs bf16/fp32 replay）
# - frozen  : frozen text encoder / VAE
_CANONICAL = ("fp32", "bf16", "fp16", "fp8", "fp4")
```
用户写 `precision: bf16`（标量两边同 dtype）或 `precision: {forward: bf16, rollout: fp8}`（劈叉）。

### 1.2 派生桥接字段（bridged，不可填）

```python
# vrl/config/builders.py:180
payload["mixed_precision"] = precision.compute        # canonical: fp32/bf16/fp16
payload["bf16"] = precision.compute == "bf16"         # 冗余布尔
payload["rollout_precision"] = precision.rollout
payload["math_precision"] = precision.math
```
```python
# vrl/trainers/core/types.py:303
mixed_precision: str = field(default="", metadata={"yaml": "bridged"})
bf16: bool = field(default=True, metadata={"yaml": "bridged"})
```
（`vrl/scripts/common/online.py:96-101` 又重做一遍同样四个赋值 —— 两处 expand 逻辑重复，见 §4.3。）

### 1.3 Accelerate 拼写层

```python
# vrl/trainers/precision.py
def normalize_mixed_precision(mixed_precision, *, bf16=False) -> str:  # → "no"|"fp16"|"bf16"
    ...
    aliases = {"fp32": "no", "float32": "no", "float16": "fp16", "bfloat16": "bf16", ...}
```
所以 resolved config 里的 `mixed_precision: 'no'` = fp32。仅 Wan DPO 经 `Accelerator(mixed_precision=...)` 需要这个拼写：

```python
# vrl/scripts/diffusion/wan_2_1/train_dpo.py:213
mixed_precision=mixed_precision,   # HF Accelerate 只认 no/fp16/bf16
```

### 1.4 撞名的 FSDP 策略（无关概念）

```python
# vrl/config/schema.py:380
mixed_precision: Literal["actor", "none"] = "actor"   # actor=bf16 参数+fp32 reduce
```
```yaml
# configs/base/distributed/training_fsdp.yaml:15
mixed_precision: actor      # bf16 params/compute, fp32 grad reduction
```
消费方 `vrl/trainers/fsdp.py:93 mixed_precision_policy(name)`。词同义不同，纯撞名。

## 2. 为什么会这样（来历，非甩锅）

最早直接抄 HF Accelerate 的 `actor.mixed_precision`（`no`/`fp16`/`bf16`）+ `actor.bf16` 布尔 —— 用户看的那份旧 `resolved_config.yaml`（`docs/runs/sd3_5_ocr_grpo_crossnode_continuous_3epochs/`）就是这个年代的产物，所以那俩还挂在 `actor:` 下、用老拼写。后来 [[SPRINT_fullparam_and_fp8_precision]] 统一成 `precision:` 单一真源，但为了不一次性改所有下游消费方（DPO autocast、model dtype helper、FSDP），把旧的 `mixed_precision`/`bf16` 留成派生桥接字段。新旧叠加 → 三套拼写并存。

## 3. 落地方案

### 3.1 删冗余 `bf16` 布尔（架构卫生：派生布尔不手维护）

`bf16` 完全 = `mixed_precision == "bf16"`（亦即 `precision.compute == "bf16"`）。统一路径下 `mixed_precision` 恒有值，`normalize_mixed_precision` 里那条 `return "bf16" if bf16 else "no"`（`precision.py:20`）只在 `mixed_precision` 为空时触发 —— 统一流程下永不命中。

- 删 `TrainerConfig.bf16`（`types.py:304`）及 `builders.py:181` / `online.py:97` 两处赋值。
- `trainers/precision.py` 里 `normalize_mixed_precision(..., bf16=...)`、`trainer_mixed_precision`、`torch_dtype_for_mixed_precision`、`torch_dtype_for_trainer_precision` 去掉 `bf16` 形参，从 `mixed_precision` 单源判定。
- 复核：grep `\.bf16`（属性读）当前**零消费方**（只有赋值），符合 [[SPRINT_resolved_struct_field_audit]]「只被赋值/日志读 = 死字段」判据，删之零行为变更。

### 3.2 FSDP `mixed_precision` 改名消撞

- `distributed...fsdp.mixed_precision: actor|none` → 改名为 `fsdp.precision_policy`（或 `param_reduce_policy`），表达「这是参数/规约策略，不是 dtype」。
- 改 `schema.py:380` 字段名、`fsdp.py:93` 消费方、`configs/base/distributed/training_fsdp.yaml:15`、以及任何引用该 key 的实验 config（grep 全量）。
- 这是性价比最高的一处：消除「同一个词两种 value 词汇（`no/fp16/bf16` vs `actor/none`）」的认知坑。

### 3.3 评估收掉 Accelerate 拼写层（条件性，先判定）

- `'no'/'fp16'/'bf16'` 这套只为 HF Accelerate 边界存在；当前唯一真消费方是 Wan DPO（`train_dpo.py:213` 的 `Accelerator(mixed_precision=...)`）。其余 `torch_dtype_for_trainer_precision` 调用方其实只要 canonical→torch.dtype 映射，不需要 accelerate 拼写。
- 决策点：若 Wan DPO 这条 Accelerate 路仍保留 → `normalize_mixed_precision` 缩小到「仅 DPO/Accelerate 适配器内部」一个局部 helper，不再是全局精度词汇；其余消费方改走 canonical（`fp32/bf16/fp16`→torch.dtype）。若 Wan DPO 计划迁出 Accelerate（见 §6 关联）→ 整层可删。
- 本 sprint **先做判定 + 收窄作用域**，不强行删（避免误伤 DPO）。

### 3.4 去重两处 expand 逻辑

`builders.py:178-183` 与 `scripts/common/online.py:96-101` 各写一遍「policy → 四个派生字段」。抽成一个 `apply_precision_policy(trainer_config, policy)` 单点，消除「改一处忘另一处」的漂移面。

## 4. 验证（finishing criteria）

1. `precision: bf16` / `fp16` / `fp32` 三档 resolve 后 `mixed_precision`/`rollout_precision`/`math_precision` 取值正确；删 `bf16` 后所有 dtype 解析（`torch_dtype_for_trainer_precision`）与改动前逐位一致。
2. FSDP 改名后 `tests/` 全绿；grep 确认无遗留旧 key（`fsdp.mixed_precision`）。
3. Wan DPO 路径 `Accelerator(mixed_precision=...)` 仍拿到合法 `no/fp16/bf16`。
4. `tests/config/` + `tests/trainers/` 全绿；`test_load_all_experiments` 对受影响实验 config 零未知 key。
5. 旧 `resolved_config.yaml` 是历史产物，不作为输入模板 —— 文档注明（§5）。

## 5. 非目标 / Non-Goals

- 不动统一 `precision` policy 的四轴语义与 fp8/fp4 rollout 劈叉路径（那是 [[SPRINT_fullparam_and_fp8_precision]] 的盘子）。
- 不重写 `docs/runs/**/resolved_config.yaml` 历史产物（它们是快照、非配置真源）。
- 不在本 sprint 强删 Accelerate 拼写层（§3.3 仅判定 + 收窄）。
- 不改 frozen/math 轴的行为。

## References

vrl 代码（本轮实际读过）：
- `vrl/config/precision.py:1-150`（统一 policy 真源、`_CANONICAL`、`forward/rollout/math/frozen` 四轴、fp8/fp4 rollout-only）
- `vrl/config/builders.py:178-185`（policy → 四个派生字段 expand）
- `vrl/trainers/core/types.py:303-306`（`mixed_precision`/`bf16` bridged 派生字段定义）
- `vrl/trainers/precision.py:8-77`（`normalize_mixed_precision` accelerate 拼写、`bf16` fallback、`torch_dtype_for_*`）
- `vrl/scripts/common/online.py:96-101`（第二处重复 expand）
- `vrl/config/schema.py:380` + `vrl/trainers/fsdp.py:93` + `configs/base/distributed/training_fsdp.yaml:15`（撞名的 FSDP `mixed_precision: actor|none`）
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py:213`（HF Accelerate `mixed_precision=` 唯一真消费方）
- `docs/runs/sd3_5_ocr_grpo_crossnode_continuous_3epochs/resolved_config.yaml:22-23`（旧产物里的 `mixed_precision: 'no'` / `bf16: false`，触发本 sprint）
