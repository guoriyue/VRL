# SPRINT：Precision parity：敏感参数前向精度与重放一致性

状态：**implementing；先做 dtype/数值审计，再决定是否改生产精度。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与源代码证据

论文 §3.1、§5.3、§6 分别处理量化数值、采样 token 的引擎一致性和 diffusion 敏感参数。
`vrl/trainers/fsdp.py::normalize_fsdp_parameter_dtype` 会把不匹配参数转成统一 dtype，
而 `vrl/trainers/optimizer.py::FP32MasterWeightOptimizer` 保留的是更新精度。
FP32 master 不代表前向中 timestep/modulation/normalization 仍用 FP32。

`vrl/config/precision.py`、`vrl/models/precision.py`、
`vrl/trainers/online/precision_guard.py` 和
`vrl/scripts/perf/quantized_rollout_drift_probe.py` 已有精度及漂移边界，必须复用。
Miles-diffusion 的参数 dtype patch 只接受 torch 2.11.0；本机恰好 2.11.0，
但 VRL 依赖范围更宽。不能直接复制 monkey patch。

## 实施步骤与输出

1. 对 SD3.5 和一个有敏感参数的 Wan/H3 实例记录：
   加载前后、FSDP wrap 后、forward 和 checkpoint recompute 的参数/输入 dtype。
   报告哪些转换来自 family loader，哪些来自 trainer。
2. 固定权重和采样 transition，对比 native rollout 与 replay：
   model prediction、scheduler mean/std、逐步 logprob、ratio deviation、
   clipping fraction、有效样本量和最终梯度。改变 microbatch 大小和样本排列。
3. 如发现可归因差异，先用现有 family loading/forward 边界保留必要 FP32 运算，
   再评估是否可通过公开 FSDP wrap/mixed-precision API 表达。
   仅在公开路径不能表达时，提出版本隔离补丁和明确支持范围。
4. 用同一组权重/输入比较修复前后；记录 FP32 参数比例、HBM、耗时及误差变化。
   不把降低误差自动解释为提高学习质量。
5. 另测复现性：同进程/新进程相同配置重复运行。
   train/rollout parity、rerun determinism、policy staleness 是三份独立结果。

## 验收

- 单 rank 与真实两 rank FSDP 的 forward/recompute dtype 一致，保存/恢复后保持语义。
- 不适用 family 不被强制添加 glob 名单；必要 pattern 必须命中且有实际 tensor 消费者。
- 未支持的 torch/backend 明确拒绝窄补丁；不静默 fallback 或全局改 torch。
- 数值阈值在看结果前按算法、轨迹长度声明；原来的 TIS/guard 不因 dtype 同名自动关闭。
- 一条真实短曲线排除 NaN/梯度失效；长期 reward 收益交给独立重复实验验收。

## 应改／应留／非目标

只改变证据指出的 dtype 转换与相应测试；保留 FP32 reduction、master optimizer、
PrecisionPolicy、family adapter 和现有 correction。模型维度/协议常量保持。
不创建 FamilyTrainingContract，不追求把分支变成表；不宣称所有 family bit-exact。
Miles 对 BF16-train/NVFP4-rollout 的限制是其 recipe 约束，不是对 VRL 的普遍禁令。


## 2026-09-09: Explicit module precision tracing

`vrl.models.precision.ModulePrecisionTrace` records metadata for explicitly selected
modules. Snapshots identify all selected parameters/buffers at caller-labelled
load/wrap boundaries. Forward hooks record direct parameter/buffer state, tensor
inputs/outputs, grad mode and effective CPU/CUDA autocast configuration. Tensor
payloads are never copied to CPU or retained in events. Hooks are removed both on
normal exit and exceptions. This stateful context owns actual hook lifetimes;
it is not a declarative family contract or a new global precision policy.

Example for a caller that already owns a loaded model and representative inputs:

```python
from vrl.models.precision import ModulePrecisionTrace

selected = {name: model.get_submodule(name) for name in exact_module_names}
with ModulePrecisionTrace(selected) as trace:
    trace.snapshot("loaded")
    # A caller wrapping this same model can add snapshot("after-fsdp-wrap").
    trace.phase = "forward"
    output = model(**inputs)
    trace.phase = "backward"
    loss_from(output).backward()
records = trace.events
```

Choose exact module names from the actual family model. Do not infer sensitivity
from a universal norm/timestep glob. Non-reentrant checkpoint recomputation can
run with grad enabled just like the original forward, so the caller explicitly
labels the backward phase. Early-stop recomputation may produce enter events
without exit events. Structured tensor observation covers dict/list/tuple inputs
and outputs; opaque custom objects need their consuming module boundary selected.
Autocast state and boundary tensor dtype are observations, not an assertion about
internal operator/kernel compute dtype. Hooks can change compilation behavior and
performance, so these traces are acceptance diagnostics, not a production default
or an uninstrumented latency benchmark.

Source observations: MiniMax-H3's `_lora_dtype` returns None to preserve the
loader's mixed storage precision during LoRA attachment. FSDP actor normalization
currently converts mismatched floating parameters to the target dtype, whereas
its native policy rejects that conversion. No production casting policy changed
in this commit. The local SD3.5 Medium transformer safetensors header at revision
`b940f670f0eda2d07fbb75229e779da1ad11eb80` contains 909 BF16 tensors. This was read
through `safetensors.safe_open(...).get_slice(key).get_dtype()` without loading
those weights. Header dtype does not establish post-loader or forward precision,
and does not establish whether FP32 would improve numerical accuracy.

Validation: 89 precision, FSDP and architecture tests passed (16 dependency warnings
plus a SWIG deprecation notice). New CPU tests observe real non-reentrant checkpoint
recomputation, mixed FP32/BF16 module parameters and BF16 inputs, actual FSDP
normalization changes, unchanged FP32 buffers, finite backward gradients and hook
cleanup after forward failure. They are not a full SD3.5/H3 forward audit or a
real two-rank FSDP precision-equivalence result. Those traces, fixed-transition
prediction/logprob/gradient comparisons and repeated real short curves remain open.

Existing family loading hooks, FP32 reductions, master-weight optimizer and
correction/guard logic stay unchanged. The context manager is necessary to own
hook state and cleanup; its helpers share tensor-metadata formatting. No new
ALL_CAPS business vocabulary or family support table was introduced.
