# SPRINT：Weight delivery：版本 ACK 之外验证真实参数内容

状态：**implementing；默认训练路径不增加全量 checksum 开销。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与现状

论文 §4.4 的启发是先污染接收端，再验证首次同步，避免同源初始化掩盖漏写。
`vrl/generation/ray/weight_sync.py` 验证每 rank 的安装版本；
`vrl/generation/execution/worker.py::update_weights` 调用实际模型安装方法。
`vrl/trainers/weight_sync.py` 提供不可变 CPU snapshot。
版本正确是必要条件，但不是每个参数写对的充分条件。

复用 `tests/trainers/test_weight_sync.py`、
`tests/generation/ray/test_weight_sync.py` 及 family 的 trainable-state loader。
先审计现有 name/shape completeness 检查，不能另造平行状态 schema。
`vrl/trainers/diagnostics.py::trainable_state_digest` 已支持本地/DTensor 的
driver 状态摘要；优先复用其格式与可比较范围，不发明第二个哈希表示。
driver 步前/步后摘要不能替代接收端安装结果，两端模型布局不同也不能直接比摘要。

## 实施步骤

1. 建立 opt-in startup/acceptance probe，取得 trainer 真实导出与 worker 真实安装后视图；
   family/precision owner 决定比较逻辑，排除项必须列出原因与精确名字。
2. 在隔离的验收进程中，把即将被同步的接收端 trainable 参数置为确定性不同值，
   同步后逐项验证 name、shape、dtype 和内容。禁止污染 live training fleet。
3. 非量化同构路径逐位比较；merge/requantization 路径把 trainer snapshot
   经同一已验证转换后与接收端比较，同时测前向/logprob；禁止一个万能 atol。
4. 将 probe 接入真实一 worker、两 worker 和 version-slot 测试。
   检查全部 rank，不能只看 primary；缺失 slot、混版本和部分成功仍走现有 quarantine。
5. 输出同步 bytes、snapshot/export/install/verify 时间及内容差异报告。
   默认 runtime 不启用逐 tensor 回读，硬件验收/升级/新 family 才运行 probe。

## 故障注入与 DoD

- 漏掉一个 adapter、交换两个同形 tensor、错 base、只更新一个 rank、
  返回正确版本但不写参数：均不能通过。
- 两次安装同一状态得到同一推理结果；slot 淘汰不影响仍在用的版本。
- LoRA merge/requantization 的等价目标明确；冻结 base 不被误当需同步的参数。
- probe 报错不能允许新版本进入 admission；清理后未修改 trainer 参数。
- 真实 Ray worker 实验通过，不以模拟 ACK 的单测替代。

## 应改／应留／非目标

增加真实 verifier 和验收入口；保留原 snapshot、同步锁、all-rank ACK 与失败边界。
薄 Ray worker 方法是 RPC adapter，应保留；checkpoint/protocol 名称是边界常量。
不在此 sprint 添加 NCCL/RDT/delta transport；不重写 state loader 或默认 LoRA 同步。


## 2026-09-09: Opt-in live receiver readback

`RayGenerationWeightSync(..., verify_content=True)` now propagates an acceptance
request to every engine rank. The real Ray worker adapter forwards it to
`GenerationWorkerCore.update_weights`, which checks actual installed parameters
before committing its version ACK. A no-op install raises through the existing
terminal Ray actor boundary; that worker does not advance its policy version.

Native diffusion and token bases expose readback in their existing module/key
namespace. Multi-root verification uses the family's trainable roots, so selected
Wan experts retain their ownership. The comparison reuses strict name/shape/dtype
validation and compares materialized bytes, preserving bf16/float64, NaN payloads
and signed zero. No second digest format or universal numerical tolerance was
introduced. Frozen parameters are excluded by the existing trainable-state
contract, not by a new family-name table.

This is currently a constructor/API-level acceptance option, not a YAML flag or a
production-wide default. It performs synchronous full tensor readback only when
explicitly requested. Retained version slots, DTensor/sharded values, quantized
representations and meta tensors are rejected rather than credited with a live
parameter comparison. Slot activation, merge/requantization forward checks,
base-checkpoint mismatch acceptance, timing reports and a real-model/GPU CLI
remain required before this sprint is complete. Do not enable live-fleet poison
injection: the tests create isolated CPU parameter modules initialized to -99.

The real-process acceptance test uses the production Ray adapter, worker core and
loader/readback path with a small CPU model. It checks two independent engines and
one two-rank engine, including a deliberately skipped second-rank install. These
are actual Ray processes and tensor copies, not scripted version ACKs, but they
are not a multi-GPU model-parallel numerical validation. Default synchronization,
all-rank failure propagation and RPC adapters remain unchanged; the thin methods
are necessary family/RPC boundaries. Shared helpers remove duplicate namespace
validation without introducing contract dataclasses or algorithm constants.

Validation: 67 model readback, worker-slot, Ray weight-sync and architecture tests
passed (one existing dependency warning). The CPU bitwise tests cover poisoned
parameters, reordered values, bf16/float64, signed zero/NaN, omitted expert roots,
and attempts to include frozen parameters. The Ray tests retain the sender
snapshot and verify failed receivers keep their previous version.
