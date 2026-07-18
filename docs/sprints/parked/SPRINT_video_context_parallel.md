# SPRINT: Video context parallel — 训练侧 CP 解锁 predict2_5 双卡 replay

**Date**: 2026-07-13  **Status**: **PARKED — requires a real 2-GPU host or CI runner**.
**来源**: FlashDreams 引擎评审教训 ②——它的多卡故事只有上下文并行且总共
157 行（`core/distributed/context_parallel.py:32-157`：token 维
`split_inputs_cp`/`cat_outputs_cp` + ring attention 组），证明视频 DiT 的
第一根并行轴是 CP/SP 而非 TP/流水线，且参考实现极小。

**vrl 侧痛点（记忆档案）**：predict2_5 是"2 卡家族"——480p 33f 的历史 run 是
ddp_2x1，单卡 replay 结构性放不下（activations 峰值），compile live gate
因此一直 OPEN；wan i2v 16.4B 单卡训练同样结构性不可能。DDP 只切 batch，
对"单样本 activations 就超卡"的形状无解——需要切序列。

## 目标

训练/replay 前向支持 2 卡上下文并行：单个样本的 token 序列切到 2 卡、
attention 走 ring（或先 allgather-attention 起步），使 predict2_5 480p 33f
的 replay 单步在 2×32GB 上跑通，logprob 契约不破。

## KILL-gate（顺序执行，任一 FAIL 即停）

- **P0（CPU/单卡可做）数值等价性**：CP=2 的 ring-attention 前向 vs 单卡前向，
  同权重同输入，logprob 差异必须落在 precision_drift_guard 现行阈值内
  （1e-3 量级）。ring 的分块 softmax 数值路径与单卡不同——这是本 sprint
  最大的正确性风险，先用小形状 + fp32 钉死，再看 bf16 漂移。若 bf16 漂移
  超阈且 TIS 校正兜不住，sprint 转 FAIL 记录。
- **P1 显存收益实测**：2 卡 CP 下 480p 33f replay 单步峰值 < 32GB
  （对照单卡 OOM 基线）。CP 切 activations 但权重全复制——2B 模型权重不是
  瓶颈，activations 是（rollout bound-class 档案），预期成立但必须实测。

## 变更清单

1. **CP 原语**（落 `vrl/trainers/` 现有分布式文件，不新建单类模块）：
   序列维 split/cat + ring-attention 前向。参考实现直接读 FlashDreams 的
   157 行（Apache-2.0，可节选改写）；起步可用更简单的
   allgather-KV attention（通信多但实现直、数值路径更接近单卡），ring 作为
   P2 优化。
2. **mesh 集成**：`distributed.training.fsdp.mesh` 扩展 `cp` 轴
   （2D：`["dp_shard","cp"]`），`build_strategy` 读取；单独 CP（无 FSDP）
   作为 ddp 的对位形态也要能配。
3. **replay 路径接线**：diffusion replay forward 的输入在进 transformer 前
   split、输出 cat；`sde_step_with_logprob` 在完整序列上算（cat 之后），
   保证 logprob 语义与单卡一致。家族先只接 predict2_5（痛点正主），
   接口放 base 层但不强推全家族。
4. **rollout 侧不动**：生成单卡放得下（V2W gen 峰值 25.3GB 档案），CP 只进
   训练/replay。colocated 时分复用语义不变。
5. **compile live gate**：CP 跑通后重开 predict2_5 的 compile 验证
   （此前因单卡放不下一直 OPEN）。

## 非目标

- TP / 流水线并行（FlashDreams 的教训正是：视频 DiT 不需要先做这些）。
- 跨节点 CP（cross_node 路径另案；先 2 卡单机）。
- rollout 侧 CP、生成加速（rollout 已 94% MFU，无空间）。

## 验证

- P0 数值门（上）；P1 显存门（上）。
- 2 卡 480p 33f 真实 GRPO 单 epoch dry-run：first-step logprob round-trip
  check 过 drift guard；grad_norm 非零；ckpt save/resume 一致。
- ws=2 CPU gloo 单元测试进 tests/trainers/（沿用 FSDP2 层的 CPU 可测惯例）。
