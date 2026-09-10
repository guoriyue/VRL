# SPRINT：Routing replay：真实 MoE policy 出现后再保存专家选择

状态：**parked。触发：VRL 的可训练 policy 确实包含 MoE router，且同权重 rollout/replay 的专家分歧可测。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 现状与边界

论文 §2.5 的 R3 记录每 token/layer/top-k 的实际专家选择。它不适用于 dense DiT，
也不等于分层 attention 路由或生成任务选择。论文的大规模 reference run 本身未开启 R3。

扩展点是 `vrl/trajectory/types.py` 的 segment tensor、
`vrl/trajectory/builders.py` 的真实 producer，以及
`vrl/rollouts/evaluators/` 和对应 family forward。先确认新 family 模型的真实 router；
不能因为名字里有 hybrid 就判定有 MoE。

## 启动后步骤

1. 在同权重同输入下记录 router logits、selected experts 和 logprob，
   分开 token/精度/路由三类差异，确认 R3 值得做。
2. 将实际专家选择以显式 token/layer/top-k axes 存入轨迹，绑定模型 revision
   和 policy version；设计容量时计入 queue bytes 和 host transfer。
3. family replay 前向消费记录的专家选择；缺失/越界/重复语义错误必须拒绝。
   梯度只流向实际参与该动作的参数，但 router 本身的训练目标必须单独说明，
   不能以强制 index 替代后宣称所有 router 梯度仍等价。
4. 单 GPU正确性通过后再测多 rank expert placement。跨 EP 布局的 ID 映射
   必须由实际模型 owner 完成，不建立 workflow 中的专家名名单。
5. 比较 R3 on/off 的 replay 差异、存储和吞吐；检查异步陈旧权重下收益是否仍存在。

## 验收与资源预算

例：32K tokens × 60 layers × 8 experts × 4 bytes 约 60MiB/trajectory，
完整 group 和多批队列会放大，不能只测一条。
注入 expert index 越界、层数变更、错版本和缺 payload，必须在 forward 前报错。
dense 路径不分配、不传输任何 routing tensor；受影响专家的梯度有实证。

## 应改／应留／非目标

只扩展真实 MoE family 的采样/replay 数据边界；保留通用 trajectory 容器。
模型维度是合法常量；禁止按 family 名创建全局 R3 支持名单。
不替代 TIS，不声称解决权重陈旧，不实现通用 MoE training backend。
