# SPRINT：Token OPD：在学生实际动作上取得 teacher 信号

状态：**parked。触发：Janus/token-policy 信用分配 pilot 成立，且找到 token vocabulary 和视觉 token 语义兼容的 teacher。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 现状与来源

论文 §5.2 的 sampled-token OPD 是
`A_t <- A_t - lambda * (logp_student_sampled - logp_teacher_same_token)`。
两个 logprob 是固定数据；不能将 current replay logprob 错当 detached rollout logprob。
论文还讨论 top-K 和多 teacher，但第一条 VRL pilot 不需要这些。

相关 owner：`vrl/algorithms/grpo/token.py`、
`vrl/algorithms/grpo/multisegment.py`、`vrl/algorithms/advantages.py`、
`vrl/trajectory/builders.py`、`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py`。
现有 image reward scorer 不能自动成为 token-probability teacher。

## 启动后步骤

1. 固定 tokenizer、special/visual token 映射、processor、模板及视觉上下文编码。
   只比较 token 名字不够；teacher 必须实际能评分学生的相同动作序列。
2. 复用真实 rollout token/mask/old-logprob，teacher 在无梯度路径评分，
   输出严格对齐的每 token 信号及 teacher revision。拒绝缺分数或长度错位。
3. 在 algorithm/advantage owner 内做可见的增量修正；不建立 reward registry 旁路。
   先单 teacher、sampled-token；任务 reward 可保留，distill-only 是显式实验。
4. 执行 reward-only、teacher-only、combined 的相同 prompt/seed/compute 对照。
   若只是输出变短或少生成图片，不能称质量提升。
5. 只有 sampled-token pilot 成立后再研究 top-K 和 metadata teacher routing。

## 验收

teacher=student、相同权重输入时额外项应接近零；logprob 差的正负产生正确方向。
tool/environment/padding token 无梯度；teacher 参数和缓存不会被 optimizer 更新。
注入 tokenizer mismatch、缺 image context、超时与截断，拒绝错误轨迹。
held-out 质量与总算力预算均有报告，多 seed 给出区间。

## 应改／应留／非目标

扩展现有 token trajectory 和 advantage 消费路径；保留 diffusion objective 与普通 reward。
teacher scoring 是真实模型边界，薄 adapter 有用；不要引入“通用知识蒸馏 Contract”。
不把这个 token reverse-KL 公式直接用于 NFT/V-GRPO，不把教师当真实质量标签。
