# SPRINT: NGU 自适应采样 — 把 rollout 预算分配给尚未解决的问题

**日期**: 2026-09-17  
**状态**: PLANNED — 尚未实现；本文件记录方案与验收要求。  
**来源**: [Learning to Solve Hard Problems in RL for LLMs by Never Giving Up](https://arxiv.org/abs/2609.13443)，[作者博客](https://mnoukhov.github.io/posts/ngu/)。

## 目标与适用范围

实现默认关闭的 Never Give Up（NGU）采样策略：每个 prompt 先生成小组样本；
首组全成功则跳过本次训练，有成功和失败则形成训练组，全失败则以配置概率继续
采样。重试保留历史尝试，并设置资源预算，避免不可解问题无限占用算力。
跳过本次采样不等于永久删除该 prompt。

先在 OCR 任务上建立明确的文本匹配成功判定，并完成固定计算预算对比。
PickScore、美学分和一般视频连续奖励没有天然的“全对／全错”；其重试条件属于
后续方法扩展，不默认套用二值规则。不能把奖励方差很小直接解释成任务已解决。

论文的启示是区分已有能力的可靠性提升与难题突破，并审计真正参与更新的
样本分布；不预设 NGU 必然改善视觉生成任务。

## 当前代码与接入位置

以下基于本次代码阅读；开始实现前重新核对调用方、配置和测试。

| 位置 | 当前职责与计划 |
| --- | --- |
| `vrl/rollouts/orchestration/continuous/producer.py` | 已负责异步生成和评分收集。在 `_harvest_done()` 后引入采样决策，将重试与运行失败重试分开计数。 |
| 新增 `vrl/rollouts/orchestration/continuous/adaptive_sampling.py` | 管理逻辑 prompt 身份、重试决策、历史奖励统计、轨迹引用和预算；producer 驱动其状态转换。 |
| `vrl/rollouts/orchestration/continuous/consumer.py` | 当前 `_take_ready_groups()` 等待指定 batch 全部 slot 完成，且要求同一 policy version。NGU 需要新的有效组接收语义。 |
| `vrl/rollouts/orchestration/continuous/{schedule,thread,types,staleness,scored_queue}.py` | 接通补充 prompt、状态所有权、队列容量、实际生成版本和逐样本过期处理。 |
| `vrl/rollouts/orchestration/schedule.py` | 保留调度协议边界，明确 NGU 的适用模式与 capability 校验。 |
| `vrl/rollouts/collector/{core,requests,batch_builder}.py` | 复用生成、评分和组装接口；必要时补充尝试身份及 provenance，不在 collector 内藏训练策略。 |
| `vrl/algorithms/advantages.py` | 扩展历史 baseline 统计入口和过期样本后的 advantage 处理。 |
| `vrl/trainers/online/trainer.py` | `collect_training_batch()` 接收历史统计；检查变长逻辑组的 loss 权重、microbatch 和分布式同步。 |
| `vrl/rollouts/batch/core.py`、trajectory 类型 | 保留每次尝试的原始轨迹、行为策略 log-prob 和版本；避免为了逻辑分组强行拼接不兼容轨迹。 |
| `vrl/trainers/core/types.py` | 定义并验证显式配置，再投影到 continuous settings；默认关闭。 |

## 关键设计约束

### 1. 异步执行不等于自适应训练批次

现有 continuous consumer 要求当前 prompt batch 的每个 slot 恰好返回一组，
且整个 batch 同版本。仅在 producer 加重试循环仍会让训练等待最慢的 prompt。

NGU 必须允许已完成的有效逻辑组组成更新批次，未解决问题保留在有界重试队列，
跳过的简单题释放预算后补充其他 prompt。需要沿 `next_iteration()` 的调用链
核对数据迭代器、预取和 epoch 耗尽语义，不能仅放宽 consumer 的断言。
原有固定 batch 路径保持兼容；明确无有效组、预算耗尽和数据结束时的退出行为。

### 2. 身份、随机性与轨迹保真

- 用稳定的 example／采样周期身份关联历史，不能只用 prompt 字符串；相同文本
  可能有不同图像条件或 reward metadata。
- 每轮重试使用新的采样随机性，同时保留输入条件；记录 attempt 和实际生成版本。
- 同一逻辑组可以跨版本，但每个样本保留真实行为策略信息；禁止把旧样本统一
  标记为最新版本。复用算法对 off-policy staleness 的 capability 校验。
- diffusion 的 SDE window、时间步和 conditioning 等 replay 事实不能在合组时
  丢失。逻辑 advantage 分组与物理 replay 分块可以不同。
- 历史轨迹必须计入内存上限；退出、过期、取消和异常时释放。定义恢复时如何
  处理 pending 历史，禁止静默复用没有 provenance 的缓存。

### 3. 历史 baseline 与训练样本分开

按真实版本过滤过旧轨迹，允许保留其奖励计数用于组 baseline。
在二值奖励路径上实现并验证论文的 anchor-positive 处理：保留正样本优势，
调整仍参与训练的负样本优势，避免过滤旧负样本后无意改变组优势中心。
明确没有剩余负样本等边界行为。

当前 advantage 归一化／裁剪与论文 centered advantage 的区别必须显式处理；
不能把二值推导直接用于连续或多目标奖励。producer 不计算 advantage 或 loss。

## 执行计划

- [ ] **P0：建立基线。** 选择现有 OCR 实验，明确文本归一化和成功判定；固定
  独立评估集，按初始模型成功率分桶。记录每题成功率、实际训练组组成、零优势
  比例、生成／评分成本及所有丢弃样本。保存难度分桶规则和采样数。
- [ ] **P1：配置与调度。** 添加 NGU 开关、continuation probability、每次采样数、
  有界尝试／内存预算及过期策略；实现状态机、fresh/retry 公平调度和有效组补充。
  更新 producer/consumer/owner 协议及数据迭代器调用方，保持默认路径不变。
- [ ] **P2：历史训练信号。** 接入逐样本版本、历史奖励 baseline 与负样本 rescale；
  核对变长组的损失归一化、梯度累积、DDP/FSDP collective 次序和 replay 兼容性。
- [ ] **P3：比较与记录。** 在匹配计算预算下比较固定小组、固定大组和 NGU；
  使用多个随机种子，报告整体／各难度桶成功率、可靠性变化、成本和误差范围。
  在 `docs/runs/` 保存配置与结果，结论决定是否扩大任务范围。

建议增加指标：各 prompt 尝试数、首次成功所需样本数、跳过／放弃数、有效组比例、
历史轨迹字节数、过期样本数、版本滞后、队列等待和每次成功的计算成本。
训练 reward 均值受采样分布变化影响，不能替代固定评估集。

## 验证与验收

- [ ] 状态机覆盖全成功、混合结果、全失败、概率退出、预算耗尽及零有效组退出。
- [ ] 两个相同文本但不同条件的 example 不混组；重试随机性变化且可复现。
- [ ] 可用组不被单个长重试 prompt 阻塞；数据耗尽、取消、weight-sync drain 和
  队列背压均能有界结束；评分服务错误不会被当成任务失败。
- [ ] 历史版本和 log-prob 保真；过期轨迹不进入 loss，允许的历史奖励统计保留。
- [ ] 用小型手算用例验证 baseline、anchor-positive 和变长组损失权重。
- [ ] 多 rank 不同重试次数、空组和不同组大小不会造成 collective 不匹配。
- [ ] NGU 关闭时现有 strict/continuous 行为与回归测试保持通过。
- [ ] GPU 实验将所有尝试、评分和丢弃成本计入预算；报告难题收益及简单题退化，
  不能只报告训练 reward 上升。无收益也如实记录，不据此默认启用。

## 架构取舍与非目标

**应该改变**：调度的有效组接收语义、历史状态和 provenance、advantage 历史统计
接口及相应配置。独立 `adaptive_sampling.py` 承载真实状态和共享决策，不是薄转发。

**保持不变**：family 生成接口、后端职责、replay evaluator 的概率计算，以及
collector 的生成／评分／组装边界。现有 schedule protocol 和 request adapter
虽薄，但提供协议边界及跨 family 一致性，保留。

**常量**：保留 `_DENOISE_FIELDS` 等 schema 字段集合和协议／metadata key 常量，
因为它们代表真实边界。NGU 概率、预算和成功阈值进入具名 typed config，避免
在 workflow 中新增 ALL_CAPS 参数表或任务词表。本 sprint 不顺带清理无关常量。

**非目标**：重写 GRPO loss、改模型架构、合并各 family 文件以减少行数、把 NGU
强加给 NFT/DPO、自动覆盖连续奖励语义、永久删除简单题，或默认开启 NGU。
跨 family 一致性优先于局部 LOC 缩减。
