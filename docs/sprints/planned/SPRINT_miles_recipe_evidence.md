# SPRINT：Recipe evidence：把冒烟、完整曲线和确定性回归分开

状态：**planned；先执行证据盘点，不等待新增模型或集群。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与现状

论文 §6–8 把证明对象限定为具体 recipe、拓扑和环境。VRL 的
`tests/e2e/test_real_checkpoint_rl.py` 已验证真实 checkpoint 的有限步更新、
loss/grad 有限性和 logprob parity；`tests/architecture/test_real_cover_labels.py`
已有 fake/real 对应关系。它们不能单独证明完整学习曲线或夜间稳定复现。

当前必须读取的生产消费者是 `vrl/scripts/train.py`、
`vrl/trainers/online/trainer.py`、`vrl/trainers/checkpointing.py` 和
`vrl/scripts/eval/image_checkpoint_eval.py`。先确认具体 trainer 路径；
不能把老 sprint 的历史路径作为修改依据。

## 目标与产物

每个被宣传为“验证过”的 recipe 都能追溯到一次具体运行：
recipe/config hash、代码和模型 revision、硬件/驱动/torch/kernel、
种子、拓扑、精度、步数、checkpoint、固定评估协议、原始指标及运行日期。
使用实验输出中的一个 manifest 和人可读索引；不向 family registry 加“verified”布尔值。

证据区分完整训练曲线、确定性全尺寸回归、声明过缩小差异的 proxy、仅 smoke。
硬件没有覆盖就明确留空，不从同名 GPU 架构或相邻模型推断。

## 实施步骤

1. 盘点 e2e CASES、experiment presets、已有 curve artifacts；输出每项证据及缺口，
   不把未运行测试写成通过。沿用 real-cover 机制连接 proxy 与真实实验。
2. 选 SD3.5 最小真实 recipe 做参考，固定数据/reward/model 来源及指标集合。
   在相同环境重复运行至少两次，区分采样随机性、GPU 非确定性、外部 reward 波动。
3. 确定性只对已验证的本地 backend/reward 组合启用；遇到非确定性算子直接报告，
   不靠放宽阈值让“bit-exact”成立。外部 API reward 另做统计比较。
4. 保存不可静默覆盖的参考指标与环境身份。确定性指标逐位比较；
   统计性指标预先确定重复次数、置信区间和最小关心效应。
5. 接入现有 GPU CI lane，CPU CI 只验证 manifest、证据文件和引用完整性。
   更新基线必须展示旧/新结果和原因，不允许失败时自动重录。

## 验收与失败注入

- 删除证据文件、改 config/hash、替换模型 revision、伪造 proxy 为 full，必须失败。
- 相同环境可重放；环境变化标记为不可直接比较，不偷偷使用旧 golden。
- 真实多步曲线与 held-out 指标可访问；确定性未实现时仍诚实标记 smoke/curve-only。
- 性能数字单列 warm-up、采样次数和量测边界。不能用 reward 上涨一个 seed 宣布学习收益。

## 应改／应留／非目标

改变证据记录及检查；保留现有训练入口、e2e helper、真实接口和 real-cover 标签。
CASE 常量是测试 fixture，保留；不创建模型支持名单或验证状态解释器。
不因论文使用 nightly 就虚构可用 GPU runner，不重写 CI 调度平台。
