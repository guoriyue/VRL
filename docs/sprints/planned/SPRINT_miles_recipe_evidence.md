# SPRINT：Recipe evidence：把冒烟、完整曲线和确定性回归分开

状态：**implementing；启动身份记录已接入 online，曲线与确定性验收仍待执行。**

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


## 2026-09-09：启动证据记录已实施

`vrl/trainers/evidence.py` 与 online runner 已接线：每次启动/resume 原子发布独立
`run_evidence/<launch_id>.json`，记录 resolved config/hash、model identity、configured
manifest/report 内容 hash、Git/dirty-diff、软件版本、GPU/driver 和数值运行开关。
不添加 family verified 名单，不给仅启动过的 recipe 打成功或确定性标签。

配置/证据输出和原 CSV preflight 共用 `run_primary_io`：它是实际跨 rank 的失败传播边界，
避免 primary 写盘失败时其他 rank 继续进入训练 collective。协议名、文件目录与环境 key
是边界常量，保留；没有新增 per-algorithm vocabulary 或第二套 model identity。

验证：64 项 evidence、online lifecycle/metrics 和 architecture 测试通过，其中包含真实
双进程 Gloo 的 primary IO 成功/失败传播、启动证据失败后的清理，以及配置解析/hash、
manifest 内容变化、resume 不覆盖和发布冲突测试。一个警告来自 PyTorch TF32 旧接口提示。

使用及证据范围见 [说明](../../training_examples/RUN_EVIDENCE.md)。
尚待：运行结束后的 metrics/checkpoint/eval 证据封存与校验、历史 recipe 盘点、真实重复短曲线、
确定性/统计性比较和 GPU CI 接线。当前快照不是完整训练复现报告。
