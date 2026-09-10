# SPRINT：Evaluation provenance：独立评估的迟到结果仍归属于正确 checkpoint

状态：**planned；先补结果身份，自动派发仅在已有 evaluator 可复用时实现。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与现状

论文 §2.2.4 的可借鉴点是 delayed result attribution，不是重新把 eval 塞入 trainer。
VRL 已在 `docs/sprints/done/SPRINT_remove_inline_fixed_eval.md` 删除 inline eval。
`vrl/scripts/eval/image_checkpoint_eval.py::EvaluationPlan` 已记录模型/runtime 身份，
并校验 checkpoint digest；`vrl/trainers/checkpointing.py` 是 checkpoint owner。

差距审计必须逐个检查实际 output report 是否保留 run identity、权重对应的 step、
dispatch/completion 时间和失败原因；已有字段直接复用，不平行创建 manifest。

## 实施步骤

1. 为 image evaluator 和一个实际使用的 video evaluator 列出 input→report 身份链，
   补缺失的 training-run/step、checkpoint hash、评估协议 hash、开始/结束时间。
2. 独立离线汇总工具把分数挂到 checkpoint step；完成顺序不影响图的横轴。
   对已有充分证据路径，本步骤只增加验收，不重复实现。
3. 如需要自动追踪训练目录，使用单个外部 CLI 调用已有 evaluator：
   只消费已完整提交且可校验的 checkpoint；限制 pending 数、临时磁盘和进程数。
4. 临时失败、文件缺失、hash 不符、GPU 不可用分别记录。失败不更改 optimizer、
   checkpoint 发布或训练 admission。单卡共用时默认训练结束后评估。
5. 复用周期性 checkpoint，不为每次评估额外加入 trainer collective/export。

## 验收

- v12 比 v10 先返回，报告仍归于各自版本；重复结果按 run/hash/protocol 去重。
- checkpoint 写到一半、评估期间被替换、同文件名不同内容必须拒绝。
- 一个 worker 评错版本不能被“平均版本恰好等于目标”掩盖；有样本级版本时逐项核验。
- 超时/取消留下 skipped/failed 记录，临时进程和文件被回收。
- 真实 checkpoint 完成一次独立评估，并证明训练代码没有新增 eval 调用。

## 应改／应留／非目标

改变报告及外部汇总/派发工具；保留 CheckpointTarget、EvaluationPlan 和独立 evaluator。
它们是输入身份与执行边界，不因较薄就合并；schema key 常量保留。
不恢复 `trainer.eval`，不添加共享-engine eval phase，不新建 Ray eval fleet。
