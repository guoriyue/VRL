# SPRINT：Multi-LoRA：共享 base 前先证明 adapter 状态隔离

状态：**parked。触发：同一 base 上至少两个真实训练任务需要同时服务/训练，并有可量测的 base 重复显存成本。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 现状与风险

论文 §5.1 的多 adapter 路径是实验性的，且有 placement/backend 限制；
operation-driven/Tinker frontend 不能当已发布功能。
VRL `vrl/models/steps/token/lora.py`、
`vrl/models/steps/denoise/common/lora.py`、`vrl/trainers/optimizer.py`、
`vrl/trainers/weight_sync.py` 和 `vrl/generation/execution/worker.py`
形成当前单训练任务与 policy version 路径。先审计现有 merge/version-slot 行为，
不把 LoRA support 等同于多 adapter 隔离。

## 启动后步骤

1. 只选一个现有 family 和两套实际任务，证明共享 base 的显存收益大于调度/切换成本。
2. 以 (adapter identity, installed version) 标识 rollout 与训练样本，
   base checkpoint hash 一致才可共享；不能让一个全局 version 覆盖所有 adapter。
3. 每 adapter 拥有 optimizer moments、step、数据游标与 reward 配置。
   一个 adapter 更新只发布其变化；无梯度的 adapter 不推进 step。
4. 调度显式选择 adapter，验证 forward 和 replay 读同一 adapter。
   单个可变 merged base 不能并发承载两个版本；不具备隔离能力就串行。
5. checkpoint 保存/恢复两套状态及共同 base 身份；第一版不处理任意动态租户加入退出。

## 验收

交替训练 A/B 与两个独立单 adapter 基线相符；更新 A 不改变 B 的参数、moments、
step 或固定输入输出。错 adapter/错 base/错版本请求在生成前失败。
量测驻留 HBM、切换延迟、训练吞吐；OOM/取消不能污染另一 adapter。
只同步已更新 adapter，导出 bytes 可验证；恢复后两任务继续步进正确。

## 应改／应留／非目标

扩展有真实需求的 adapter/optimizer/版本所有权，保留现有 family adapter 和同步接口。
必要复合身份属于协议；不建新业务 taxonomy 或“所有多租户场景”框架。
不提供 SaaS serving，不默认混 batch，不让 diffusion merge 路径假装隔离。
