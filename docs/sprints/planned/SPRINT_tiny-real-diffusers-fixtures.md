# SPRINT: 完成 tiny-real diffusers fixtures 转换（轨道四剩余工作）

状态：**planned**。轨道顺序 4 / 6。风险：medium。

审计与已完成子集见
`docs/sprints/done/SPRINT_tiny-real-diffusers-fixtures_audit.md`。commit `300ef8c7`
已经加入真实 VAE fixture，并把 VAE 内存策略与 Wan DPO encoder 测试从方法调用记录改成
真实对象状态断言。本计划只保留仍未落地的工作，不重复执行该子集。

## 剩余实施清单

1. **Pipeline shell：**加入无下载、CPU、config-init 的 tiny pipeline shell builder，
   用它替换 frozen-offload 测试中自声明的 pipeline 替身。
2. **Scheduler 与生产 guard：**用真实 scheduler 替换 `_TinyScheduler`；真实测试覆盖到位后，
   删除 Mochi 与 PixArt Sigma 中只为替身存在、会静默跳过标准化的两个 scheduler guard。
3. **Anima：**先在 parity 测试中加入两个基于真实 transformer 的调用与分支断言，再删除旧的
   `test_forward_step.py`；必须保持“先加后删”。
4. **Cosmos3：**在兼容的 diffusers 版本上加入 tiny transformer / pipeline fixtures，
   完成 packed-static 装配、forward 参数与 CFG/decode 三组真实对象测试。
5. **NextStep：**加入真实 f8 VAE，并把重复 fixture 构造收敛到家族 fixture 模块；保留不可导入
   上游包的边界替身。
6. **SANA：**把重复的 scheduler 构造收敛为共享 builder，同时保留用于验证 hub 参数投影的
   `from_pretrained` recorder。
7. **基础设施标注收尾：**按当前 `real_cover` 契约补齐审计登记的真实对位与诚实缺口；
   命名目标可以位于 default lane，但目标必须存在且 `why` 必须解释覆盖差距。

## 明确非目标

- 保留 `_IdentityDecodeVAE`：它是让 layout 与反归一化算术可观测的 identity 探针，不是模型替身。
- 保留 scheduler wrong-class 测试中的假对象：它隔离“类名不匹配”这一半校验，真实错误类会同时
  引入 config 不匹配。
- 保留 NextStep 的 `sys.modules` 注入与 `UpstreamModel.unpatchify`：对应上游包不是仓库依赖，
  这是必要的包边界适配。
- 保留 SANA `from_pretrained` recorder：它验证 revision/subfolder 参数投影，不试图复现 Hub。
- 保留 Wan 与 SANA 的 fail-loud scheduler guards；它们保护真实可选字段或提供明确错误。
- 不把 diffusers pipeline 下载车道并入本轨；本轨 fixtures 必须 config-init、CPU、无网络。

## 完成判据

- 上述七组剩余工作全部落地，原审计中对应替身和重复构造消失。
- 每个删除动作先有真实对象测试覆盖，且相关家族测试、默认测试全集与 scoped Ruff 全绿。
- `real_cover` 标注通过架构守卫，所有 `tracked_in` 路径在磁盘上存在。

## References

- 审计快照：`docs/sprints/done/SPRINT_tiny-real-diffusers-fixtures_audit.md`
- 已落地子集：commit `300ef8c7`
- Track 1 契约：`docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`
