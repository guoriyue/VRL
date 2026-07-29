# SPRINT: 零 fan-in preset 审计（done / no deletion）

状态：**DONE（2026-07-28）**。结论：原提议的 3 份 preset 全部保留。

## 核心结论

“仓库内没有 compose”不等于“不可达”。config loader 明确支持公开的 `/group=option` override；用户可以直接选择未被任何 experiment 引用的 menu option。删除后全量测试不失败，只能证明仓库内部 recipe 没用它，不能证明公开 CLI 面没有用户。

| Preset | 判决 | 理由 |
|---|---|---|
| `base/distributed/training_single_process.yaml` | **KEEP** | 显式 pin 默认 training strategy，与 FSDP option 形成可发现的配置面 |
| `sampling/denoise/10_step_no_cfg.yaml` | **KEEP** | 已实测可通过 group override 到达；是 10-step/no-CFG 的公开菜单项 |
| `sampling/image/896x1152.yaml` | **KEEP** | 已实测可通过 group override 到达；是 portrait/full-body shape 菜单项 |

原审计自己的 reachability 实验已经证明后两项生效，因此不能再把它们称为 dead code。

## 暴露出的真实问题

`base/distributed` 同时容纳：

- training strategy（`training_single_process`, `training_fsdp`）
- rollout resource topology（`ray_rollout*`）

loader 的 group replacement 会把同组 option 整体替换，所以在一个 online recipe 上选择
`/base/distributed=training_single_process` 会连 rollout resources 一起替换掉。这个问题属于 **group taxonomy / override semantics**，不是 `training_single_process.yaml` 单文件问题；只删默认项会隐藏症状，而 `training_fsdp` 仍保留同样的跨维度替换语义。

若后续修复，应单独设计 training-strategy 与 rollout-topology 两个正交 group，并为旧 override 提供明确迁移/拒绝行为。不要借 cleanup sprint 静默改变公开 config API。

## 后续 preset 审计规则

删除前必须先给候选分类：

1. family 唯一入口；
2. 仓库内部 composition；
3. 公开 group-override menu；
4. schema/default 的显式 pin；
5. test fixture；
6. 真正无 owner、无入口、无历史承诺的孤儿。

只有第 6 类可以仅凭 reachability 进入删除候选。`cosmos3_nano.yaml` 的实验也说明：一个 preset 即使 fan-in 很低，也可能是整个 family 的唯一入口。

## 保持不变

- 不删除或移动上述 3 个 YAML。
- 不把 sampling option 内联到 experiment；独立菜单有 grep、debug 与 CLI 复用价值。
- 不把 preset 扫描结果做成手维护 ALL_CAPS allowlist；family 入口应从 registry/config source of truth 验证。
- 不压平跨 family 的统一 preset 形状来省几行 YAML。
- 不在本审计里修改 loader/group 语义。

## References

- `vrl/config/loading.py`
- `vrl/config/presets/base/distributed/training_single_process.yaml`
- `vrl/config/presets/base/distributed/training_fsdp.yaml`
- `vrl/config/presets/base/distributed/ray_rollout.yaml`
- `vrl/config/presets/sampling/denoise/10_step_no_cfg.yaml`
- `vrl/config/presets/sampling/image/896x1152.yaml`
- `tests/config/test_load_all_experiments.py`
