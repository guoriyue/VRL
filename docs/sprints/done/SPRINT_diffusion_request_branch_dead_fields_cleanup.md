# SPRINT: diffusion request / branch 输入层死字段清理（done）

状态：done（落地 commit `49c2a14`；2026-06-21 归档）。
范围：删除 diffusion generation 输入 struct 上从不被读的死字段：`VideoGenerationRequest.{model_name, shift, references}`、`DiffusionBranch.name`，以及随之孤立的 `DiffusionBranchName` 类型别名。
来源：dead-dataclass-hunt confirmed-dead，承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

`VideoGenerationRequest` 的三个字段是输入 DTO 上的旧残留：
- `model_name`：executor 构造请求时不设置，下游不读取。
- `shift`：scheduler shift 由 scheduler/model config 管，不走 request。
- `references`：diffusion request 下游不读取；仓内同名活字段是无关的 `PromptExample.references`。

`DiffusionBranch.name` 同样是 write-only：family runner 构造 branch 时写入 `"cond"` / `"uncond"`，但 CFG packing 路由靠函数参数和 `metadata`，不读 `.name`。删除 `name` 后，`DiffusionBranchName` 类型别名失去唯一用途，一并删除。

## 1. 已删除内容

- `vrl/generation/diffusion/layout.py`：删除 `VideoGenerationRequest.model_name` / `shift` / `references` 字段。
- `vrl/models/diffusion/common/cfg.py`：删除 `DiffusionBranch.name` 和 `DiffusionBranchName`。
- `wan_2_1`、`sd3_5`、`cosmos predict2`、`cosmos predict2.5` runner：删除 `DiffusionBranch(name=...)` / `build_branch(name=...)` 传参。
- tests：删除只为这些死字段存在的构造参数和 round-trip 断言。

本 commit 还顺带清理了 `ARAttention*Output.metrics` 的 display-only 字段；该项已有独立 done 记录：[[SPRINT_ar_attention_output_metrics_cleanup]]。它不是本 sprint 的主题，但同一提交中落地。

## 2. 验证

- 落地提交记录：相关测试 `126 passed`。
- Review 复核：`rg "DiffusionBranchName|branch\.name\b|DiffusionBranch\(|VideoGenerationRequest\(|model_name=|shift=|references=" vrl tests configs -g '*.py' -g '*.yaml'` 只剩无关同名字段和正常构造；无 `branch.name` reader。
- Review 复核：`pytest tests/models/diffusion tests/generation/pipeline tests/generation/ar tests/rewards/inference tests/rollouts/replay tests/trainers/online/test_step_split.py tests/algorithms -q` → `289 passed`。
- Review 复核：`python -m vrl.config.lint` 通过；`git diff --check origin/main...HEAD` 通过。

## 3. Non-Goals

- 不删 `VideoGenerationRequest.task_type`：它仍经 prompt collection / rollout request metadata / reward artifact provenance 进入 reward 打分链路。
- 不删 `VideoGenerationRequest.seed`：它是 diffusion 采样 seed 的活输入。
- 不删 `DiffusionBranch.metadata`：CFG/postprocess 路由仍使用它。
- 不碰 `DiffusionBackboneOutput.metrics`：backbone contract 测试仍断言 `transformer_calls`。

## References

- `49c2a14 refactor(diffusion): drop dead request/branch fields`
- `vrl/generation/diffusion/layout.py`
- `vrl/models/diffusion/common/cfg.py`
- `vrl/models/diffusion/wan_2_1/runner.py`
- `vrl/models/diffusion/sd3_5/runner.py`
- `vrl/models/diffusion/cosmos/predict2/runner.py`
- `vrl/models/diffusion/cosmos/predict2_5/runner.py`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]、[[SPRINT_ar_attention_output_metrics_cleanup]]
