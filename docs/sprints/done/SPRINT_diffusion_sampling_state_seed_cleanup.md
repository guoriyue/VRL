# SPRINT: diffusion `*SamplingState.seed` 死字段清理（done）

状态：done（落地 commit `55c9410`；2026-06-21 归档）。
范围：删除 6 个 diffusion `*SamplingState` dataclass 上「`prepare_sampling()` 写入、但从不被 `state.seed` 读回」的死 `seed` 字段。
来源：repo-wide dead-dataclass-hunt（194 structs 审计 → 对抗验证）confirmed-dead，承接 [[SPRINT_segment_signal_dead_field_cleanup]] 的派生结构体审计方法论。

## 0. Core Decision

种子值的唯一活跃读取点是 request-level `seed`，例如 family runtime 在 `prepare_sampling()` 内读取 `request.seed` 并立即初始化 `torch.Generator`。把同一个值再拷进 `SamplingState.seed` 是 write-only 存档副本：不进控制流、不进 compute、不进 replay/export 序列化。

勿连坐：
- `GenerationRequest.seed` 保留：它仍是 diffusion 采样的权威 seed 输入。
- `ARSamplingParams.seed` 保留：AR runtime 仍用它驱动 `torch.manual_seed(params.seed)`。
- `TrainerConfig.seed` 保留：训练随机性配置仍是活字段。

## 1. 已删除内容

删除 6 个 `seed: int` 字段及所有构造传参：

| struct | 所在文件 |
|---|---|
| `WanT2VSamplingState.seed` | `vrl/models/diffusion/wan_2_1/model.py` |
| `WanI2VSamplingState.seed` | `vrl/models/diffusion/wan_2_1/model.py` |
| `CosmosPredict2SamplingState.seed` | `vrl/models/diffusion/cosmos/predict2/model.py` |
| `CosmosPredict25SamplingState.seed` | `vrl/models/diffusion/cosmos/predict2_5/model.py` |
| `AnimaSamplingState.seed` | `vrl/models/diffusion/cosmos/anima/model.py` |
| `SD3SamplingState.seed` | `vrl/models/diffusion/sd3_5/model.py` |

同时删除测试/镜像构造点里的 `seed=...`，包括 replay-forward mirror 里原本的 `seed=0`。`prepare_sampling()` 里用于初始化 generator 的局部 `seed` 逻辑保留。

## 2. 验证

- 落地提交记录：`tests/models/diffusion/` 相关回归 `99 passed`。
- Review 复核：`rg "\.seed\b|seed:" vrl/models/diffusion tests/models/diffusion tests/scripts -g '*.py'` 只剩 request-level seed、测试 fixture seed、脚本参数 seed 等活语义；无 `state.seed` 读者。
- Review 复核：`pytest tests/models/diffusion tests/generation/pipeline tests/generation/ar tests/rewards/inference tests/rollouts/replay tests/trainers/online/test_step_split.py tests/algorithms -q` → `289 passed`。
- Review 复核：`python -m vrl.config.lint` 通过；`git diff --check origin/main...HEAD` 通过。

## 3. Non-Goals

- 不动 `GenerationRequest.seed`、`ARSamplingParams.seed`、`TrainerConfig.seed`。
- 不改 generator seed 初始化行为。
- 不把 replay seed 语义迁移到 `SamplingState`；该字段已确认不是 replay source of truth。

## References

- `55c9410 refactor(diffusion): drop dead SamplingState.seed copies`
- `vrl/models/diffusion/wan_2_1/model.py`
- `vrl/models/diffusion/cosmos/predict2/model.py`
- `vrl/models/diffusion/cosmos/predict2_5/model.py`
- `vrl/models/diffusion/cosmos/anima/model.py`
- `vrl/models/diffusion/sd3_5/model.py`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
