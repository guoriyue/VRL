# SPRINT(auto): vrl/rollouts/collector/batch_builder.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rollouts/collector/batch_builder.py` (420 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是 collector 真正的核心转换器（trajectory-backed GenerationOutput -> RolloutBatch），逻辑该留，但 `RolloutBatchBuildContext.extra` 字段是死字段，且 `RolloutBatchBuildContext` 命名带引擎味，和周边 `RolloutBatchBuildContext`/`RuntimeBundle` 概念易混。

## 1. 现状（读代码得出）
`TrajectoryRolloutBatchBuilder` 按 segment 的 distribution 分派到 `_pack_diffusion` / `_pack_ar_discrete` / `_pack_ar_continuous` / `_pack_ar_multisegment`，把 trajectory 张量打包成 `RolloutBatch`。这是真核心，无可质疑。

`RolloutBatchBuildContext` 是构建期元数据载体：
```python
@dataclass(slots=True)
class RolloutBatchBuildContext:
    metadata: dict[str, Any]
    device: Any | None = None
    kl_reward: float = 0.0
    reward_view_name: str | None = None
    trajectory_storage_policy: TrajectoryStoragePolicy = ...
    reward_artifact_policy: RewardArtifactPolicy = ...
    extra: dict[str, Any] = field(default_factory=dict)   # line 42
```

## 2. 质疑点 / 改进机会
1. `extra` 字段（batch_builder.py:42）是死字段。grep 确认：所有 `RolloutBatchBuildContext(...)` 构造点（core.py:132、三个测试）均未传 `extra=`；body 内也无 `self.context.extra` 读取（grep `\.extra` 在本目录只命中 `self.output.extra` at line 123，那是 `GenerationOutput.extra`，与本 dataclass 无关）。这是一个预留但从未接线的逃生舱口，按 AGENTS.md "thin / dead" 规则应删。

2. 命名：`RolloutBatchBuildContext` 以 `Context` 结尾，属于引擎味泛词。AGENTS.md 明确：build-time spec 不应叫 runtime/context/manager/handler，尤其周边已有 `GenerationRuntime`/`RuntimeBundle` 概念时会碰撞。这里它纯粹是"构建一个 batch 所需的输入参数包"，叫 `RolloutBatchBuildSpec` 或 `RolloutBatchBuildParams` 更准确，能与 runtime 概念区分。注意：dataclass docstring 自己都写 "Non-engine metadata"，说明作者也意识到它不该带引擎味。

## 3. 建议动作
- 删除 `RolloutBatchBuildContext.extra` 字段（line 42）。无任何构造/读取点，安全删除。
- 将 `RolloutBatchBuildContext` 重命名为 `RolloutBatchBuildSpec`（或 `...Params`）。同步更新 core.py:132、batch_builder.py 的 import/`__all__`、以及三个测试文件的引用。
- 分派逻辑（`build` 的 distribution switch）和 `_pack_*` 方法不动——这是真核心。

## 4. 不动什么 / 为什么不是过度清理
- 四个 `_pack_*` 方法看似重复（都 new 一个 RolloutBatch），但各家族字段语义不同（observations 是 prompt_ids vs latents、videos 维度不同、multisegment 还要算 r1_segment_names），强行合并会引入条件分支地狱，降低可读性。保留分派形状属于"consistency over cleanup"，不动。
- `_role_tensor` / `_named_tensor` / `_optional_named_tensor` 是带语义校验的取值器（role 必须唯一、缺失给清晰报错），不是薄 wrapper，保留。

## 5. 验证
- 删 `extra` 后跑 `grep -rn "RolloutBatchBuildContext.*extra\|\.context\.extra" --include=*.py` 应为空。
- 重命名后 `grep -rn "RolloutBatchBuildContext" --include=*.py` 应为空，且 `pytest tests/rollouts/test_collector_runtime.py tests/rollouts/test_janus_pro_r1_wiring.py tests/generation/diffusion/test_storage_policy_adoption.py` 通过。
- `ruff check vrl/rollouts/collector/batch_builder.py`。
