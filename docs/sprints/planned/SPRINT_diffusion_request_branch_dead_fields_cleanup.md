# SPRINT: diffusion request / branch 输入层死字段清理（planned）

状态：未开始（2026-06-20）。
范围：删除 diffusion generation 输入 struct 上从不被读的死字段——`VideoGenerationRequest` 三字段 + `DiffusionBranch.name`。纯机械删。
来源：dead-dataclass-hunt confirmed-dead，置信 high。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

两个 diffusion 输入 struct 各带若干「构造写入、下游零读取」的字段：request 三字段从不被 `model.encode_prompt()` / `prepare_sampling()` 读取；branch 的 `name` 在所有 `build_branch()` 写入但 `postprocess_branch()` / `pack_batched_cfg()` 只读 `.metadata`。

## 1. 现状实锤

### 1.1 `VideoGenerationRequest` 三死字段
`vrl/generation/diffusion/layout.py`（`class VideoGenerationRequest`，`@dataclass(slots=True)`）：

```python
references: list[str] = field(default_factory=list)   # dead
...
model_name: str = ""                                   # dead
...
shift: float = 1.0                                      # dead
```

`grep -rEn "\.(model_name|shift|references)\b" vrl/generation/diffusion/ vrl/models/diffusion/`（排除定义/类型注解）**零读取命中**。三字段都不进任何 diffusion family 的 encode/sampling 路径。

> ⚠️ **勿连坐**：同 struct 的 `task_type` 是**活字段**（OVERTURNED）——经 `prompt_collection.py:135` → `requests.py:112` metadata → `rewards/artifacts.py:131-149` `_artifact_provenance` 进 reward 打分。**保留 `task_type`**，只删 `model_name` / `shift` / `references`。

### 1.2 `DiffusionBranch.name`
`vrl/models/diffusion/common/cfg.py:15-25`：

```python
@dataclass(slots=True)
class DiffusionBranch:
    name: DiffusionBranchName    # cfg.py:18 — dead
    hidden_states: torch.Tensor
    ...
    metadata: dict[str, Any] = field(default_factory=dict)
```

`grep -rEn "branch\.name\b" vrl/` 零命中。各 family runner 用 `build_branch()` 写入 `name="cond"/"uncond"`，但 cfg 后处理（`postprocess_branch()` / `pack_batched_cfg()`）路由只读 `.metadata`，从不读 `.name`。`name` 是被 `metadata` 取代后残留的旧路由键。

## 2. 落地方案

### A. 删 `VideoGenerationRequest` 三字段
- 删 `layout.py` 中 `references` / `model_name` / `shift` 三个字段定义。
- grep 所有 `VideoGenerationRequest(...)` 构造点，删对应 kwarg（若有显式传入）。
- 保留 `task_type`。

### B. 删 `DiffusionBranch.name`
- 删 `cfg.py:18` 的 `name` 字段。
- 删所有 `build_branch(name=...)` / `DiffusionBranch(name=...)` 构造处的 `name=`（family runner：wan_2_1、sd3_5、cosmos 等）。
- 路由逻辑已走 `metadata`，无需替换。

## 3. 验证

- `grep -rEn "\.(model_name|shift|references)\b" vrl/generation/diffusion/ vrl/models/diffusion/` 零命中（`task_type` 仍在）。
- `grep -rEn "branch\.name\b|DiffusionBranch\(.*name=" vrl/` 零命中。
- `pytest tests/generation/diffusion/ tests/models/diffusion/ -q` 全绿；diffusion CFG rollout 冒烟一致。

## 4. 非目标 / Non-Goals
- 不删 `VideoGenerationRequest.task_type`（活，reward provenance）。
- 不删 `DiffusionBackboneOutput.metrics`（OVERTURNED：`tests/models/diffusion/common/test_backbone_contract.py:94,117` 断言 `transformer_calls` 计数）——它另属 backbone 契约，不在本 sprint。
- 不动 `DiffusionBranch.metadata`（活路由键）。

## References
- `vrl/generation/diffusion/layout.py`（`VideoGenerationRequest`：`references` / `model_name` / `shift`）
- `vrl/models/diffusion/common/cfg.py:15-25`
- `vrl/rollouts/orchestration/prompt_collection.py:135`、`vrl/rollouts/collector/requests.py:112`、`vrl/rewards/artifacts.py:131-149`（`task_type` 活链，勿删）
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
