# SPRINT(auto): vrl/generation/ray/launcher.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/ray/launcher.py` (409 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件本身是 Ray generation 启动的真核心（launch / launch_from_cfg / build_inputs），但 `__all__` 里把 `RayPlacement` 和 `create_generation_placement_group` 当成自己的 API 再导出一遍是死的对外表面，另外 `_cfg_get`/`_cfg_path` 是项目里重复抄了 ~10 份的同一个 config 取值 helper，应收敛到共享工具。

## 1. 现状（读代码得出）
launcher 负责三件互相关联的事，职责是内聚的，不算 god-file：
- `launch(...)`（line 53）：建 placement group、起 actor group、校验 GPU、组装 `RayGenerationRuntime`。
- `launch_from_cfg(...)`（line 151）：从训练 cfg 解析 + driver CUDA 校验 + 选 `ReleasableRayGenerationRuntime`/普通 runtime。
- `build_inputs(...)`（line 190）：从 family entry 造 `RayGenerationLaunchInputs`。

问题一：`__all__` 把 placement 模块的符号再导出：
```python
__all__ = [
    "RayGenerationLaunchInputs",
    "RayGenerationLauncher",
    "RayPlacement",                       # line 407
    "create_generation_placement_group",  # line 408
]
```
这两个符号 launcher 只是为了内部用而 `from vrl.generation.ray.placement import (...)`（line 21-24）。grep 确认全仓没有任何模块从 `launcher` 导入它们——只有 `placement.py` 定义、`launcher.py` 内部使用：
```
$ grep -rn "RayPlacement|create_generation_placement_group" vrl --include="*.py" | grep -v ray/placement.py | grep -v ray/launcher.py
（无输出）
```

问题二：config 取值 helper 重复。`_cfg_get`（line 389）与 `_cfg_path`（line 380）和这些文件里的实现几乎逐字相同：
```
vrl/ray/resources.py:900           def _cfg_get
vrl/trajectory/storage.py:151      def _cfg_get
vrl/generation/ray/config.py:277   def _config_get   # 同语义，名字不同
vrl/rollouts/collector/config.py:160 def _cfg_get
vrl/trainers/checkpointing.py:490/512 def _cfg_path / _cfg_get
vrl/rollouts/collector/artifacts.py:107 def _cfg_get
vrl/models/runtime_config.py:167   def _config_get
vrl/rollouts/collector/core.py:214 def _config_get
```
同一个 "duck-typed get from DictConfig / Mapping / attr" 逻辑被抄了近 10 份，命名还在 `_cfg_get` / `_config_get` 之间分裂。

## 2. 质疑点 / 改进机会
- 死对外表面（非死代码本身）：`__all__` 第 407-408 行的 `RayPlacement` / `create_generation_placement_group` 是 placement 模块的东西，从 launcher 二次导出没有 facade 价值——调用方该直接 `from vrl.generation.ray.placement import ...`。证据：launcher.py:21-24 内部 import，launcher.py:407-408 再导出，外部零引用。
- helper 重复 + 命名分裂：launcher.py:380/389 的 `_cfg_path`/`_cfg_get` 与 config.py:277 的 `_config_get` 是同一逻辑两个名字，且和全仓 ~8 处重复。属于 AGENTS.md "thin function 只在移除真实复杂度时保留"——这里复杂度是真的（DictConfig/Mapping/attr 三态取值），但应是一份共享实现，不是每文件一份。
- 不是 god-file：launch / launch_from_cfg / build_inputs 都围绕"把 cfg+entry 变成可跑的 Ray runtime"，职责单一，不拆。

## 3. 建议动作
1. 删掉 launcher.py `__all__` 里的 `"RayPlacement"` 和 `"create_generation_placement_group"`（line 407-408），只保留 `RayGenerationLaunchInputs` / `RayGenerationLauncher`。launcher 仍可内部 import placement 符号自用，无需对外导出。
2. 把 `_cfg_get` + `_cfg_path`（以及 config.py 的 `_config_get`、resources/collector/trainers/trajectory 各处副本）收敛到一个共享工具，例如 `vrl/utils/config_access.py` 暴露统一的 `cfg_get(node, key, default)` 与 `cfg_path(cfg, dotted, default)`，各文件改为 import。命名统一成一个（建议 `cfg_get`，丢弃 `_config_get` 别名）。本 sprint 只需在 launcher 内落地 import 替换；跨文件收敛可作为独立后续项，但至少消除 launcher.py 与 config.py 两个紧邻文件里的同义双胞胎。

注：第 2 项是跨文件收敛，范围大；若只做本文件级最小动作，则先删第 407-408 行的死再导出（零风险），并在 sprint 中标注 helper 收敛为 follow-up。

## 4. 不动什么 / 为什么不是过度清理
- `launch` / `launch_from_cfg` / `build_inputs` 三方法保留，职责内聚，不拆不合并。
- `_cross_node_preflight`（line 263）、`_validate_worker_gpu_ids`（line 239）是真实的 fail-fast 预检逻辑（带详细注释说明为何在 ray.init 之后跑），保留。
- `_import_from_path` / `_runtime_build_payload` / `_build_gatherer` 等是 build_inputs 的私有拆解步骤，可读性好，不内联。
- 不要为省 LOC 把 `_dtype_to_string` / `_device_to_string` 这类单行 helper 拍平——它们在 payload 序列化点有调用，留着无害（虽薄但属同形小工具）。呼应 AGENTS.md "consistency over cleanup"。

## 5. 验证
- 删 `__all__` 两项后：`grep -rn "from vrl.generation.ray.launcher import" vrl tests` 确认无人导入这两个符号（已确认为空）；跑 `tests/generation/ray/test_rollout_launcher.py`、`tests/generation/ray/test_runtime_config.py`、`tests/rollouts/test_runtime_inputs.py` 全绿。
- helper 收敛后：`ruff check vrl/generation/ray`，并跑上述三个测试 + `tests/generation/ray/`。
- `grep -rn "def _config_get\|def _cfg_get" vrl` 确认重复份数下降。
