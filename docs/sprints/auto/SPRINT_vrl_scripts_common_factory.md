# SPRINT(auto): vrl/scripts/common/factory.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/common/factory.py` (332 LOC)
角色判定: core
结论: improve

## 0. 一句话
factory 本体是真核心（从 YAML 派生 algorithm/evaluator/reward/collector 的唯一装配点，值得保留），但它的公开 API 面过宽、且有一个同名 re-wrap 函数造成命名遮蔽，需要收窄。

## 1. 现状（读代码得出）
按 `algorithm.kind` 分派构造 algorithm + evaluator 对，是整条 online recipe 的装配中枢：

```python
def build_algorithm_and_evaluator_from_cfg(cfg, *, family=None, built=None, ...):
    kind = str(OmegaConf.select(cfg, "algorithm.kind", default=""))
    if kind == "grpo": ...
    if kind == "token_grpo": ...
    if kind == "token_grpo_multisegment": ...
    if kind == "diffusion_nft": ...
    if kind == "diffusion_dpo": raise UnsupportedOnlineRecipeError(...)
```

`build_online_recipe_components` 把 reward/collector_config/algorithm/evaluator 一次性装进 `OnlineRecipeFactoryOutput`，被 `online.py` 调用。

## 2. 质疑点 / 改进机会
- 同名函数遮蔽（命名）：本文件定义 `build_rollout_config_from_cfg`（line 65），内部又 `from vrl.rollouts.collector.config import build_rollout_config_from_cfg as _build_rollout_config_from_cfg`（line 16-18）。本地这层只是把 `family` 归一成 `entry.family` 再转发：

  ```python
  def build_rollout_config_from_cfg(cfg, family=None):
      entry = _entry_from_family(cfg, family)
      return _build_rollout_config_from_cfg(cfg, family=entry.family)
  ```

  这是一个薄 wrapper，唯一价值是接受 `str | RolloutFamilyEntry | None`。它只被本文件内部三处调用（line 156/250/272），不被外部使用（`grep` 确认 `scripts/` 下只有本文件和 `__init__` 引用）。可以并进 `_entry_from_family` 的调用点，或保留但**改名**（如 `_rollout_config_for_entry`）以消除与上游同名符号的遮蔽——同名 import-as-`_x` 是坏味道，读代码时要回去看 alias 才知道哪个是哪个。
- 公开 API 面过宽：`__all__` 导出 9 个名字，但跨包消费者（5 个 train.py）只用 `online.py`，而 `online.py` 只 import `build_collector_from_cfg`、`build_online_recipe_components`。其余 `build_reward_from_cfg`、`build_algorithm_and_evaluator_from_cfg`、`build_rollout_config_from_cfg`、`resolve_online_family`、`OnlineRecipeFactoryOutput`、`UnsupportedOnlineRecipeError` 都只在本文件内部互相调用，没有外部消费者（`grep -rn` 全仓库确认）。这些是实现细节，不该全挂在 `__all__` 上当公共 API。
- `AlgorithmEvaluatorPair`（line 30-35）只是 `build_algorithm_and_evaluator_from_cfg` 的返回容器，未进 `__all__`、未被外部用——这是合理的私有 typed 返回，保留即可（不 flag）。

## 3. 建议动作
- 给本地 `build_rollout_config_from_cfg` 改名为 `_rollout_config_for_entry`（私有，只内部三处用），消除与 `vrl.rollouts.collector.config.build_rollout_config_from_cfg` 的同名遮蔽；同时 import 不再需要 `as _build_rollout_config_from_cfg` 的别名。
- 收窄 `__all__`：只保留 `online.py` 真正跨模块需要的 `build_online_recipe_components`、`build_collector_from_cfg`，以及作为 typed 输出契约的 `OnlineRecipeFactoryOutput` 和异常类型 `UnsupportedOnlineRecipeError`。其余 `build_reward_from_cfg`、`build_algorithm_and_evaluator_from_cfg`、`resolve_online_family` 仍是模块级函数可被内部调用，只是不再宣称为公共 API。
- 不拆文件：本文件是单一职责（online recipe 装配），不属于 god-file，不需要 consolidate/split。

## 4. 不动什么 / 为什么不是过度清理
- 按 `algorithm.kind` 的 if-链分派**不要**数据化成 dict 表：每个分支带不同的 isinstance 校验、不同的 evaluator 选择逻辑（如 `token_grpo` 还要按 `entry.collector.kind == "ar_continuous"` 二次分支），是真实复杂度，保持 if-链更可读可调试。
- `UnsupportedOnlineRecipeError` 是真边界（区分 online vs offline recipe，如 `diffusion_dpo` 明确拒绝），保留。
- `_with_resolved_reward_runtime_kwargs`、`_entry_from_family`、`_cfg_select`、`_collector_config_value` 都是有实质逻辑的私有 helper，不是纯转发，保留。
- 这不是 LOC 缩减，是消除命名遮蔽 + 收窄公共面，符合 AGENTS.md "consistency over cleanup"（分派形状保持一致）。

## 5. 验证
- 改完 `grep -rn "build_rollout_config_from_cfg" vrl/scripts/` 确认本地引用都指向新私有名，且不再有 `as _build_rollout_config_from_cfg` 别名。
- `grep -rn "resolve_online_family\|build_reward_from_cfg\|build_algorithm_and_evaluator_from_cfg" vrl/ | grep -v scripts/common/` 确认收窄 `__all__` 不影响任何外部调用（应为空）。
- `ruff check vrl/scripts/common/factory.py`。
- import 冒烟：`python -c "import vrl.scripts.common.online"`。
