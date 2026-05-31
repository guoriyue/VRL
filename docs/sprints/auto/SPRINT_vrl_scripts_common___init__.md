# SPRINT(auto): vrl/scripts/common/__init__.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/common/__init__.py` (34 LOC)
角色判定: package-init
结论: improve

## 0. 一句话
这个 `__init__` 把 13 个符号 re-export 成一个包级 facade，但全仓库没有任何代码从 `vrl.scripts.common` 这个包路径 import——所有消费者都直接 import 子模块，所以这层 facade 是空挂的。

## 1. 现状（读代码得出）
`__init__.py` 从三个子模块汇总并 re-export：

```python
from vrl.scripts.common.factory import (
    OnlineRecipeFactoryOutput,
    UnsupportedOnlineRecipeError,
    build_algorithm_and_evaluator_from_cfg,
    ...
)
from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import (
    OnlineRecipeDefinition,
    OnlineRecipeStack,
    RecipeDeviceContext,
)
__all__ = [ ... 13 个名字 ... ]
```

## 2. 质疑点 / 改进机会
- 包 facade 无人使用：`grep -rn "from vrl.scripts.common import\|import vrl.scripts.common\b" vrl/` 结果为空。所有 5 个 family train.py 都是从子模块直接 import：
  - `vrl/scripts/diffusion/wan_2_1/train.py:9` `from vrl.scripts.common.online import run_online_recipe`
  - `vrl/scripts/diffusion/wan_2_1/train.py:10` `from vrl.scripts.common.types import OnlineRecipeDefinition`
  - 同样模式见 `scripts/ar/janus_pro/train.py`、`scripts/ar/nextstep_1/train.py`、`scripts/diffusion/cosmos/train.py`、`scripts/diffusion/sd3_5/train.py`。
- 因此 facade 里 7 个 factory 符号（`build_*`、`OnlineRecipeFactoryOutput`、`UnsupportedOnlineRecipeError`、`resolve_online_family`）其实连内部都只被 `online.py` 直接从 `factory` import，根本不走这层 `__init__`。这层 re-export 是纯装饰，加字段时还要双份维护 `__all__`。

## 3. 建议动作
- 把 `__init__.py` 收缩成普通 package marker（保留 docstring，删掉所有 re-export 和 `__all__`），让消费者继续按现状从子模块 import。这与现有调用方完全一致，零改动它们。
- 或者，如果想保留一个"对外只暴露 recipe 入口"的窄 facade，那就只 re-export `run_online_recipe` + `OnlineRecipeDefinition`（外部真正需要的两个），删掉 7 个 factory 内部符号——它们是 `online.py` 的实现细节，不该出现在包 facade 上。推荐这个方案，因为它把"对外 API"和"内部 factory"分开。

## 4. 不动什么 / 为什么不是过度清理
- 不要去动 5 个 train.py 的 import 语句——它们直接走子模块是正确的、可 grep 的写法。
- `factory.py` / `online.py` / `types.py` 三个子模块本身是合理拆分，不在本 sprint 范围。
- 这不是"为省几行拍平"，而是删掉一个全仓库零引用、且会随子模块字段变更而腐烂的双份维护清单。

## 5. 验证
- 改完跑 `grep -rn "from vrl.scripts.common import" vrl/` 确认仍为空（没人依赖被删的 facade）。
- `ruff check vrl/scripts/common/` 确认无 F401 未使用 import。
- 跑任一 family 的训练入口 import 冒烟：`python -c "import vrl.scripts.diffusion.wan_2_1.train"`。
