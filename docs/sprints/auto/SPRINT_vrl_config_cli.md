# SPRINT(auto): vrl/config/cli.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/config/cli.py` (33 LOC)
角色判定: dead
结论: delete

## 0. 一句话
`parse_and_load` 想做「所有 YAML 训练脚本共享的 `--config` + dotlist argparse 入口」，但全仓没有任何脚本调用它——每个脚本都各自 `build_parser()` 手搓 argparse，这个共享 helper 是没接上的死代码。

## 1. 现状（读代码得出）
文件只有一个函数：
```python
def parse_and_load(default_config: str, description: str) -> DictConfig:
    """Build the standard ``--config`` + dotlist-overrides argparse parser ..."""
    ...
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, ...)
    parser.add_argument("overrides", nargs="*", ...)
    args = parser.parse_args()
    return load_config(args.config, overrides=args.overrides)
```
它被 `vrl/config/__init__.py` re-export（`__all__ = [..., "parse_and_load"]`）。

## 2. 质疑点 / 改进机会
- 死代码：`grep -rn "parse_and_load" --include="*.py" .` 去掉 `vrl/config/` 后 0 命中。它只在定义处和 `__init__.py` 的 re-export 出现，没有任何真实调用者。
- 意图落空：脚本本应复用这个共享入口，但实际上 `vrl/scripts/train.py:58`、`vrl/scripts/data/populate.py:20`、`vrl/scripts/data/setup.py:97`、`vrl/scripts/diffusion/cosmos/anima/generate.py:32` 等每个脚本都自己 `def build_parser()` 各搭一套 `argparse.ArgumentParser` 并直接调 `load_config(...)`。共享 helper 从未被接上，等于一个声明了却没人用的「标准入口」。
- 顺带：`logging.basicConfig(...)` 藏在这个 helper 里（cli.py:17-20），即使有人将来想用它，也会附带一个副作用式的全局日志配置，和「parse + load」职责混在一起。

## 3. 建议动作
- 删除整个 `vrl/config/cli.py`。
- 从 `vrl/config/__init__.py` 移除 `from vrl.config.cli import parse_and_load` 和 `__all__` 里的 `"parse_and_load"`。
- grep 证据：`grep -rn "parse_and_load" --include="*.py" .` 仅命中 `vrl/config/cli.py`（定义）与 `vrl/config/__init__.py`（re-export），无外部调用者，删除安全。
- 备选（更弱）：如果团队确实想统一脚本入口，应当反过来——让现有脚本真的调用它并删掉各自的 `build_parser`。但当前状态是「无人用」，按 AGENTS.md「死代码 -> delete」直接删更干净；要统一可另起 sprint。

## 4. 不动什么 / 为什么不是过度清理
- 不要顺手去改各脚本里的 `build_parser()`——它们当前能跑、是脚本各自的 entrypoint 形状，统一入口是单独议题，不在本次「删死代码」范围。
- `vrl/config/loading.py` 的 `load_config` 是真正被广泛使用的公共 API，保留不动。

## 5. 验证
- 删后 `grep -rn "parse_and_load" --include="*.py" .` 应 0 命中。
- `python -c "import vrl.config"` 不报错（确认 `__init__.py` 没有悬空 import）。
- `pytest tests/config -q` 全绿。
