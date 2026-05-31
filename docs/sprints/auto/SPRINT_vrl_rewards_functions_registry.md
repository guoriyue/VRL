# SPRINT(auto): vrl/rewards/functions/registry.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/functions/registry.py` (137 LOC)
角色判定: registry-config
结论: improve

## 0. 一句话
文件核心（`MultiReward` + `_register_builtins` 内置表）是 justified 的边界，但 `register_reward()` 是导出却零调用的死 public API——要么删，要么让 `_register_builtins` 真的用它。

## 1. 现状（读代码得出）
文件提供三件东西：
1. `_REWARD_REGISTRY` 名字->class 映射 + `register_reward` / `get_reward`（registry.py:16-28）。
2. `_register_builtins()` 把九个内置 reward 类塞进表（registry.py:31-52）——**直接 `_REWARD_REGISTRY.update({...})`，没有走 `register_reward`**。
3. `MultiReward`：加权组合多个 reward，带 `last_components` 追踪用于早发现 reward hacking（registry.py:55-138）。

## 2. 质疑点 / 改进机会
- **`register_reward` 是死 API**：`grep -rn "register_reward" .` 结果显示它只在 registry.py 被定义、在两个 `__init__.py` 被 re-export，**没有任何调用点**（连 `_register_builtins` 自己都不用它，而是直接 `.update(...)`）。tests 也只用 `get_reward`（`tests/rewards/test_multi.py:65,68`）。这是 AGENTS.md 意义上"无人调用的函数"。
- **内置表是 justified taxonomy，不是坏 ALL_CAPS**：`_register_builtins` 里的 `{"aesthetic": AestheticReward, ...}`（registry.py:42-52）是刻意维护的内置 reward 清单（schema-key 风格的 taxonomy 表），不是手抄某个 typed 结构的字段——保留，不需要 derive。
- **`MultiReward` 是真核心**：`from_dict`（registry.py:93-112）被 `vrl/scripts/common/factory.py:97` 用来按 config 装配，`last_components` 追踪有明确设计理由（docstring 注明防 reward hacking）。无可质疑。

## 3. 建议动作
二选一（推荐 A）：
- **A（删死码）**：删除 `register_reward`（registry.py:19-21）及其在 `vrl/rewards/__init__.py:6,25`、`vrl/rewards/functions/__init__.py:7,22` 的 re-export。grep 已确认零调用点，删除安全。
- **B（让它名副其实）**：若想保留可扩展注册入口，把 `_register_builtins` 的 `.update({...})` 改成逐个 `register_reward(name, cls)`，使该函数有真实调用路径并成为唯一写入口。

不要动 `_register_builtins` 的内置表结构、`MultiReward` 任何部分。

## 4. 不动什么 / 为什么不是过度清理
- 内置 reward 名->类映射表：justified taxonomy 边界，是 GRPO config `reward_fn` 名字的真实来源，保留（AGENTS.md "刻意隔离的 taxonomy 表"）。
- `get_reward`：被 `from_dict` 和测试使用，保留。
- `MultiReward.last_components` / `reset_components`：有防 reward-hacking 的设计理由，不是冗余状态。

## 5. 验证
- 选 A：删后跑 `grep -rn "register_reward" .` 应只剩零（无残留 import）；跑 `tests/rewards/test_multi.py`；`ruff check vrl/rewards`。
- 选 B：跑 `tests/rewards/test_multi.py` 确认 `get_reward("codex_image_qa")` 等仍解析（test_multi.py:65）。
