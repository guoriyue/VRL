# SPRINT(auto): vrl/scripts/ar/janus_pro/train.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/ar/janus_pro/train.py` (108 LOC)
角色判定: script
结论: improve

## 0. 一句话
文件本身是合法的 family entrypoint 模块（与 nextstep_1 跨家族形状一致，应保留），但其中 `train_janus_pro_grpo` 是无人引用的死 export，且与 `train_janus_pro_ocr_grpo` 字节级重复，应删除。

## 1. 现状（读代码得出）
这是 Janus-Pro 的在线 GRPO 训练 entrypoint 模块。它声明 4 个 public async 入口，统一委托给本地 `_run_janus_recipe`，后者构造 `OnlineRecipeDefinition` 并交给 `vrl.scripts.common.online.run_online_recipe`：

```python
async def train_janus_pro_grpo(cfg: DictConfig) -> None:
    await _run_janus_recipe(cfg, family="janus_pro")

async def train_janus_pro_ocr_grpo(cfg: DictConfig) -> None:
    await _run_janus_recipe(cfg, family="janus_pro")
```
(train.py:14-23)

entrypoint 解析方式是 YAML 里 `trainer.entrypoint: module:function`，由 `vrl/scripts/train.py:32` 动态 `import_module` + `getattr` 调用，因此一个 public 函数"被用到"的唯一证据是某个 config/test 引用了它的全名。

## 2. 质疑点 / 改进机会
1. **`train_janus_pro_grpo` 是死代码（无引用）**。grep 全仓 `*.py`/`*.yaml`：
   - `train_janus_pro_ocr_grpo` → `configs/experiment/ar/janus_pro/online_grpo_ocr.yaml:19`
   - `train_janus_pro_r1_ocr_grpo` → `online_r1_grpo_ocr.yaml:14` + `tests/config/test_janus_pro_r1_config.py:14`
   - `train_janus_pro_r1_codex_qa_grpo` → `online_r1_grpo_codex_qa.yaml:14` + 同 test
   - `train_janus_pro_grpo` → **0 处**（除自身定义和 `__all__`）。
   它没有任何 config、test 或代码引用，YAML 动态 dispatch 模型下等于不可达。

2. **`train_janus_pro_grpo` 与 `train_janus_pro_ocr_grpo` 完全重复**。两者 body 都是 `await _run_janus_recipe(cfg, family="janus_pro")`（train.py:14-23），仅函数名/docstring 不同。即便日后需要它，也只是 OCR 入口的别名，留着会让人误以为存在第二条 pipeline。

3. （非问题，记录）`_build_bundle`/`_build_replay_bundle` 里的 `if family == "janus_pro_r1": spec.task_variant = "ar_t2i_r1"`（train.py:68-70, 86-88）是 janus 比 nextstep_1 多出的真实分支逻辑，解释了为何这里 `_run_janus_recipe` 要带 `family` 参数、而 nextstep_1 不需要——这是合理差异，不是过度抽象。

## 3. 建议动作
- 删除 `train_janus_pro_grpo`（train.py:14-17）。
- 从 `__all__` 移除 `"train_janus_pro_grpo"`（train.py:103）。
- 不要为"对称/以后可能用"保留它；YAML 入口是按需新增的，需要时直接加回一行别名成本极低。
- grep 证据：`grep -rn "train_janus_pro_grpo\b" --include=*.py --include=*.yaml .` 仅命中本文件定义行，确认安全删除。

## 4. 不动什么 / 为什么不是过度清理
- 不要拍平 `_run_janus_recipe` / `_build_bundle` / `_build_replay_bundle` / `_configure_trainer` / `_export_modules` 这套 helper。它们与 `vrl/scripts/ar/nextstep_1/train.py` 的同名 helper 形成跨家族统一形状（同样的 5 个 hook、同样的 `OnlineRecipeDefinition` 装配方式），符合 AGENTS.md "consistency over cleanup"——保留 grepability 和家族间可对照性。
- 不要把 4 个 entrypoint 合并成一个带参数的函数：它们是 YAML 按名引用的 public 边界，名字即契约。
- `r1` 的两个入口（ocr / codex_qa）虽然 body 相同（都走 `family="janus_pro_r1"`），但各自被独立 config 引用，是两条真实实验 pipeline 的命名锚点，保留。
- `__init__.py` 仅一行 package docstring，属 package-init，无需改动。

## 5. 验证
- 删后跑 `pytest tests/config/test_janus_pro_r1_config.py`（它只引用 r1 两个入口，不应受影响）。
- `grep -rn "train_janus_pro_grpo" .` 应只剩本 sprint 文档。
- `ruff check vrl/scripts/ar/janus_pro/train.py` 确认无 F401/未用 import（`LORA_WEIGHTS_NAME` 仍被 `_export_modules` 使用，不受影响）。
