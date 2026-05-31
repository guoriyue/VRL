# SPRINT(auto): vrl/scripts/data/bootstrap.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/data/bootstrap.py` (115 LOC)
角色判定: script
结论: improve

## 0. 一句话
`for-experiment` 入口本身是好的（用真 config loader 解析实验需要哪些 manifest），但 `_MANIFEST_POPULATE_HINTS` 是一张手维护的"路径前缀 → populate 命令"映射，会和 `populate.py`/`danbooru.py` 里真正注册的子命令悄悄脱钩。

## 1. 现状（读代码得出）
模块级硬编码映射（line 20-23）：
```python
_MANIFEST_POPULATE_HINTS = (
    ("datasets/danbooru/anatomy/", "python -m vrl.scripts.data.populate anime-prompts"),
    ("data/external/video_world/", "python -m vrl.scripts.data.populate video-world-bridge"),
)
```
`_populate_hint_for_path`(88) 按路径前缀线性匹配，匹配不到就回退到一句话提示。`resolve_experiment_dataset_plan`(34) 是干净的纯函数，逻辑没问题。

## 2. 质疑点 / 改进机会
**(a) hint 表是脆弱的字符串耦合**（AGENTS.md 规则 1 精神）。这张表把"manifest 输出路径"和"生成它的 populate 子命令"用裸字符串前缀绑定，但：
- 真正的输出路径常量在 `danbooru.py`（`ANATOMY_TRAIN_OUTPUT = ANATOMY_DIR / "train_prompts.jsonl"`，line 39）；
- pickapic 分支甚至没走这张表，而是在 `resolve_experiment_dataset_plan` 里另写死了一条 `"python -m vrl.scripts.data.populate pickapic --with-images"`（line 74）。

也就是说"某 manifest 怎么生成"这个知识被劈成两半：一半在 hint 表、一半在函数体内联，且都是手抄的 magic string。源头改子命令名或输出路径时，这里不会报错只会给出过时命令。

**(b) `--run` 分支只认 pickapic**（line 106-112）：
```python
if step["get"].startswith("python -m vrl.scripts.data.populate pickapic"):
    from vrl.scripts.data.populate import main as populate_main
    populate_main(["pickapic", "--with-images"])
```
其它 manifest 即便缺失、`--run` 也不会真跑，等于半实现的 feature——要么补全 dispatch（解析 hint 命令并转发给 `populate.main`），要么明说 `--run` 仅支持 pickapic。

## 3. 建议动作
1. 让 hint 不再手抄：把"manifest 路径 → populate 命令"的映射收敛到一处。最稳妥是让 populate 子命令在 `register` 时声明它产出的 manifest 路径（或反向：从 `danbooru.py` 的输出常量 derive 前缀），bootstrap 从该来源构建查找表，而不是另存一份字符串。
2. 把 pickapic 的内联命令也并入同一映射来源，消除 line 74 的特例。
3. `--run` 要么通用化（把 `step["get"]` 解析成 argv 转发 `populate.main`），要么在 `--help`/docstring 里明确只支持 pickapic。

## 4. 不动什么 / 为什么不是过度清理
- `resolve_experiment_dataset_plan` 纯函数 + `_count_rows` 保留，设计清晰。
- `register()` 的 subparser 形状与 sibling 脚本一致，保留（跨家族一致性优先）。
- 这不是要删文件——`for-experiment` 是有价值的 user-facing front door；只是修 hint 的数据来源。

## 5. 验证
- 改完后人工跑 `python -m vrl.scripts.data.populate for-experiment <某实验>`，确认 `get` 命令与 `populate.py` 实际可用子命令一致。
- `pytest tests/data -q`、`ruff check vrl/scripts/data/bootstrap.py`。
