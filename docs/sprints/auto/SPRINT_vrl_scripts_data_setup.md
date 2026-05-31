# SPRINT(auto): vrl/scripts/data/setup.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/data/setup.py` (135 LOC)
角色判定: script
结论: question

## 0. 一句话
存在两个并行的 dataset CLI 入口——`setup.py` 和 `populate.py`——职责边界模糊：`setup anime` 与 `populate anime-prompts` 几乎调同一个 `build_anatomy_prompts`，需要确认这两个入口是不是该合并或明确分工。

## 1. 现状（读代码得出）
`setup.py` 提供两类命令：
- `anime --metadata-only`：直接转发 `danbooru.build_anatomy_prompts`（line 10-27），和 `populate.py` 里的 `anime-prompts` 子命令（`danbooru._cmd_anime_prompts`）目标重叠。
- `pickapic` / `anime-artifacts` / `video-world-artifacts`：`_setup_artifact_dirs`(49) 只是 `mkdir -p` 一组 artifact 目录，依赖 `vrl.trainers.data.artifacts` 的 `default_data_root`/`repo_root`。

被 `tests/data/test_artifact_manifest_validation.py:8` 以 `setup as setup_cli` 引用，所以**不是死代码**。

## 2. 质疑点 / 改进机会
**(a) 两个入口语义重叠**（规则 5 的反面：职责切分不清）。用户面对 `vrl.scripts.data.setup anime` 和 `vrl.scripts.data.populate anime-prompts`，无法从命名判断该用哪个。`setup` 名字泛（引擎味词），没传达"它管的是 artifact 目录创建 + metadata-only prompt 构建"这层语义。`populate` 才是真正下载/生成数据的。两者边界靠读代码才能分清。

**(b) `anime` 命令是 `populate anime-prompts` 的薄重复**（line 10-27）。两边都最终调 `build_anatomy_prompts(download_metadata=True, bucket_balance="quota", prompt_style="mixed")`，差别只是 setup 版本暴露了更多 `--train-output/--min-score` 等 flag。这是"同一动作两个入口"，应明确：要么 setup 只管 artifact 目录（mkdir），prompt 生成全归 populate；要么反过来。

**(c) 不确定点**：无法仅凭代码判定这是设计意图（setup=幂等环境准备 vs populate=拉数据）还是历史遗留重叠。故判 question 而非 consolidate——需要 owner 确认两入口的预期分工。

## 3. 建议动作
1. 先确认意图：`setup` 是否定位为"创建空 artifact 目录骨架"的幂等命令、`populate` 定位为"真正拉取/生成内容"。若是，则：
2. 从 `setup.py` 移除 `anime` 子命令（line 10-27, 101-116），prompt 生成统一走 `populate anime-prompts`；`setup` 只保留三个 artifact-dir 创建命令。
3. 考虑把 `setup` 重命名为更达意的名字（如 `init_dirs` / `scaffold`），消除与 `populate` 的概念碰撞。

## 4. 不动什么 / 为什么不是过度清理
- `_setup_artifact_dirs` 的目录创建逻辑是有价值的（artifact-backed manifest 需要预建目录），保留。
- 对 `vrl.trainers.data.artifacts` 的 lazy import（函数内 import）是刻意的 import 边界，保留。
- 在 owner 确认分工前不要硬删 `anime` 子命令——可能有外部脚本依赖该入口。

## 5. 验证
- 与 owner 确认 setup vs populate 分工后再动手。
- 若移除 `anime`：grep `scripts.data.setup` 确认无外部 argv 依赖该子命令；`pytest tests/data/test_artifact_manifest_validation.py -q` 仍绿。
- `ruff check vrl/scripts/data/setup.py`。
