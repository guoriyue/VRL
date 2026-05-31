# SPRINT(auto): vrl/scripts/data/danbooru.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/data/danbooru.py` (1858 LOC)
角色判定: script
结论: improve

## 0. 一句话
1858 行的 god-file，混了 4 条不相关管线（anatomy prompts / safety prompts / positive-image fetch / hand-crop+hard-negative+reward-eval），夹带确证的死代码和一堆只为兼容旧名字的 alias，应拆分 + 删死代码。

## 1. 现状（读代码得出）
单文件承载了多条彼此独立的数据管线：
- anatomy prompt 生成：`build_anatomy_prompts` / `build_prompt_rows` / `prompt_from_tags` / `_select_quota_rows`（line 387, 903, 772, 1516）
- safety prompt 生成：`build_safety_prompts` / `build_danbooru_safety_prompt_rows` / `_safety_prompt_from_tags`（line 451, 1050, 1218）
- 正样本图片下载：`build_positive_images` / `download_danbooru_images` / `http_download`（line 510, 1243, 1285）
- hand-crop / hard-negative / label-queue / reward-eval：`hand_crop_rows` / `hard_negative_rows` / `label_queue_rows` / `_load_reward_rollouts` / `_metric_report`（line 1332, 1363, 1387, 1459, 1489）

文件顶部 ~300 行是 module-level ALL_CAPS taxonomy（tag 集合、bucket 权重、safety rating 别名），line 71-366。

## 2. 质疑点 / 改进机会
**(a) 确证死代码——reward-eval 三件套**（AGENTS.md 规则 6）。`_load_reward_rollouts`(1459)、`_metric_report`(1489)、`_pairwise_auc`(1501) 全仓库 grep 零引用（连自身文件、`__all__`、测试都不引用）：
```
grep -rn "_load_reward_rollouts\|_metric_report\|_pairwise_auc" --include=*.py .  # 仅命中本文件定义处
```
`_mean`(1497) 只被 `_metric_report` 用，连带死亡。这是把"reward 验证 spike"留在了长期 import graph 里——属于 one-shot 混入（规则 4）。建议删除。

**(b) 一组 back-compat alias 几乎全死**（line 1697-1707）：
```python
select_quota_rows = _select_quota_rows
interleave_bucket_rows = _interleave_bucket_rows
proportional_group_counts = _proportional_group_counts
proportional_text_group_counts = _proportional_text_group_counts
interleave_manifest_rows = _interleave_manifest_rows
_http_download = http_download
_count_lines = count_jsonl_rows
```
grep 结果：`select_quota_rows`/`interleave_bucket_rows`/`proportional_*`/`interleave_manifest_rows`/`_count_lines` 在 tests 与源码里**零引用**，却都进了 `__all__`（line 1830-1842）。只有 `_http_download` 被 `tests/data/test_populate.py:146` 通过 `monkeypatch.setattr(danbooru, "_http_download", ...)` 用到，且 `_current_http_download()`(1706) 还从 `globals()` 里取它——这是为了让测试能 monkeypatch 而保留的间接层。建议：删掉无人用的 5 个 public alias（含 `__all__` 条目）和 `_count_lines`；`_http_download` 间接层可保留但应加注释说明它存在只为 test monkeypatch。

**(c) god-file 职责过载**（规则 5）。四条管线本可拆为 `danbooru/anatomy.py`、`danbooru/safety.py`、`danbooru/images.py`，共享 `iter_metadata`/`normalize_tags`/`record_*` 放进 `danbooru/_metadata.py`。当前 1858 行使 grep/调试/review 都困难。

**(d) ALL_CAPS taxonomy 是 justified 的**（规则 1 例外）。`SAFETY_RISK_TAGS`/`EXCLUDE_TAGS`/`POSE_TAGS`/`DEFAULT_BUCKET_WEIGHTS` 等是"刻意隔离的 taxonomy 表"，不是手抄某 typed 结构的字段名，符合保留条件。不要 derive、不要 flag。

## 3. 建议动作
1. 删除死代码：`_load_reward_rollouts`、`_metric_report`、`_mean`、`_pairwise_auc`（line 1459-1513）。grep 已确认零引用。
2. 删除无人用的 alias 及对应 `__all__` 条目：`select_quota_rows`、`interleave_bucket_rows`、`proportional_group_counts`、`proportional_text_group_counts`、`interleave_manifest_rows`、`_count_lines`（line 1697-1701, 1703）。
3. 保留 `_http_download` + `_current_http_download`，补一行 docstring 说明用途是 test monkeypatch hook。
4. （较大动作，可单独 PR）按管线拆分为 `danbooru/` 子包，taxonomy 表集中到一个 `danbooru/taxonomy.py`。

## 4. 不动什么 / 为什么不是过度清理
- 所有 ALL_CAPS tag/bucket/rating 表保留——它们是领域 taxonomy，符合 AGENTS.md 例外。
- `register`/`main`/`build_default_manifests` 是 CLI 入口边界，保留。
- `iter_metadata` 的多格式分发（tar.gz / gz / json / jsonl）是真实复杂度抽象，保留。
- 拆分是可选的大动作；删死代码与死 alias 是低风险高收益，优先做。呼应 "consistency over cleanup"：跨脚本一致的 `register()` 形状不动。

## 5. 验证
- `grep -rn "_load_reward_rollouts\|_metric_report\|_pairwise_auc\|select_quota_rows\|interleave_bucket_rows\|proportional_group_counts\|interleave_manifest_rows\|_count_lines" --include=*.py .` 删除后应只剩零命中（除被保留者）。
- `pytest tests/data/test_danbooru.py tests/data/test_populate.py -q` 全绿（test_populate 依赖 `_http_download`，保留即可）。
- `ruff check vrl/scripts/data/danbooru.py`。
