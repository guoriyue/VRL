# SPRINT(auto): vrl/config/validation.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/config/validation.py` (229 LOC)
角色判定: helper
结论: improve

## 0. 一句话
文件本体是 builders/scripts 真实依赖的访问与校验边界，但里面 `validate_data_config` 和 `assert_no_missing` 是迁移后遗留的死代码（无人 import/调用），且 `validate_data_config` 还和 `schema.py` 的 `DataConfig._validate_data` 重复维护同一套 loader/sampler 校验，应删。

## 1. 现状（读代码得出）
文件顶部 docstring 自己声明结构校验已经迁移到 schema：
```
Whitelist sets (...) have moved to
schema.py. Structural validation now flows through parse_config() -> RootConfig.
```
真正活着的对外函数：`require` / `optional_none` / `path_exists` / `resolve_algorithm_kind` / `validate_reward_config` / `validate_training_config` / `validate_production_video_reward_config`，分别被 `builders.py`、`scripts/diffusion/wan_2_1/train_dpo.py`、各 tests 引用。

但 `validate_data_config`（115-149 行）与它复刻的那份 sampler 白名单：
```python
valid_samplers = {"random_without_replacement", "sequential_window"}
```
和 `assert_no_missing`（78-84 行）grep 全仓后只在本文件的定义处和 `__all__` 出现，没有任何调用点。

## 2. 质疑点 / 改进机会
- 死代码：`validate_data_config`（vrl/config/validation.py:115-149）和 `assert_no_missing`（vrl/config/validation.py:78-84）全仓零调用，仅在 `__all__`（vrl/config/validation.py:224, 219）re-export。grep 结果：两者除定义与 `__all__` 外无任何引用（含 tests）。
- 重复维护：`validate_data_config` 的 loader/preprocessing/sampler 必填逻辑与 `schema.py` 的 `DataConfig._validate_data`（vrl/config/schema.py:176-214）是同一套规则的两份手写实现，而 `DataConfig` 才是 `parse_config -> RootConfig.data` 真正跑到的路径。两份 `valid_samplers = {"random_without_replacement", "sequential_window"}`（validation.py:130 与 schema.py:188）是同一个手抄集合，源头改了另一边会悄悄腐烂。删掉死的 `validate_data_config` 即消除该重复，无需再做 derive。

## 3. 建议动作
- 删除 `validate_data_config`（115-149 行）及 `__all__` 中的 `"validate_data_config"`。
- 删除 `assert_no_missing`（78-84 行）及 `__all__` 中的 `"assert_no_missing"`。
- 删后检查 import：`DataConfig`（validation.py:21 引入）删完后是否还被 `validate_data_config` 之外用到——`valid_loaders = frozenset(get_args(DataConfig.model_fields["loader"].annotation))`（validation.py:121）只在 `validate_data_config` 内，因此 `DataConfig` import 也应一并移除。
- grep 证据：`grep -rn "validate_data_config\|assert_no_missing" --include="*.py" .` 仅命中本文件定义与 `__all__`，无外部调用。

## 4. 不动什么 / 为什么不是过度清理
- `require` / `optional_none` / `path_exists` 是 OmegaConf 缺失值语义上稳定的仓库级错误信息边界，被 builders 和脚本真实依赖，保留。
- `resolve_algorithm_kind` 的 `valid = frozenset(get_args(AlgorithmConfig.model_fields["kind"].annotation))`（validation.py:95）是从 typed `Literal` 派生的，符合 AGENTS.md「从源头 derive」要求，不要动。
- `_validate_video_world_source_report` 的 `required_keys` 集合（schema 里没有对应 typed 结构）是 Video2World provenance 报告的刻意隔离 taxonomy + 必须做文件存在性/IO 校验（D4 明确不能进 Pydantic schema），保留。
- 不要为「统一」把 `validate_production_video_reward_config` 也并进 schema——它做的是 `Path(value).exists()` 和读 JSON 文件，属于运行期 IO，schema 不应承担。

## 5. 验证
- `grep -rn "validate_data_config\|assert_no_missing" --include="*.py" .` 删后应只剩 0 命中（或仅 git 历史）。
- `pytest tests/config/ tests/rewards/test_video_reward.py -q` 应全绿（这些测试只用 `validate_training_config`/`validate_reward_config`/`parse_config`，不依赖被删函数）。
- `ruff check vrl/config/validation.py`（确认无未用 import 残留，尤其 `DataConfig`/`get_args`）。
