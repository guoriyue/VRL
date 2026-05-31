# SPRINT(auto): vrl/trainers/data/artifacts.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/data/artifacts.py` (373 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件本身是合法核心（manifest 校验 + 路径解析），但 `repo_root()` / `default_data_root()` 与 `vrl/scripts/data/common.py` 里同名函数逐字重复，应消除重复，统一一个来源。

## 1. 现状（读代码得出）
本文件定义了 manifest 校验管线（`validate_artifact_manifest` / `validate_artifact_manifest_pair` / `validate_source_backed_video_world_manifest_pair`）、路径解析（`resolve_artifact_path`）、报告写出（`write_manifest_report`），都是真实业务逻辑且被多处引用（`vrl/config/validation.py`、`vrl/scripts/data/video_world.py`、`vrl/scripts/diffusion/cosmos/train.py`、tests）。

问题在末尾两个小工具函数：

```python
# artifacts.py:85-97
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def default_data_root() -> Path:
    env_value = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root() / "data" / "external").resolve()
```

`vrl/scripts/data/common.py:18-26` 有逐字等价的实现：

```python
# common.py:18-26
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def default_data_root() -> Path:
    value = os.environ.get("VRL_DATA_ROOT", "").strip()  # ← 没用 DATA_ROOT_ENV 常量，硬编码
    if value:
        return Path(value).expanduser().resolve()
    return (repo_root() / "data" / "external").resolve()
```

## 2. 质疑点 / 改进机会
1. 重复实现：`repo_root` 与 `default_data_root` 在两个模块各写一份，行为必须保持一致却没有单一来源。grep 证据：
   - `vrl/scripts/data/setup.py:31,50` 从 `artifacts` 导入这两个函数；
   - `vrl/scripts/data/danbooru.py:21,26` 和 `bootstrap.py:18` 从 `common` 导入。
   同一仓库里两套 data-root 解析路径，将来改 `data/external` 默认值或环境变量回退逻辑时极易只改一处而腐烂。
2. `common.py:23` 把 `"VRL_DATA_ROOT"` 硬编码成字面量，而 `artifacts.py:14` 已经有 `DATA_ROOT_ENV = "VRL_DATA_ROOT"` 常量。env var 名是真边界，应当只有一个常量，不该在 `common.py` 里再抄一份字符串。

非问题（确认不是坏味道）：
- `DEFAULT_ARTIFACT_FIELDS`、`IMAGE_SUFFIXES`、`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS` 都是刻意隔离的 policy/taxonomy 表（哪些字段算 artifact、哪些后缀按图片校验、real Video2World manifest 必填的 provenance 字段），不是手抄某个 typed 结构的字段名，符合 AGENTS.md「deliberately isolated taxonomy/config table」保留条件。
- `DATA_ROOT_ENV` 是 env var 名，真边界，保留。

## 3. 建议动作
让 data-root 解析只有一个权威来源。建议把 `repo_root` / `default_data_root` / `DATA_ROOT_ENV` 收敛到一处，另一处改为转发导入：

- 方案 A（推荐）：把 `repo_root`/`default_data_root` 留在 `artifacts.py`（它持有 `DATA_ROOT_ENV` 常量且被 `config/validation.py` 这条非脚本路径依赖），在 `vrl/scripts/data/common.py` 里删除这两个本地实现，改成 `from vrl.trainers.data.artifacts import DATA_ROOT_ENV, default_data_root, repo_root` 后 re-export，保持 `common.__all__` 不变以免影响 `danbooru.py`/`bootstrap.py` 的导入点。
- `common.py` 的 `default_cache_dir()` 复用 `repo_root()`，迁移后仍可正常工作。

不要为了「去重」反向把 `artifacts.py` 依赖 `scripts/`（scripts 是工具层，不应被 `config/validation.py` 这种核心路径反向依赖）。

## 4. 不动什么 / 为什么不是过度清理
- 三个 taxonomy 常量保持原样，不要 derive、不要拍平——它们是策略表不是 typed 结构镜像。
- 校验函数族（含 `validate_source_backed_video_world_manifest_pair` 这个看似薄的 wrapper）保留：它通过固定一组 `artifact_fields` + 必填 metadata 给 real Video2World manifest 提供了一个有语义的命名入口，被 `video_world.py`、`config/validation.py`、tests 直接调用，属于跨调用点一致的 public facade，符合「consistency over cleanup」。
- `common.py` 的 `emit`/`write_jsonl`/`dedupe_text`/`write_report` 与本 sprint 无关，不动。

## 5. 验证
- grep 确认迁移后没有第二份实现：`grep -rn "def repo_root\|def default_data_root" vrl/` 应只剩 `artifacts.py` 各一处。
- grep 确认 env var 字面量唯一：`grep -rn '"VRL_DATA_ROOT"' vrl/` 应只剩 `artifacts.py:14`。
- 跑相关测试：`pytest tests/data/test_populate.py tests/data/test_artifact_manifest_validation.py tests/data/test_video_world_manifests.py -q`，并冒烟 `python -m vrl.scripts.data.setup --help` / `danbooru` / `bootstrap` 导入不报错。
- `ruff check vrl/trainers/data/artifacts.py vrl/scripts/data/common.py`。
