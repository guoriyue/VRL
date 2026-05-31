# SPRINT(auto): vrl/trajectory/storage.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trajectory/storage.py` (179 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件整体是真核心（runtime 存储策略，被 collector/metrics/executor 多处使用），但 `_VALID_DEVICES`/`_VALID_DTYPES` 两个 frozenset 手抄了 `Literal` 类型的成员，违反 AGENTS.md「ALL_CAPS 不应手抄 typed 结构」规则，应从 `Literal` derive。

## 1. 现状（读代码得出）
模块顶部先定义两个 `Literal` 别名，紧接着又手写两个 frozenset 重复同一份枚举：

```python
TrajectoryStorageDevice = Literal["preserve", "cpu"]
TrajectoryStorageDType = Literal["preserve", "float32", "float16", "bfloat16"]

_VALID_DEVICES = frozenset({"preserve", "cpu"})
_VALID_DTYPES = frozenset({"preserve", "float32", "float16", "bfloat16"})
```
（storage.py:13-17）

`__post_init__` 用这两个 frozenset 做校验：

```python
if self.device not in _VALID_DEVICES:
    raise ValueError(... f"{sorted(_VALID_DEVICES)}, got {self.device!r}")
if self.dtype not in _VALID_DTYPES:
    raise ValueError(... f"{sorted(_VALID_DTYPES)}, got {self.dtype!r}")
```
（storage.py:27-37）

另外 `_torch_dtype` 里第三处又手写了一份 dtype 名 → torch dtype 的映射 dict（storage.py:136-140），其 key 集合同样必须和 `TrajectoryStorageDType` 保持一致。

## 2. 质疑点 / 改进机会
- **ALL_CAPS 手抄 typed 结构（rule 1）**：`_VALID_DEVICES`/`_VALID_DTYPES` 不是真边界（不是 schema key / env var / checkpoint 名 / 协议名），只是把上面两行 `Literal` 的成员又抄了一遍。源 `Literal` 加一个 dtype（如 `"float8"`）时，这两个 frozenset 不会自动跟上，校验会悄悄放过/拒错——典型的「源类型加字段时悄悄腐烂」。证据：storage.py:13-17。
- 同一份 dtype 枚举在文件里出现三次：`Literal`（14）、`_VALID_DTYPES`（17）、`_torch_dtype` 的 dict key（136-140）。三处必须人工同步，没有单一真相源。

## 3. 建议动作
- 删除 `_VALID_DEVICES`/`_VALID_DTYPES` 两个手写 frozenset，改为从 `Literal` derive：
  ```python
  from typing import get_args
  _VALID_DEVICES = frozenset(get_args(TrajectoryStorageDevice))
  _VALID_DTYPES = frozenset(get_args(TrajectoryStorageDType))
  ```
  这样 `Literal` 仍是唯一真相源，校验和报错信息自动跟随。保留 frozenset 变量名本身（`__post_init__` 与报错信息引用它，且 `sorted(...)` 输出友好提示），只是改成 derive。
- `_torch_dtype` 的映射 dict 是「dtype 名 → torch 对象」的真实映射（torch 是 lazy import 边界，不能放进 `Literal`），属于 rule 1 允许的「真实需要的查找表」，可保留；但可在该 dict 上加一个断言或注释，说明其 key 必须覆盖 `_VALID_DTYPES \ {"preserve"}`，避免新增 dtype 时漏改。

## 4. 不动什么 / 为什么不是过度清理
- `TrajectoryStoragePolicy` dataclass、`apply_trajectory_storage_policy`、`trajectory_storage_policy_from_cfg`、`trajectory_tensor_bytes` 全部是真核心，被 `vrl/rollouts/collector/{batch_builder,core,artifacts}.py`、`vrl/generation/diffusion/{metrics,executor}.py` 广泛调用，不动。
- `_cfg_get` / `_to_builtin` 是 OmegaConf 适配边界（framework adapter），justified，不动。
- `_is_torch_tensor` 的 try/except ImportError 是 lazy import 边界，不动。
- 不要为省两行把 frozenset 整个删掉改成 inline `get_args(...) in` 判断——保留命名常量更利于报错信息与 grepability（consistency over cleanup）。本次只把「数据来源」从手抄改成 derive。

## 5. 验证
- `grep -n "_VALID_DEVICES\|_VALID_DTYPES" vrl/trajectory/storage.py` 确认仍只在校验处引用。
- 构造非法 policy（如 `TrajectoryStoragePolicy(device="cuda")`）应仍抛 `ValueError` 且报错信息列出 derive 出的合法值。
- `ruff check vrl/trajectory/storage.py` 通过（确认 `get_args` import 被使用）。
- 跑现有 trajectory storage 相关测试：`grep -rl "trajectory_storage\|TrajectoryStoragePolicy" tests/ 2>/dev/null` 找到对应测试后执行。
