# SPRINT: config schema 未知 key 警告

状态：parked / future。从已完成的 `SPRINT_yaml_config_cleanup.md` §4b 抽出。

决策记录（2026-06-10，owner 拍板）：
1. **单一行为，写死，不给任何人选模式**。「ignore / warn / forbid 三选一」的开关
   本身就会成为新的困惑配置项——正是 yaml 审计刚清理掉的那一类。
2. **ignore 消灭**（eval.* 死树的成因，无合理场景）；**forbid 永不实现**（挡实验
   key，owner 明确不要）。warn 是唯一行为，因此它不是「模式」、没有名字、没有开关。
3. **命名**：基类叫 `ConfigBase`（平实名词，不带机制词）。ignore/warn/forbid 只是
   pydantic 内部参数名，不出现在任何用户可见的地方；用户唯一感知是「写错 key 会
   看到一行指名道姓的警告」。

## 0. 一句话

`vrl/config/schema.py` 有 10 处 `model_config = ConfigDict(extra="ignore")`：
不认识的 YAML key **静默通过**。这是 2026-06 审计发现整棵 `eval.*` 死树（14 个
key 在所有实验里存活多年、零读者、零报错）能存在的唯一原因。修法：未知 key
**照常通过但打 warning**。

## 1. 「已知 key 清单」不需要维护——schema 字段声明就是清单

不存在「逐个登记已知 key」的工程。pydantic 模型的字段声明即全集；
`extra="allow"` 时声明外的 key 自动收进 `model_extra`。约 10 行的共享基类：

```python
class ConfigBase(BaseModel):
    """All config models inherit this: unknown YAML keys load fine
    but are reported loudly (typo / dead-key guard)."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _report_unknown_keys(self):
        if self.model_extra:
            logger.warning(
                "unknown config keys under %s (typo? dead?): %s",
                type(self).__name__, sorted(self.model_extra),
            )
        return self
```

10 个模型把 `extra="ignore"` 换成继承 `ConfigBase` 即可。新增字段照常声明，
清单自动更新，零额外维护。

## 2. 行为对照（为什么这是唯一行为，而非可选模式）

```text
ignore（现状）: 拼错/死 key 静默吞掉 —— eval.* 死树的成因,消灭
forbid        : 拼错被拦 ✅,但加实验 key 也被拦 ❌ —— 永不实现
报告并放行    : 实验 key 照样跑、拼错 key 日志里大声一行 ✅✅ —— 写死为唯一行为
```

不设模式开关；也不需要「存量白名单」——存量未知 key 只是被报出来，不挡路，
按日志慢慢清。

## 3. 实施（一步到位，无需分期）

- `vrl/config/schema.py` 加 `ConfigBase` 基类（用 `vrl/utils/logging.init_logger`）。
- 10 处 `ConfigDict(extra="ignore")` 模型改继承它；删第 5 行的 migration 注释。
- 测试：加载一个带未知 key 的 cfg → 断言 warning 文本含该 key 名；
  正常实验全量加载 → 记录当前会报出的存量未知 key 清单（即待清理清单）。

## 4. 验收

```text
pytest tests/config/ 全绿
实验 YAML 加一个拼错 key（如 num_stps）→ 加载时日志出现 warning 指名道姓
实验 YAML 加一个有意的新实验 key → 照常运行,仅 warning
```

## 5. 参考

- `vrl/config/schema.py`（extra="ignore" ×10；第 5 行的 migration 注释）
- 审计来源：yaml-config-audit workflow `wf_84e8047e-fcb`（254 key / 30 findings）
- pydantic `model_extra`: https://docs.pydantic.dev/latest/concepts/models/#extra-data
