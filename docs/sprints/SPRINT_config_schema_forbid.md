# SPRINT: config schema extra="forbid" 迁移（parked）

状态：parked / future。从已完成的 `SPRINT_yaml_config_cleanup.md` §4b 抽出——
那次审计的根因修复，刻意分阶段，等触发条件再做。

## 0. 一句话

`vrl/config/schema.py` 有 10 处 `model_config = ConfigDict(extra="ignore")`：
不认识的 YAML key **静默通过**。这是 2026-06 审计发现整棵 `eval.*` 死树（14 个
key 在所有实验里存活多年、零读者、零报错）能存在的唯一原因。schema.py 第 5 行
自己写着 ignore 是「during migration」的临时态——临时态已经变成永久态。

## 1. 为什么不直接翻 forbid

不先清存量直接翻，会一次性炸出大量无关失败（census 提示 model 域有一批
「被 pydantic 静默吞掉的 implicit defaults」）。顺序不能反。

## 2. 分两步做

### T1 先加 CI 守门测试（低风险，可立即做）
- 对照测试：resolved experiment config 与 schema 字段集 diff，**新增**未知 key
  即 fail（存量白名单豁免）。放 `tests/config/`。
- 跑稳一个迭代周期，期间白名单只许减不许增。

### T2 翻 forbid + 清存量
- 10 处 `extra="ignore"` 统一改 `extra="forbid"`。
- 修复翻转暴露的存量未知 key（逐个判定：真死 key 删 YAML；漏建模的字段补进 schema）。
- 删掉 T1 的白名单机制（forbid 本身就是守门）。

## 3. 验收

```text
pytest tests/config/ 全绿
任何实验 YAML 里加一个拼错的 key（如 num_stps）→ 配置加载立刻报错
```

## 4. 参考

- `vrl/config/schema.py`（extra="ignore" ×10；第 5 行的 migration 注释）
- 审计来源：yaml-config-audit workflow `wf_84e8047e-fcb`（254 key / 30 findings）
