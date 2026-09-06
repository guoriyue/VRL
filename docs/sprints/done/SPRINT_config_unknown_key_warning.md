# SPRINT: config 未知 key 警告（implemented）

状态：implemented（2026-06-11）。owner 拍板扩大范围后落地：**不只加警告，还删光了
全部 legacy 兼容机器**——「已删除的旧 key = 拼错的 key = 没见过的 key」，统一一种
处理：警告并放行。

落地记录：
- **最终形态（2026-06-11 第二轮，回应 owner「警告应该覆盖整棵树，不该取决于哪个类
  继承了谁」）**：警告收敛为**一个全树遍历器** `vrl/config/unknown_keys.py` ——
  单入口（require_training_config / validate_reward_config 各调一次），整棵树
  每一层对照该层的已知 key 集；已知集**从真正消费它的类型派生**（pydantic
  `model_fields` / dataclass `fields()`），嵌套块（sde、lora、optim、
  distributed.resources 各角色、orchestration、torch_profiler…）全部覆盖。
  开放块（worker_config、未建模 reward 的 kwargs、cosmos）显式标 OPEN 不下钻。
- `ConfigBase` 退回纯类型边界（extra="ignore"，无行为）；字段声明兼任已知 key
  注册表（~80 个 key，含 7 个离线 DPO key）。
- **删除全部 legacy 拒绝器**：precision 的 `_reject_legacy_precision_keys` 与
  compute/rollout 专用报错；validation 的 adv_estimator/per_prompt_stat_tracking；
  schema 的 kling backend/endpoint 删除字段拒绝（保留 worker_config 结构检查与
  production 守门）；rewards/base.py 的 backend 哨兵参数。
  「已删除的旧 key = 拼错 = 没见过」统一为一行警告。
- 22 个 legacy 断言测试改写；新增 `tests/config/test_unknown_keys.py`
  （全深度/开放块/单行警告三用例）；安全性保持测试（未知 precision key 无法拆分
  rollout/replay 精度）。
- 验证：普查 18 实验 × 变体零误报（途中抓回 4 个漏登记真 key：cross_node、
  lora.dropout/init、memory.frozen_offload）；全量 725 passed；演示
  `rollout.sde.window_sze` / `resources.reward.share_with_rolout` 等任意深度
  拼错均被点名。

## 0. 结论（先读这个）

给 schema 加「未知 YAML key 警告」这件事，**试过一次、回滚了**（2026-06-11）：

- 机制本身只要 10 行（`ConfigBase`：`extra="allow"` + 收集 `model_extra` 打 warning）。
- 但实测全是误报：`num_steps`/`lora`/`path` 等最常用真实 key 全被报成未知。
- 根因：本仓库 schema 是**有意的部分建模**——每个模型只声明它要校验的字段（约三成），
  其余七成真实 key 一直靠 `extra="ignore"` 放行。「字段声明 = 已知清单」不成立。
- **真实前提 = 先把约 70 个真实 key 补成 schema 字段声明**（半天起的细活，顺带成为
  key 级文档）。误报警告比没有警告更糟（人会学会无视），所以不补全就不要装。

## 1. 已拍板的设计决策（触发后照此实施，不再讨论）

1. **单一行为，写死，无模式开关**——「ignore/warn/forbid 三选一」本身会成为新的困惑
   配置项。
2. ignore 消灭；**forbid 永不实现**（挡实验 key，owner 明确不要）；「报告并放行」是
   唯一行为，没有名字、没有开关。
3. 基类名 `ConfigBase`（平实名词）；ignore/warn/forbid 只是 pydantic 内部参数，
   不出现在用户可见处。

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

## 2. 触发条件（满足任一才启动）

- 又一次发生「拼错/死 key 静默生效」造成的实际损失（白跑的训练、错误结论），或
- 正在系统性补全 schema 字段（为了类型化/文档化本身），warn 顺手就能上。

在那之前的廉价替代：**偶尔重跑 yaml 审计 workflow**（2026-06 那次：8 域清点 +
8 域对抗复核，半小时，254 key 全覆盖）——周期性审计比维护完整 schema 便宜。

## 3. 实施步骤（触发后）

1. 用上次审计的 census + 一次全实验加载的误报清单，把缺的真实 key 逐个补成
   schema 字段声明（带类型）。
2. 加 `ConfigBase`，10 处 `ConfigDict(extra="ignore")` 模型改继承它。
3. `RootConfig` 补声明 `trainer/actor/distributed/precision/cosmos` 五个段
   （`Any` 透传，各自有独立验证层）。
4. 验收：全部实验加载零 warning；拼错 key（`num_stps`）被指名道姓；
   新实验 key 照常运行仅 warning。

## 4. 参考

- `vrl/config/schema.py`（extra="ignore" ×10；第 5 行 migration 注释）
- 审计来源：yaml-config-audit workflow `wf_84e8047e-fcb`（254 key / 30 findings）
- pydantic `model_extra`: https://docs.pydantic.dev/latest/concepts/models/#extra-data
