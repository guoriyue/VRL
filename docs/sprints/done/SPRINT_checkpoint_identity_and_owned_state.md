# SPRINT：Checkpoint identity and owned state

状态：**done（2026-07-23）**。

## 0. 结论先行

训练恢复、rollout worker与 eval现在共享同一份 immutable model identity。checkpoint schema v2只保存
当前训练真正拥有的 state，而不是把 immutable base model复制进每个 checkpoint：

```text
typed family model schema
    -> checkpoint identity metadata
    -> path-independent model identity
    -> sidecar preflight / authoritative checkpoint payload
    -> online, DPO, rollout worker and eval compatibility checks

requires_grad parameters + explicitly registered frozen mutable state
    -> exact checkpoint-owned state
```

这里保留两个真实边界：

- model identity回答“这份训练 state属于哪套 immutable权重/拓扑”；
- checkpoint-owned state回答“为了精确继续训练，哪些 mutable tensors必须保存”。

两者不能合并。identity不是模型 state，owned state也不能证明 immutable base来源一致。

## 1. 审计判定

| Suspect | 判定 | 证据与落地 |
|---|---|---|
| family手写 identity表 | **REMOVE/DERIVE** | 从 registry选择的 typed model schema及 field metadata派生，不维护第二张 family表 |
| local checkpoint path | **DERIVE** | identity保存 path-free content kind/SHA-256/bytes/files，不把机器路径当身份 |
| remote model source | **FIX** | 要求可审计 revision/pin，source/member/value语义由 schema metadata声明 |
| schema-v2 model payload | **KEEP** | 精确协议为 `identity + owned_state`，拒绝额外或缺失 key |
| checkpoint-owned keys | **DERIVE** | 从 `requires_grad`加显式 frozen-mutable registry得到 |
| frozen mutable exception | **KEEP** | 例如 DiffusionNFT `previous` adapter无法从 `requires_grad`派生，必须显式注册 |
| schema version coercion | **FIX** | 只接受 exact `int`；拒绝 bool、float与数字字符串 |
| schema-v1 checkpoint | **KEEP/MIGRATE** | full-state继续恢复 frozen base；selective strict restore必须有可验证 identity |
| sidecar与payload | **KEEP** | sidecar用于构建前低成本 preflight；存在时必须与权威 `checkpoint.pt`一致 |
| worker identity payload | **KEEP** | `GenerationRuntimeLaunchContract.expected_model_identity`是真实 Ray wire契约 |
| worker构建前/后检查 | **KEEP/FIX** | 前者发现 driver/worker source差异，后者发现模型构建期间 source TOCTOU |
| compatibility facade | **KEEP** | training、eval与旧 caller共用同一 restore/preflight API，避免各自实现规则 |

## 2. Immutable model identity

`vrl/models/checkpoint_identity.py`不枚举 family。每个 public model field必须通过
`checkpoint_identity_metadata()`声明为：

- `source` / `source_revision`：权重来源与 pin；
- `member`：选中 source内部影响实际构建的成员；
- `value`：影响 topology或构造语义的值；
- `lora`：只有启用 LoRA时才进入 identity的 typed配置；
- `exclude`：确认不影响 checkpoint兼容性的字段。

`validate_checkpoint_identity_schema()` fail closed：新增 field若没有分类、引用不存在 source/revision、
声明未知 metadata key或形成矛盾 override，会在模型构建前失败。允许值集合从
`IdentityKind`派生，不手写第二份 `_ALLOWED_KEYS`。

local file/tree identity按内容计算，不保存绝对路径；remote source必须有可审计 pin。token family的
effective defaults通过 registry-backed runtime config解析，因此“省略默认值”与“显式写同一个默认值”
得到相同 identity。

## 3. Checkpoint schema v2

schema v2的 model root精确为：

```python
{
    "identity": {...},
    "owned_state": {
        "<runtime root>": {"<tensor FQN>": tensor},
    },
}
```

`checkpoint_owned_state_names(module)`从以下两部分派生：

1. `requires_grad=True` parameters；
2. `register_checkpoint_owned_state()`登记的 frozen mutable parameters/buffers。

显式 registry只承载不可派生例外。它拒绝不存在的 FQN，也拒绝重复注册 trainable parameter，避免
“自动 owner + 手写 owner”两份真值。

restore在修改任何 module前验证：

- schema、family与 immutable identity；
- runtime roots精确集合；
- 每个 root的 owned keys；
- tensor类型与 global logical shape；
- schema-v1 compile prefix迁移是否无冲突。

FSDP通过 strategy seam把完整 owned tensors scatter回 local DTensor shards。strict restore完成后再
export一次 checkpoint-owned projection并做 tensor-exact比较，证明 schema-v2/selective-owned恢复
结果而不只相信 `load_state_dict()`返回值。schema-v1 full-state的 frozen base由 strict full-state
loader覆盖，但不属于这次 post-load owned projection比较范围。

### Schema v1 compatibility

旧 full-state checkpoint的完整 root仍是旧协议的 source of truth，包含 frozen base tensors，不能在
迁移时只选当前 trainable keys。旧 selective checkpoint在 strict模式下必须同时有 saved/runtime
identity；否则它无法证明缺失 frozen base来自同一模型，必须拒绝。

## 4. Integration boundaries

### Online与DPO

online replay bundle和 Wan DPO都在模型构建前解析 identity并执行 checkpoint compatibility preflight；
构建后再次解析，若 local source在 load期间变化则 fail closed。full resume同时拒绝不相关
`model.lora.path` warm start，避免先装一个 adapter topology再覆盖 checkpoint。

### Generation worker

driver把 primitive `ModelBuild` payload与非空 expected identity放入
`GenerationRuntimeLaunchContract`。launcher在创建 Ray worker前比较 rollout/replay build identity；
worker重建 `ModelBuild`后在模型构建前检查一次，构建后再检查一次。这个 wire字段不是 duplicated
runtime state，而是跨进程必须传递的协议证据。

### Eval

SANA aesthetic/compare与 Cosmos eval在 sidecar存在时先做低成本兼容检查，再由 shared checkpoint
loader验证 authoritative payload并恢复 owned state。对 local model source同样执行构建前/后
content identity检查，避免“checkpoint正确但 inference加载期间源目录被替换”。

## 5. ALL_CAPS 与薄函数

### KEEP

- `MODEL_IDENTITY_SCHEMA`、`CHECKPOINT_SCHEMA_VERSION`、checkpoint/meta/file names：持久化协议常量；
- metadata key与 module attribute name：schema/internal protocol key；
- `validate_checkpoint_compatibility()`与 `validate_checkpoint_meta_compatibility()`：payload与
  optional sidecar两个不同成本/权威级别的边界；
- `restore_model_checkpoint()`：training/eval共享的 public restore facade；
- `TrainingCheckpoint.trainable_state`：schema-v1 API compatibility facade；
- registry dotted schema path与 generation launch contract：lazy import和 Ray wire边界。

### REMOVE / DERIVE

- 不维护 family identity allow-list；
- 不把 owned keys手写进 checkpoint code；
- 不从路径字符串或 HF cache目录布局猜权重身份；
- 不让 caller另外传一个可与 strategy矛盾的 rank/primary事实。

这些薄函数不是为了少几行而拆；它们分别承担 cheap preflight、authoritative load、distributed
scatter与 public compatibility。跨 training/eval/family的一致失败语义比 LOC减少更重要。

## 6. Non-goals与 deferred risk

- 不把 immutable base model完整复制进每个 schema-v2 checkpoint。
- 不删除 schema-v1读取能力。
- 不把 checkpoint runtime state与 Ray worker lifecycle/version state合并。
- 不把 eval的 sidecar preflight当成 authoritative restore。
- 不运行 Ray/GPU来证明纯 schema/restore契约。

MAGI-1的 `checkpoint_path`、`t5_pretrained_path`、`vae_pretrained_path`会选择实际权重，但当前标为
identity `exclude`。MAGI目前是 generation-only，没有 training replay/checkpoint resume consumer，
所以本 Sprint没有虚构一个未被使用的多源训练协议。启用 MAGI训练前，必须把这三个路径表示为彼此
独立的 checkpoint sources并加入 identity；这是明确 deferred gate，不是当前训练恢复的 silent
fallback。

## 7. Verification

```text
identity + online/DPO/worker/eval focused: 188 passed
trainer/checkpoint shared gate:            460 passed, 3 skipped
eval focused / extended:                    92 passed / 105 passed
```

覆盖 identity相等/不等、缺 pin、local file/tree mutation、schema metadata true/false、v1 full/selective、
v2 exact roots/keys/type/shape、online/DPO/worker构建前后漂移，以及 eval在 checkpoint load或
generation前失败。全部为 CPU-only；未启动 Ray cluster或 GPU。

相关 commits：

- `5f7b6ab4` — typed checkpoint revision；
- `595f2df8` — compatibility preflight；
- `92116a91` — frozen mutable state registry；
- `b7851d64` — LoRA warm-start topology validation；
- `95952909` — immutable checkpoint identity；
- `ce11b16c` — schema-v2 distributed save/restore integration；
- `171e95da` — eval identity与 inference source drift gate。

## 8. Definition of Done

- [x] 每个 public model field都有 fail-closed identity分类。
- [x] schema-v2只保存 exact checkpoint-owned state。
- [x] frozen mutable例外显式注册，trainable state自动派生。
- [x] strict restore在任何 mutation前验证 identity、roots、keys、type与shape。
- [x] schema-v1 full/selective语义被明确迁移而非静默降级。
- [x] online、DPO、generation worker与 eval复用同一 identity contract。
- [x] sidecar与payload权威级别清楚；sidecar存在时二者不漂移。

## 9. References

- `vrl/models/checkpoint_identity.py`
- `vrl/models/interfaces/runtime.py`
- `vrl/config/model_schema.py`
- `vrl/families/registry.py`
- `vrl/trainers/checkpointing.py`
- `vrl/trainers/fsdp.py`
- `vrl/trainers/strategy.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/execution/worker.py`
- `vrl/scripts/common/online.py`
- `vrl/scripts/families/wan_2_1/train_dpo.py`
- `vrl/scripts/eval/sana_aesthetic_checkpoint_eval.py`
- `vrl/scripts/eval/sana_checkpoint_compare.py`
- `vrl/scripts/eval/cosmos_predict25_kling_eval.py`
- `tests/models/test_checkpoint_identity.py`
- `tests/models/interfaces/test_checkpoint_owned_state.py`
- `tests/trainers/test_checkpointing.py`
- `tests/generation/execution/test_worker_checkpoint_identity.py`
- `tests/scripts/test_wan_dpo_checkpoint_identity.py`
