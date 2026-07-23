# SPRINT：EMA adapter artifact export

状态：**done（2026-07-23）**。

## 0. 结论先行

`checkpoint.pt`永远保存当前 raw training state；EMA只影响可选的 PEFT发布 artifact。两者不是同一
source of truth：

```text
raw model + trainer/optimizer/EMA state
    -> checkpoint.pt                  # exact resume

temporary EMA parameter swap
    -> gathered adapter state
    -> selected PEFT artifact         # inference/publishing
    -> restore raw parameters
```

所有 FSDP gather与 EMA swap必须由每个 rank参与；只有最终文件写入由
`strategy.context.is_primary`决定。checkpoint boundary自己传播每个阶段的 rank-local failure，
caller不再用额外 barrier猜测 publication是否成功。

## 1. 审计判定

| Suspect | 判定 | 证据与落地 |
|---|---|---|
| `is_primary` save参数 | **DERIVE** | 唯一 writer从 `strategy.context.is_primary`取得；caller无法传矛盾副本 |
| caller checkpoint barrier | **REMOVE** | checkpoint boundary已包含 publication agreement；重复 barrier会在 rank0 IO失败时挂住 peers |
| adapter export tuple/dict | **FIX** | `AdapterExport(module, adapter_name)`表达 typed artifact契约 |
| optional `peft_config` introspection | **KEEP** | `save_pretrained(..., selected_adapters=...)`是协议；暴露 mapping时提前验证 name，不强绑具体 PEFT class或破坏 test fake |
| adapter root/state prefix | **DERIVE** | 从 `RuntimeBundle.trainable_modules` object identity与 gathered state FQN派生 |
| arbitrary artifact/adapter path | **FIX** | 拒绝 absolute、drive、backslash、`.`/`..`、空 segment、ancestor/descendant与effective output collision |
| PEFT adapter选择 | **FIX** | `save_pretrained(..., selected_adapters=[adapter_name])`只发布指定 adapter |
| raw checkpoint与EMA artifact | **KEEP** | resume与inference/publishing是两个真实协议边界 |
| EMA swap temp snapshot | **KEEP** | raw parameter无法从EMA值派生；失败恢复需要明确 ownership |
| strategy `barrier()` | **KEEP** | 公共 distributed capability仍被其他调用者需要；只从 checkpoint flow删除重复调用 |
| checkpoint/file constants | **KEEP** | schema与持久化协议名，不是业务 vocabulary |

## 2. Typed adapter export

`AdapterExport`接收：

- 暴露 `save_pretrained()`的 PEFT-compatible module；
- 一个安全 adapter name；若 module暴露 mapping形式的 `peft_config`，该 name必须存在。

adapter name必须是安全的单路径段。artifact mapping的 key可以是安全相对嵌套路径，但所有 effective
outputs必须互不相同且不能互为 ancestor/descendant。验证在任何 checkpoint IO前完成。

不强制 concrete `PeftModel`或必需 `peft_config` attribute是有意的 protocol boundary：production
wrapper和 test fake都可实现同一 `save_pretrained(state_dict=..., selected_adapters=...)`能力。
当 dependency提供可读 adapter registry时提前 fail closed；否则由这次明确方法调用验证协议，不为
方便 introspection复制或收紧第三方类型层次。

每个 export通过 module object identity定位唯一 `trainable_modules` root，并从 root/module state FQN
派生 prefix。caller不再传 root name或 prefix副本，因此无法把一个 adapter的权重错误写到另一个
artifact。gather后只截取该 prefix下的 checkpoint-owned state；空结果直接失败。

PEFT写入显式传：

```python
selected_adapters=[adapter_name]
```

这使 frozen `previous` adapter可以继续保留在 raw checkpoint中用于精确训练恢复，而默认 inference
artifact只包含被选择的 `default` adapter。

## 3. Raw checkpoint与EMA artifact

save顺序固定：

1. 所有 rank gather raw checkpoint-owned state；
2. 所有 rank gather trainer state；
3. 若需要且 EMA已经更新，所有 rank暂存 raw parameters并 swap到 EMA；
4. 所有 rank gather EMA artifact state；
5. 所有 rank恢复 raw parameters；
6. primary rank在同 filesystem staging目录写 checkpoint、artifacts与 meta；
7. primary原子发布完整目录，所有 rank对 publication结果达成一致。

因此 `checkpoint.pt`不会因开启 EMA artifact export而变成 EMA resume source。若 EMA尚未更新，
artifact明确使用 raw checkpoint-owned weights；不会写一份未初始化 shadow state。

EMA swap部分失败时，已成功 swap的 ranks先尝试恢复 raw weights：rollback成功则恢复 raw后共同
失败；rollback本身失败则所有 rank在下一 collective/publication前显式失败，不会假装 live参数已经
恢复。若原始参数 snapshot不存在，`copy_temp_to()`显式抛错；不使用可被 `python -O`删除的
`assert`。EMA gather或 raw restore失败也在进入下一 collective/publication前达成一致。

## 4. Distributed ownership与failure agreement

FSDP state export是 collective，不能把整个 save包在 `if is_primary`内。每个阶段都遵循：

```text
rank-local work
    -> strategy.all_ranks_succeeded(local_success)
    -> all ranks continue, or all ranks fail/rollback
```

agreement覆盖：

- export schema/path setup；
- EMA presence与 update state；
- raw checkpoint gather；
- trainer/optimizer/EMA state gather；
- parameter preflight；
- EMA swap；
- EMA artifact gather；
- raw parameter restore；
- primary publication。

primary只拥有 filesystem IO，不拥有 collective decision。rank0写盘失败时，primary保留原始异常，
peer收到明确 publication failure；没有 rank停在 caller-owned barrier。

## 5. ALL_CAPS 与薄函数

### KEEP

- `CHECKPOINT_SCHEMA_VERSION`、`TRAINING_CHECKPOINT_NAME`、`LORA_WEIGHTS_NAME`、
  `CHECKPOINT_META_NAME`：持久化协议常量；
- `AdapterExport`：dependency-facing typed artifact边界；
- `_safe_relative_output_path()`：所有 dependency output path共用的安全验证；
- strategy `all_ranks_succeeded()`与 `barrier()`：distributed public capabilities；
- raw-vs-artifact两阶段：resume与publishing职责不同，不能为减少 LOC合并。

### REMOVE / DERIVE

- 删除 caller传入的 `is_primary`；
- 删除 online checkpoint后的重复 barrier；
- root name、state prefix与 PEFT selected adapter不手写第二份；
- rank0不是 collective owner，只从 strategy context派生 filesystem writer。

薄函数按 path validation、state selection、atomic publication与 rank agreement分开，因为它们有不同
失败边界和测试注入点。这里 cross-strategy一致性与异常可诊断性比少几行更重要。

## 6. Non-goals

- 不把 EMA weights写进 raw checkpoint model root来替代训练参数。
- 不删除 `Strategy.barrier()`公共能力。
- 不让 rank0单独进入 FSDP gather。
- 不导出全部 PEFT adapters作为默认 artifact。
- 不改变 EMA更新公式或训练 step cadence。
- 不运行 GPU/Ray；distributed验证使用真实 CPU Gloo进程。

## 7. Verification

```text
tests/trainers:                     460 passed, 3 skipped
checkpoint focused:                130 passed
real 2-rank CPU Gloo checkpoint:     4 scenarios passed
```

真实 2-rank测试覆盖：

1. raw checkpoint + EMA artifact成功，raw参数恢复；
2. 单 rank EMA swap失败，双方退出且成功 rank回滚；
3. 单 rank swap失败且另一 rank rollback失败，双方显式退出且不发布 checkpoint；
4. rank0 artifact write失败，双方退出且不发布 checkpoint。

额外覆盖 true/false adapter selection、非法/重叠路径、错误 root/prefix、无 update EMA、restore失败、
trainer/optimizer state、FSDP exact owned state与 atomic publication。Ruff、format和
`git diff --check`通过；未启动 Ray或 GPU。

实现 commits：`ce11b16c`、`d2cf23ca`。

## 8. Definition of Done

- [x] raw checkpoint与EMA inference artifact有明确、不可混淆的 source of truth。
- [x] adapter/root/prefix/path在 IO前验证或派生。
- [x] PEFT只导出选中的 adapter。
- [x] 每个 collective阶段由所有 rank参与并传播失败。
- [x] EMA swap成功/失败路径恢复 raw；rollback失败时所有 rank在 publication前显式失败。
- [x] rank0 publication失败不会让 peers挂在 barrier。
- [x] primary ownership只有 strategy context一个来源。

## 9. References

- `vrl/trainers/checkpointing.py`
- `vrl/trainers/online/ema.py`
- `vrl/trainers/strategy.py`
- `vrl/trainers/fsdp.py`
- `vrl/scripts/common/online.py`
- `vrl/models/interfaces/runtime.py`
- `tests/trainers/test_checkpointing.py`
- `tests/trainers/test_ema.py`
- `tests/trainers/test_fsdp.py`
- `tests/trainers/test_fsdp_gather_distributed.py`
- `tests/scripts/test_online_metrics.py`
- `docs/sprints/done/SPRINT_multi_gpu_training.md`
