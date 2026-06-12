# SPRINT: Rollout wire diet（uint8 过线 + 过线前存储策略 + 优化器批量化）

状态：T1 / T2a / T3 **implemented**（2026-06-11）。T2b **deferred**（owner 决定，
2026-06-11——牵动 trainer 契约，等真需要那 216MB 时再独立实施，触点清单见 §2）。

## 0. 背景与收益

rollout worker → driver 每个 prompt group 过线约 717MB，其中 ~474MB 是 fp32
解码视频（唯一下游是 uint8 mp4 编码，精度纯浪费）。本 sprint 砍掉视频那份。

## 1. 已落地

- **T1 uint8 过线**：worker 在 decode 输出处用全仓唯一的 `to_uint8` 打包
  （`vrl/generation/diffusion/executor.py`）；driver 在 `reward_outputs()`
  单点 `k/255` 还原（`rollouts/collector/batch_builder.py`）。还原值经下游
  同一 `to_uint8` 量化**逐位往返**（穷举 256 个字节值的 pin 测试）。训练张量
  （latents/log_probs/observations/actions）一个字节未动。
- **T2a 存储策略前移**：`rollout.trajectory_storage` 的 dtype/device 下转
  从 driver 侧（batch_builder）前移到 worker 过线前
  （`apply_wire_storage_policy`，policy 走 `request.sampling` 现成通道）；
  driver 侧原应用保留为幂等兜底。默认 `preserve` 严格恒等（有 tensor
  identity 测试钉死，GRPO 基线逐位不变）。
- **T3 优化器批量化**：EMA 同设备参数 `_foreach_lerp_` 批量更新；AdamW
  `fused=True`（仅全 CUDA float 参数时启用，否则回退默认）。
- 测试：4 个 pin 测试（uint8 打包 / wire 下转 / reward 还原 / 256 值往返穷举），
  相关套件 441 passed。

## 2. T2b — 字段级裁剪（deferred）

NFT 不消费 observations/actions（~216MB/组）但 worker 照寄。砍掉它需要教会
整条接收链容忍字段缺失，触点：

```text
1. trainer.py:534 改从 timesteps.shape[1] 数步数
2. RolloutBatch.observations/actions 可空化(concat/select/切片容忍 None)
3. trajectory builder/validator 的 observation/action role 改按声明
4. TrajectoryStoragePolicy 加 fields 维度(full | replay_minimal),NFT 配方
   声明 replay_minimal,GRPO 缺省 full 零变化
5. 回归:GRPO lr=0 parity 不变;NFT loss 曲线与全量导出逐位一致
```

## 3. 待跑验收（下次 GPU smoke 顺带）

```text
G1 字节实测:改前后对比 wire 字节(锚点 executor.py 的 diffusion_*_bytes)
G2 GRPO lr=0 parity 与基线同量级(predict2 基线 5.3e-6)
```

## 4. References

- `vrl/generation/diffusion/executor.py`（打包 + wire policy + 字节计量）
- `vrl/rollouts/collector/batch_builder.py`（还原 + 兜底）
- `vrl/utils/media.py`（`to_uint8`，全仓唯一量化公式）
- `docs/sprints/SPRINT_cross_model_performance.md`（717MB 数字出处）
