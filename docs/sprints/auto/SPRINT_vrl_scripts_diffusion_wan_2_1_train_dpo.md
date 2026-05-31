# SPRINT(auto): vrl/scripts/diffusion/wan_2_1/train_dpo.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/scripts/diffusion/wan_2_1/train_dpo.py` (325 LOC)
角色判定: script
结论: improve

## 0. 一句话
这是一个真实在用的离线 DPO 训练驱动（被 `configs/.../offline_dpo_pickapic.yaml` 的 `trainer.entrypoint` 引用），不是死代码；但它把 6 件不相关的事塞进一个函数、checkpoint 导出块原样重复两遍、并且手抄了 `DPOStepMetrics` 字段做 CSV header/row（会随 dataclass 加字段悄悄腐烂），这几处该收拾。

## 1. 现状（读代码得出）
`train_wan_2_1_dpo(cfg)` 是 Wan 离线 Diffusion-DPO 的入口。它被 YAML 引用：
```
# configs/experiment/diffusion/wan_2_1/offline_dpo_pickapic.yaml:20
entrypoint: vrl.scripts.diffusion.wan_2_1.train_dpo:train_wan_2_1_dpo
```
与 GRPO 走 `run_online_recipe`（train.py）不同，DPO 没有共享 recipe，整个驱动手写在此函数内：
- `_build_encoders(...)`（43-86）：构造 `encode_pixels/encode_text` 闭包，把单图复制成 T 帧再 VAE 编码。
- 配置桥接（111-205）：把 YAML 切片翻译成 `OfflineDPOTrainerConfig`。
- 数据（158-185）：`load_pickapic` + `DataLoader(collate_preference)`。
- trainer 装配（209-218）。
- 输出/CSV（233-244）。
- 训练循环 + checkpoint（246-324）。

`OfflineDPOTrainer` 自己只暴露 `.step(batch)` / `.global_step`（`vrl/trainers/offline/dpo.py:291,211`），不拥有循环——所以循环留在脚本里是当前设计的合理结果，不是 bug。

## 2. 质疑点 / 改进机会
1. **手抄 typed 结构（AGENTS.md ALL_CAPS/手抄规则）**：CSV header 与 row 把 `DPOStepMetrics` 的字段名/顺序手写了两遍：
   ```
   train_dpo.py:241  "step,loss,raw_model_loss,raw_ref_loss,model_diff,ref_diff,"
   train_dpo.py:242  "implicit_acc,sft_loss,grad_norm\n"
   train_dpo.py:281  f"{step},{m.loss:.6f},{m.raw_model_loss:.6f},{m.raw_ref_loss:.6f},"
   ```
   对照 `DPOStepMetrics`（`vrl/trainers/offline/dpo.py:68-78`）字段恰好是 `loss, raw_model_loss, raw_ref_loss, model_diff, ref_diff, implicit_acc, sft_loss, grad_norm`。给 dataclass 加一个指标，这两处不会报错、CSV 会静默错列。应从 `fields(DPOStepMetrics)` derive header 与 row。
2. **重复块**：checkpoint 的 `export_modules={LORA_WEIGHTS_NAME: transformer} if use_lora and hasattr(...) else None` 在 299-302 与 319-322 原样出现两遍。应抽成一个局部 helper（例如 `_export_modules() -> dict | None`）。
3. **职责过载**：单函数 ~236 行同时管编码器、配置桥接、数据、trainer 装配、CSV、循环+ckpt。可拆出 `_build_dpo_trainer_config(cfg, ...)`、`_build_dataloader(cfg, ...)`、`_run_dpo_loop(trainer, dataloader, cfg, out_dir, ...)` 三段，让顶层函数变成可读的编排。
4. **命名（轻微）**：`_build_encoders` 返回 `(encode_pixels, encode_text)` 元组闭包是合理的；不 flag。

非死代码、非薄 wrapper：grep 确认 YAML 引用其入口，`OfflineDPOTrainer` 不自带循环，故脚本内驱动属正常分层。

## 3. 建议动作
- 用 `from dataclasses import fields` 从 `DPOStepMetrics` derive CSV header 与每行 row（保持当前精度格式），消除手抄漂移。
- 把 299-302 / 319-322 的 export 块抽成局部 `_export_modules()` 闭包，调用两次。
- 把配置桥接 / dataloader / 训练循环拆成 3 个模块级 helper，顶层 `train_wan_2_1_dpo` 只做编排。
- 不删文件、不改入口签名、不改 YAML 契约。

## 4. 不动什么 / 为什么不是过度清理
- `_build_encoders` 的 num_frames 复制语义是 Wan 特有逻辑（注释已解释 image→video 适配），保留。
- 不要为了"统一"硬把 DPO 塞进 `run_online_recipe`：DPO 是离线纯函数 loss、无 rollout/algorithm-ABC（见文件 docstring 1-10 行），与 online recipe 形状本就不同，强行合并会引入坏抽象。这是 consistency-over-cleanup 的反向边界：此处差异是真实的。
- `require/optional_none` 显式校验（YAML-as-source-of-truth）保留。

## 5. 验证
- `ruff check vrl/scripts/diffusion/wan_2_1/train_dpo.py`
- `python -c "from dataclasses import fields; from vrl.trainers.offline.dpo import DPOStepMetrics; print([f.name for f in fields(DPOStepMetrics)])"` 确认 derive 出的 header 与现有字符串逐字一致（顺序也一致）。
- `grep -rn "train_wan_2_1_dpo" configs/` 确认入口仍被 YAML 引用、签名未变。
- 跑一次 `offline_dpo_pickapic.yaml` 的最小步数（`trainer.max_train_steps` 调小）冒烟，确认 metrics.csv 列与旧版一致、checkpoint 仍能保存。
