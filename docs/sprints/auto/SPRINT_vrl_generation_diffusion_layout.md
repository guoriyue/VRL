# SPRINT(auto): vrl/generation/diffusion/layout.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/diffusion/layout.py` (315 LOC)
角色判定: core
结论: improve

## 0. 一句话
`DiffusionRequestLayout` 是真核心共享层（解析/校验/排序），但同文件里的 `VideoGenerationRequest` dataclass 携带约 13 个全仓库无人读取的死字段，应裁剪到实际被消费的字段。

## 1. 现状（读代码得出）
本文件两块内容：
1. `DiffusionRequestLayout`（layout.py:84）—— 跨 executor / gatherer 共享的 prompt-major 布局工具（parse_sampling_params / repeat_encoded_batch / ordered_chunks / SDE window 校验）。真边界，被广泛使用。
2. `VideoGenerationRequest`（layout.py:17-44）—— "backend-agnostic 模型请求"，被所有 5 个 model family 的 `model.py` / `runtime.py` 作为入参类型引用。

`build_video_request`（executor.py:233）构造它时只填这些字段：
```
prompt, num_steps, guidance_scale, height, width, frame_count, fps, negative_prompt, seed, extra
```

## 2. 质疑点 / 改进机会
`VideoGenerationRequest` 是一个 god-dataclass，混入了大量 CLI/官方推理脚本时代的遗留字段。grep 全仓库（`vrl/` + `tests/`）确认这些字段**无任何读取点**：

```
model_size, ckpt_dir, t5_cpu, sample_solver, shift,
offload_model, convert_model_dtype, high_noise_guidance_scale,
references, action_sequence, action_dim, action_conditioning_mode, task_type
```

验证命令与结果：`grep -rn "\.model_size\|\.ckpt_dir\|\.t5_cpu\|\.sample_solver\|\.offload_model\|\.convert_model_dtype\|\.high_noise_guidance_scale\|\.action_sequence\|\.action_conditioning_mode" vrl/` → 0 命中（`task_type=` 的命中来自 `ar/janus_pro` 与 `scripts/data/video_world.py` 的无关 dict，非本类字段）。

这违反 AGENTS.md：这是 build-time 的请求 spec，却把一份手抄的、和实际消费脱节的字段集合留在类里 —— 加 family 时谁都不知道哪些字段真生效，会悄悄腐烂。`extra: dict` 已经是逃生通道（executor.py:256 用它放 `max_sequence_length` / `init_latents`），更说明这些固定字段是冗余的。

命名上 `VideoGenerationRequest` 本身没问题（它确实是请求 spec，不是 runtime/manager 类）。

## 3. 建议动作
裁剪 `VideoGenerationRequest` 到实际被读取的字段集：
- 保留：`prompt, negative_prompt, num_steps, guidance_scale, height, width, frame_count, fps, seed, extra`（executor 构造 + model 消费的字段）。
- 删除上面列出的 13 个无引用字段。删除前对每个字段再跑一次 `grep -rn "<field>" vrl/ tests/` 二次确认（本审计已确认 0 读取，但删除是破坏性动作，逐字段复核）。
- family 特有的需求（如 anima 的 action 条件、wan 的 high-noise guidance）若将来需要，走 `extra` dict，与现有 `init_latents` / `max_sequence_length` 一致。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `DiffusionRequestLayout` 及其所有方法 / 校验私有方法（`_parse_sde_window_range` 等）：这是真共享抽象，executor 和 gather 都依赖，校验逻辑移除会丢失 SDE window 的正确性保证。
- 不动 `DiffusionBaseParams` / `DiffusionSDEParams` / `DiffusionSamplingParams`：frozen typed 解析结果，是 parse 输出的稳定边界。
- 不是为省 LOC：是为防止 typed 请求 spec 和真实消费集合脱节腐烂（AGENTS.md ALL_CAPS/手抄结构条款的同类问题）。

## 5. 验证
- 逐字段 `grep -rn "<field>" vrl/ tests/` = 0 后再删。
- 跑 `pytest tests/generation/diffusion/ tests/models/diffusion -q`（至少 import + 构造路径不炸）。
- `python -c "import vrl.models.diffusion.wan_2_1.runtime, vrl.models.diffusion.sd3_5.runtime, vrl.models.diffusion.cosmos.anima.runtime"` 确认所有 family 仍能 import。
- `ruff check vrl/generation/diffusion/layout.py`。
