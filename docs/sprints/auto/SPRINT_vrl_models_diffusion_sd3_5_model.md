# SPRINT(auto): vrl/models/diffusion/sd3_5/model.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/diffusion/sd3_5/model.py` (488 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件本体是合格的 SD3.5 family model 实现（与 wan_2_1 family 形状一致），唯一可质疑点是文件尾部 `_resolve_torch_dtype` 这个纯字符串→`torch.dtype` 的工具函数被逐字复制到了另一个 family，应提到 `common/` 共享，而不是每个 family 各抄一份。

## 1. 现状（读代码得出）
`SD3_5Model` / `SD3_5ReplayModel` 实现 `DiffusionModelBase` 契约（`from_spec` / `apply_lora` / `encode_prompt` / `prepare_sampling` / `forward_step` / `decode_latents` 等），forward 走共享 `DiffusionBackboneCaller` + family 专属 `SD3DiffusionBackboneRunner`，timestep 走共享 `expand_batch_timestep` / `pack_eval_timestep`，decode 走共享 `ChunkedLatentDecoder`。这些都正确复用了 `vrl/models/diffusion/common/`。

问题只在文件末尾这个 module-private 工具：

```python
# model.py:467
def _resolve_torch_dtype(value: Any) -> torch.dtype:
    ...
    aliases = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    }
```

## 2. 质疑点 / 改进机会
- `def _resolve_torch_dtype` 在两个 family 里逐字重复：
  - `vrl/models/diffusion/sd3_5/model.py:467`
  - `vrl/models/diffusion/cosmos/anima/model.py:839`
  grep 确认全仓只有这两处定义（`grep -rn "def _resolve_torch_dtype"` 命中 2 次）。两份实现一致（同样的 alias 表、同样的 `removeprefix("torch.")`、同样的 `ValueError`）。
- 该 family 目录已经有 `vrl/models/diffusion/common/`（`timestep.py` / `latent_decode.py` / `backbone.py` / `cfg.py`），dtype 解析属于同一类"diffusion family 共享纯函数"，没有放进去是遗漏，而非有意隔离。这是 AGENTS.md 说的"移除真实复杂度的共享抽象"应当存在却缺失的情形——再加 family 时第三份拷贝会继续腐烂。

注意：这不是薄 wrapper 问题，也不是 ALL_CAPS 手抄 typed 结构问题（alias 表是字符串别名 taxonomy，是真边界，保留合理），只是放错了位置 + 重复。

## 3. 建议动作
- 在 `vrl/models/diffusion/common/` 新增（或并入现有 dtype 相关模块）一个 `resolve_torch_dtype(value) -> torch.dtype`，把 alias 表搬过去。
- `sd3_5/model.py` 与 `cosmos/anima/model.py` 改为 `from vrl.models.diffusion.common import resolve_torch_dtype`，删掉各自的 `_resolve_torch_dtype` 定义。
- 不要顺手改动 family 的 model 契约方法或 runner——那些是有意的跨家族一致形状。

## 4. 不动什么 / 为什么不是过度清理
- `SD3SamplingState` dataclass、`SD3_5Model` / `SD3_5ReplayModel`、各契约方法：保留。`SD3_5ReplayModel` 故意 `raise` 掉 `encode_prompt` / `decode_latents` / `prepare_sampling`，是 trainer replay bundle 不加载 text encoder / VAE 的真实边界，不是死代码。
- alias 表本身保留（字符串→dtype 的协议映射，是真边界）。
- 与 wan_2_1 family 对齐的方法签名/结构保留——跨家族一致性优先于 LOC。
- 本 sprint 只动 dtype helper 的位置，不重写 model。

## 5. 验证
- grep 确认搬迁后无残留：`grep -rn "def _resolve_torch_dtype" vrl/` 应为 0 命中，`grep -rn "resolve_torch_dtype" vrl/models/diffusion/` 全部指向 common。
- 跑相关测试：`pytest tests/models/test_sd3_model_loading.py tests/models/test_diffusion_model_base.py -q`（覆盖 `from_spec` dtype 解析路径）以及 cosmos anima 对应加载测试。
- `ruff check vrl/models/diffusion/`。
