# SPRINT: Single-caller inlines & form-4 hoist — 单调用内联与同体上提

**日期**: 2026-07-12  **状态**: PLANNED
**来源**: 与 SPRINT_dead_code_wrapper_sweep 同一轮 392 文件通读审计。
本 sprint 收"有真调用方但不该独立存在"的项：单调用内联（no single-caller helpers）
与跨 family 同体重复上提（死代码形式四）。全部零行为变化。

---

## 1. 六家 byte-identical `_encoder_device` 上提基类（本 sprint 最大收益）

`pixart_sigma/model.py:191`、`mochi/model.py:155`、`hunyuan_image/model.py:165`、
`hunyuan_video/model.py:125`、`cogvideox/model.py:178`、`qwen_image/model.py:247`
六份**逐字节相同**的私有方法（单 text_encoder，`next(enc.parameters()).device`
try/except 回退 `self.device`），各自唯一调用方是本家 `encode_prompt`。

- 落点：`DiffusersPipelineModelBase`（`vrl/models/diffusion/base.py`）——
  已有文件，不新建模块（no new lean files）。
- FLUX（`flux/model.py:310`）保留其双 encoder（`text_encoder_2` 优先）覆写。
- 这是"六份同体 hoist"，不是"拍平某一家的统一钩子"——与 grab-bag audit 里
  10 份 `prompt_encoder_dtype` 块同一性质的收尾。
- 顺带：`_guidance_embeds` `@property` 在 flux/hunyuan_image/qwen_image 三家逐字
  相同，可同批上提；属可选项（@property 本在豁免清单内），做不做由执行时定。

## 2. 单调用内联（各 ≤5 行收益，合并成一个 commit）

1. `vrl/rollouts/collector/core.py:400` `_reward_view_name` → 内联进
   `score_rollouts` 的 `RolloutBatchBuildContext` 构造处（core.py:253）。
   同级字段 `kl_reward_coef` 本来就是内联的 `float(cfg_get(...))`——
   同一构造点两种风格，是提取不一致而非共享 helper。
2. `vrl/trainers/data/prompts.py:116` `_load_prompt_examples_from_config` →
   并入唯一 wrapper `load_prompt_examples_from_config`（:167）。docstring 声称的
   train/eval 双入口共享从未成立——eval 入口没接线，`manifest_key` 参数只有
   一种取值。注意 `vrl/config/schema.py` DataConfig 校验注释引用了此函数名，
   同步改指向合并后的公有函数。
3. `vrl/trainers/offline/dpo.py:127` `_trainable_forward_model`（2 行 getattr
   解包）→ 内联进唯一调用方 `wan_forward`（:143）。
4. `vrl/models/model_build.py:84` `_optional_block`（4 行）→ 内联进唯一调用方
   `resolve_model_build`（:53）。
5. `vrl/nn/quantization/targeting.py:37` `is_mlp_linear_path` → 并入唯一非测试
   调用方 `matches_linear_target`（:53）；其独立测试改为对
   `matches_linear_target` 的 MLP 分支断言（不降覆盖）。

## 非目标（审计核过、明确不动）

- `vrl/trainers/online/trainer.py`（2123 行）内的全部单调用私有 helper：
  长流程概念命名提取，保留清单内。
- `vrl/trainers/fsdp.py` 全部单调用函数：FSDP/DCP framework adapter
  （grab-bag audit 已裁定"一袋函数"是合法模式，不要类化）。
- `vrl/rewards/service/wire.py` 编解码对：对称版本化 wire 协议边界。
- `vrl/trajectory/` 全部单调用私有 helper（`_split_ref`/`_slice_axis`/
  `_loss_axis` 等）：resolver/validator 长流程的概念提取，全数 KEEP。
- `sd3_5/model.py:72` `_candidate_transformers`、wan 五个单调用私有 helper、
  `echo/_sigma_from_timestep`、`anima/_non_empty_prompts`：有真逻辑/独立测试，KEEP。
- `resolve_online_family`（factory.py:66）：字符串分发入口 `ar/train.py` 在用的
  公共 recipe facade，KEEP。

## 验证

- 全量 pytest 与基线逐数对齐（内联不得改任何断言路径；
  `is_mlp_linear_path` 的测试迁移除外，迁移前后用例数一致）。
- ruff check + format --check 仅限触碰文件。
- §1 上提后六家 `encode_prompt` 走一次 CPU 构造 smoke（现有单测已覆盖即可，
  确认无新增 attribute 解析路径）。
