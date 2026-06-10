# SPRINT: 跨家族真重复下沉到共享层

状态：implemented（2026-06-09，T6 除外）。父：`SPRINT_solid_architecture_audit.md`（子 sprint C）。

落地记录：
- T1 ✅ `LoraModelMixin`（`diffusion/common/lora.py`）：wan/sd3_5/predict2/anima 四家收口，hook 参数化（`_lora_default_init_weights`、`_lora_transformer`、`_lora_dtype`）。**实测发现 doc「仅 init_lora_weights 不同」不准确**：sd3_5 的 `.to()` 带 dtype、anima 的 transformer 不在 pipeline 上。predict2_5 两处**有意偏离**（NFT previous adapter），按守则跳过并注释记录。
- T2 ✅ `require_tensor`（`diffusion/common/tensors.py`）：4 个 runner 收口，统一带 `name` 的错误信息（无测试断言旧文字）。
- T3 ✅ `ARPipelineExecutorBase` 加 `_embed` / `_ar_runner` 模板（`_runner_cls` + `_runner_attention_family`）/ `_align_tokenizer_output`。**注意**：attention family 用独立属性而非 `self.family`——`janus_pro_r1` 的 family 与 backend 注册键（janus_pro）不同，直接用 family 会变行为。
- T4 ✅ `to_builtin_deep` 进 `vrl/utils/config.py`（以 launcher 深递归版为准），launcher + collector/config 两处本地版删除。tuple 行为差异不在 YAML 输入域内。
- T5 ✅（形式与 doc 不同）：videocon 的 `_torch_dtype` 经先前 dtype 统一后已是 `resolve_torch_dtype` 纯转发皮——直接删皮改直调；kling 版有真实语义（required/auto/fallback），保留。无需新建 `_video_utils.py`。
- T6 ⏭️ 跳过：doc 要求与子 sprint B 的 danbooru 拆分合做，B 被明确跳过。
- T7 ✅ `JANUS_IMAGE_PIXEL_SIZE` 改派生（`int(576**0.5) * 16`），断言 ==384 验证。
- 验证：models/generation/rollouts/rewards/scripts 全量 335 passed，ruff 干净。

## 0. Core Decision

把在多个模型/reward 家族里**逐字节复制**的 family-agnostic 逻辑下沉到薄共享层。

- **关键判别**：只下沉「逻辑与家族无关、复制纯属缺共享层」的；**绝不**碰「有意的跨家族一致 shape」（三套 sampling-state dataclass、并行 GRPO 变体——见父 sprint §6）。区别：前者是同一段实现被复制，后者是同一种**形状**被各家族**各自实现**以保 grepability。
- **守则**：下沉后行为逐位不变；保留必要的 per-family 参数化点（如 `init_lora_weights`）。

## 1. 目标清单（已实测确认位置）

| 重复符号 | 实测位置 | 差异 | 下沉目标 |
|---|---|---|---|
| `apply_lora` | `wan_2_1/model.py:140`、`sd3_5/model.py:144`、`cosmos/predict2/model.py:141`、`cosmos/predict2_5/model.py:185+584`、`cosmos/anima/model.py:158`（`base.py:193` 是 no-op 默认） | 仅 `init_lora_weights` 不同 | `LoraModelMixin` 或 `diffusion/common/lora.py`，`init_lora_weights` 参数化 |
| `_require_tensor` | `cosmos/predict2_5/runner.py:91`、`cosmos/predict2/runner.py:115`、`sd3_5/runner.py:93`(带 `name`)、`wan_2_1/runner.py:148` | sd3_5 多一个 `name` 参数 | `diffusion/common/`，可选 `name`/`custom_message` 参数 |
| `_embed` + `_tokenize_prompts` padding 段 | Janus `runtime.py` 与 NextStep `runtime.py`（`_embed` 逐字节相同） | family 字符串 | `ARPipelineExecutorBase` 默认实现 + 共享 `_align_tokenizer_output` |
| `_ar_runner` | Janus / NextStep `runtime.py`（仅 runner 类型 + family 字符串不同） | runner 类、family 名 | 基类模板方法 `_create_ar_runner(family_name, sampling)`（family 已是类属性） |
| `_to_builtin` | `generation/ray/launcher.py:407` 与 `rollouts/collector/config.py:162` | launcher 版处理 Mapping 更全 | `vrl/utils/config.py`，以 launcher 版为准 |
| `_torch_dtype` | `rewards/models/kling_video_reward.py:729` 与 videocon_physics `:268` | 无 | `vrl/rewards/models/_video_utils.py`（保留跨视频家族一致 shape） |
| `_proportional_*counts` / `_interleave_*rows` | `scripts/data/danbooru.py` 内部 | key 类型 tuple vs str / pop 方向 | `_proportional_distribution` / `_round_robin_interleave(pop_from_end=bool)` |
| `JANUS_IMAGE_PIXEL_SIZE = 384` | `janus_pro/model.py:57` | 硬编码字面量 | 派生 `int(JANUS_IMAGE_TOKEN_NUM**0.5) * JANUS_IMAGE_PATCH_SIZE`（=24×16=384，注释自写「→ 384 px」） |

## 2. 分步实施

每行一个小任务，互相独立，可分批做。**下沉前必先 grep 全部出现点确认逐字节一致**——若发现某家族其实有意偏离，那不是重复，跳过它并在注释记录差异。

### T1 [收益最大] `apply_lora` → `LoraModelMixin`
- 6 处实现（5 家族 + base no-op）；先 diff 5 个非 no-op 版本确认仅 `init_lora_weights` 不同。
- 新建 `vrl/models/diffusion/common/lora.py` 的 `LoraModelMixin.apply_lora(spec)`，把 `init_lora_weights` 做成 mixin 的类属性或 `_lora_init_weights()` hook。
- 各家族 model 类 mixin 它，删本地实现；`base.py:193` 的 no-op 默认保留（非 LoRA 模型）。
- 注意 `predict2_5/model.py` 有**两个** `apply_lora`（:185 和 :584，两个模型类）——都收口。

### T2 `_require_tensor` → `diffusion/common/`
- 4 处；sd3_5 版带 `name`。统一签名 `_require_tensor(value, name=None)`，`name` 进错误信息。
- 放 `vrl/models/diffusion/common/tensors.py`，4 个 runner 改 import。

### T3 AR executor 基类模板方法（`_embed` / `_ar_runner` / `_tokenize_prompts` padding）
- 确认 `ARPipelineExecutorBase`（grep）是 Janus/NextStep 的共同基类。
- `_embed` 逐字节相同 → 提到基类默认实现。
- `_ar_runner` 差异只在 runner 类 + family 名（family 已是类属性）→ 基类 `_create_ar_runner(sampling)` 模板方法，子类只声明 `_runner_cls`。
- `_tokenize_prompts` 的 padding 段抽 `_align_tokenizer_output`，主体留子类（tokenizer 调用可能家族相关）。

### T4 `_to_builtin` → `vrl/utils/config.py`
- 以 launcher 版为准（处理 Mapping 更通用）。launcher + collector/config 两处改 import。grep 是否还有第三处。

### T5 `_torch_dtype` → `vrl/rewards/models/_video_utils.py`
- kling + videocon_physics 两处下沉，保留跨视频 reward 家族一致 shape。与子 sprint B 的 T5（kling 拆分）协同——拆 kling 时顺手抽走它。

### T6 danbooru 内部 helper 合并
- `_proportional_*counts` → `_proportional_distribution`；`_interleave_*rows` → `_round_robin_interleave(pop_from_end=bool)`。与子 sprint B 的 T1（danbooru 拆分）**合做**，别单独开。

### T7 `PIXEL_SIZE` 派生
- `janus_pro/model.py:57`：`JANUS_IMAGE_PIXEL_SIZE = int(JANUS_IMAGE_TOKEN_NUM ** 0.5) * JANUS_IMAGE_PATCH_SIZE`，附注释说明派生关系。
- **只改这一个**——`PATCH_SIZE=16`/`TOKEN_NUM=576`/`VOCAB_SIZE` 是真实架构维度常量，保留字面量。

## 3. 测试策略

- 每个 T 下沉前 grep 全部出现点逐字节比对；下沉后跑对应家族用例（`tests/models/`、`tests/rewards/`、`tests/generation/ar/`）。
- 行为零变化——下沉是纯重构。
- T7 加一行断言 `JANUS_IMAGE_PIXEL_SIZE == 384` 防派生公式写错。

## 4. Non-Goals（这些是有意一致，严禁合并）

- **三套 sampling-state dataclass**（diffusion 各家族）——并行 shape 提升 grepability，保留。
- **并行 GRPO 变体**——有意一致的 signal-handling 结构，不扁平化。
- **`MISSING` sentinel 各文件本地重建**、**segment 名 default_factory lambda 重复**——审计判为 per philosophy 可接受。
- **`_import_from_path` 在 worker.py / launcher.py 各一份**——不同子系统、不同错误处理，保留。
- **`_copy_adapter_weights`（cosmos/predict2_5）**——同文件 3 次调用的合法 file-internal factoring。

## 5. 关键参考文件

- diffusion：`vrl/models/diffusion/{wan_2_1,sd3_5,cosmos/predict2,cosmos/predict2_5,cosmos/anima}/{model,runner}.py`、`base.py`
- AR：`vrl/models/ar/{janus_pro,nextstep_1}/runtime.py`、`ARPipelineExecutorBase`（grep）
- 共享落点：`vrl/models/diffusion/common/`（新建）、`vrl/utils/config.py`、`vrl/rewards/models/_video_utils.py`（新建）
- 常量：`vrl/models/ar/janus_pro/model.py:54-59`
