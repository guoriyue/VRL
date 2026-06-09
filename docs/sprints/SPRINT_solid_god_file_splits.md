# SPRINT: God-file 拆分（5 个文件 → 职责清晰的模块）

状态：planned。父：`SPRINT_solid_architecture_audit.md`（子 sprint B）。

## 0. Core Decision

把 5 个「契约方法 + 基础设施细节混在一个对象里」的 god-file，按隐藏职责拆成多个内聚单元，入口退化为薄 facade。

- **原则**：拆分目标是**每个新单元有独立的变更原因**（SRP），不是单纯减行。facade 仍暴露原有公共契约，外部调用点零改动。
- **次序**：按「确定性 × 收益」排，T1（danbooru）确定性最高，T4/T5 涉及热路径要谨慎。
- 每个 god-file 是独立任务，可分别开 PR。

## 1. 五个目标（实测规模）

| # | 文件 | 规模 | 隐藏职责 |
|---|---|---|---|
| T1 | `vrl/scripts/data/danbooru.py` | 1798行/65函数 | 5 数据管线 + IO + 报告 |
| T2 | `vrl/ray/resources.py` | 932行/26函数 | 7 职责 |
| T3 | `vrl/nn/modules/ar_decoder.py` → `VllmDecoderPagedAttentionBackend` | 612行/20方法 | 协议实现只占 2，其余 18 是基础设施 |
| T4 | `vrl/models/ar/janus_pro/model.py` → `JanusProModel` | 1300行/31方法 | 8 类职责 |
| T5 | `vrl/rewards/models/kling_video_reward.py` | 770行 | 类只 4 方法，模块混 5 关注点 |

## 2. 分步实施

### T1 [最高确定性] `danbooru.py` → 3 Dataset + Metadata IO + ReportWriter
隐藏 5 个独立数据管线 + IO + 报告层；350+ 行模块级常量（`ANATOMY_*`/`SAFETY_*`/`POSITIVE_*` + 共享 vocab）混在一起，加一个数据集类型要新增 15+ 常量（这同时是 OCP 违例，见 §子 sprint D 的 `build_anatomy/safety_prompts` 重复）。

**拆成**：
- `AnatomyDataset` / `SafetyDataset` / `PositiveImagesDataset`——各自 `build + split + prompt`，共享基类承载 `download → build_rows → split → write` 编排模板（消除 `build_anatomy_prompts` vs `build_safety_prompts` 的整段复制）。
- `DanbooruMetadata`——`iter_metadata` / download IO 层。
- `ReportWriter`——所有 `write_*_report`。
- 共享 vocab → `TagVocabulary`；`ANATOMY_*`/`SAFETY_*` → 各自 `@dataclass Config`。
- `danbooru.py` 退化为 facade / registry。

**注意**：这是 `vrl/scripts/`，是脚本不是库。先 grep 它的入口（CLI / `__main__`）确认对外接口，facade 要保住 CLI 行为。

### T2 `ray/resources.py` → 5 类 + 薄编排
隐藏 7 职责：config 解析归一、GPU 设备解析、role 设备分配、worker 计数+整除校验、reward 专属推理需求、placement bundle 计算、38 条约束校验。

**拆成**：`ResourceConfigParser` / `DeviceResolver` / `WorkerCountResolver` / `RewardPlacementCalculator` / `ResourceValidator`；`resolve_distributed_resources` 留作薄编排（依次调用 5 个单元）。每个单元可单测——当前 932 行无法单独测某一步。

### T3 `VllmDecoderPagedAttentionBackend` → 协议薄壳 + 3 助手
实现 `ARAttentionBackend` 协议只占 `prefill`/`step` 2 方法，其余 18 方法是基础设施（六组状态字段一一对应六组职责）。

**拆成**：
- `ARVllmDecoderBackend`——实现协议 + 编排 prefill/step（薄）。
- `VllmPagedKVManager`——KV block 生命周期状态（`_allocate_blocks`/`_ensure_kv_caches`/`_next_block_id`/`_kv_caches`）。
- `VllmRequestPacker`——无状态输入打包（`_pack_prefill`/`_pack_step` + 校验）。
- `HFTrunkIntrospector`——HF 模型内省以 property 暴露（`_num_attention_heads`/`_head_dim`/`_sliding_window_for_layer`）。
- `_rotate_half`/`_apply_rotary_pos_emb` 是 **kernel 不是 backend 逻辑** → 移入 `vrl/nn/kernels/attention/rotary`。

**风险**：这是 AR 解码热路径。拆分必须**逐位数值不变**——复用 `tests/generation/ar/test_*_vllm_paged_attention_backend.py` 和 `tests/nn/layers/test_paged_attention_contract.py` 做回归网，拆分前抓 golden tensor。与已落地的 `ARAttentionBackend` / `resolve_attention_backend` 边界对齐。

### T4 `JanusProModel` → 4 helper + 薄 facade
隐藏 8 类职责：tokenizer 工具（`_pad_token_id`/`_eos_token_id`/`_encode_text_ids`/`_looks_like_bos`）、embedding/text、reflection 采样（`_sample_selfcheck_text` 80+行）、prompt padding（`_left_pad_replay_context`）、VQ 通道解析、logit 计算、replay 编排、`generate_with_refine`（308行）。

**拆成**：`JanusProTokenizerUtils`（静态）、`JanusProReflectionSampler`、`JanusProPromptPreparer`、`JanusProVQHelper`；`JanusProModel` 留薄 facade，仅暴露公共契约（`forward_image_logits`/`replay_forward`/`generate_with_refine`/`decode_image_tokens`/LoRA）。用 helper-bundle 组合避免构造参数爆炸。
- `generate_with_refine` 的 308 行 OCP 拆分（提取 `RefinementPolicy`）**与本任务合做**——见子 sprint D 的 D3。

### T5 `kling_video_reward.py` → 3 类（收窄，勿过拆）
模块把 checkpoint 加载/remap（5函数）、config 校验（3函数）、processor+LoRA 构建（1函数70行）、prompt 模板、归一化全堆一处。

**拆成**：`_KlingConfigLoader`（configs + `_from_dataclass` + `_load_configs`）、`_KlingCheckpointManager`（路径解析 + state-dict remap + adapter 注入）、`_KlingProcessorFactory`（model + processor + LoRA）。
**保留**：`_resolve_model_root` / `_build_video_reward_prompt` / `_torch_dtype` 是 Kling 家族专属校验/模板，**留在模块内**——不要拆成 5 类（过度设计）。其中 `_torch_dtype` 与 videocon_physics 重复，下沉到 `_video_utils.py` 见子 sprint C。

## 3. 测试策略

- **每个 T 拆分前先抓 golden / 跑现有用例**，拆分后逐位一致（尤其 T3/T4 在热路径）。
- T1：grep `tests/` 下 danbooru 相关用例 + CLI 冒烟。
- T3：`tests/generation/ar/`、`tests/nn/layers/test_paged_attention_contract.py`。
- T4：`tests/models/test_janus_*`。
- 拆分纯属**结构搬运**，零行为变化——任何输出 diff 都是 bug。

## 4. Non-Goals

- **不拆** `anima/model.py`(664)、`trajectory/validation.py`(418)——见父 sprint §6（有意 bundling / 内聚尚可）。
- T5 不拆成 5 类——`_resolve_model_root`/`_build_video_reward_prompt`/`_torch_dtype` 留模块内。
- 不改任何对外公共契约——facade 保住所有现有入口签名。
- 不顺手「优化」拆出来的逻辑——本 sprint 只搬运、不改算法。

## 5. 关键参考文件

- `vrl/scripts/data/danbooru.py`、`vrl/ray/resources.py`、`vrl/nn/modules/ar_decoder.py`、`vrl/models/ar/janus_pro/model.py`、`vrl/rewards/models/kling_video_reward.py`
- 协议：`vrl/nn/layers/attention/paged.py`（`ARAttentionBackend`）
- 回归网：`tests/generation/ar/test_*_vllm_paged_attention_backend.py`、`tests/nn/layers/test_paged_attention_contract.py`、`tests/models/test_janus_*`
- 相关边界：`vrl/nn/layers/attention/paged.py`（`ARAttentionBackend`）、`vrl/nn/modules/ar_attention_backends.py`（backend selector）
