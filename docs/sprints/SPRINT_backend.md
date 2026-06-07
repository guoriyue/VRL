# SPRINT: Real attention-backend dispatch (SGLang/vLLM-style)

状态：done（T1–T4 已落地，2026-06-06；后续细粒度优化见 `SPRINT_attention_kernel_medium.md`）。

## 0. Core Decision

把 AR 解码的注意力从"一个布尔开关 + inline 的 naive 分支"升级成 **SGLang/vLLM 式的按名字选择的 attention-backend dispatch**:每条解码路径都是一个实现了同一协议的 backend,由 AR backend selector 按名字选择。

- **为什么**:现在 naive 路径 inline 在 runner 里、选择是 `use_vllm_paged_attention: bool`,加第 3 个 backend(flashinfer / triton / flash-attn 直连)要改 runner 内部。真 dispatch 后,加 backend = 写一个 impl 类 + selector 映射一行,runner 不动。
- **范围**:只动 AR 注意力解码这一处(janus / nextstep runner + 它们的 runtime 选择点 + AR backend builder)。diffusion 的 SDPA(单实现)不碰。
- **前置**:命名已对齐(`ARAttentionBackend` / `attention_backend`,见 commit `97dfd39`)。本 sprint 接着把**机制**补上。
- **关键纪律**:naive 与 paged 两条路必须**逐位数值不变**——已有回归网(见 §4)。

## 1. 现状 → 目标

### 现状(三个 gap)
1. **naive 不是 backend**:它是 runner 里的 `if self.attention_backend is None:` 分支
   (`janus_pro/runner.py:108,319`、`nextstep_1/runner.py:98,259`),用 HF `past_key_values`,
   存在引擎的 `cache_lanes`。只有 vLLM-paged 是真 backend(走 `ARAttentionBackend.prefill/step`)。
2. **选择是布尔不是名字**:`<family>/runtime.py:_ar_runner` 读 `sampling["use_vllm_paged_attention"]`。
3. **没有按名字选择**:vLLM backend 由 `build_<family>_vllm_attention_backend(...)` 直接构造。

### 目标
- 一个 `ARAttentionBackend` 协议(已有);**两个**实现:`VllmDecoderPagedAttentionBackend`(已有)+ 新的 `TorchNaiveAttentionBackend`。
- 一个 selector `{name: shared builder}`;runner **永远**持有 backend,无 `None` 分支。
- 配置 `attention_backend: str`(= SGLang 的 `--attention-backend`)取代布尔。

## 2. 目标架构

```
sampling.attention_backend = "vllm_paged" | "torch_native"     ← 选择(名字)
        ↓ resolve_attention_backend(family, name, model, **kw)  ← AR backend selector
ARAttentionBackend (协议, vrl/nn/layers/attention/base.py)       ← 抽象
   ├── VllmDecoderPagedAttentionBackend  → VllmPagedAttentionKernels   (impl: 分页)
   └── TorchNaiveAttentionBackend        → HF past_key_values forward  (impl: naive) ← 新增
        ↑ runner 只调 .prefill()/.step()/.free(),不知道是哪个 impl
```

协议(已存在,新 backend 照着实现)——`vrl/nn/layers/attention/paged.py`:
```
def __init__(self, config: ARAttentionConfig)
def prefill(self, request: ARAttentionPrefillInput) -> ARAttentionPrefillOutput
def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput
def free(self, sequence_states: Sequence[Any]) -> None
def debug_info(self) -> Mapping[str, Any]
```

## 3. 分步实施

### T1 [最高价值·最大重活] naive 路径 → `TorchNaiveAttentionBackend`
把 runner 里 `attention_backend is None` 的逻辑搬进一个 backend impl。

**映射**:
| 现在(runner 内) | 搬到(backend) |
|---|---|
| `_prefill_ar_prompt(embeds, mask)` → `(past, last_hidden)` | `prefill()` → `ARAttentionPrefillOutput(last_hidden, sequence_states=(past,))` |
| `_advance_kv_cache_after_sample` 的 else(naive)分支 | `step()` → `ARAttentionStepOutput(last_hidden, sequence_states)` |
| `cache_lanes={"cond_past","uncond_past"}` | 统一进 `sequence_states`(协议里 `sequence_states: tuple[Any,...]` 是**不透明**的,HF `past_key_values` 直接塞进去) |

**关键设计点**:
- `ARAttentionPrefillOutput.sequence_states` 不透明 → 不用改协议,naive 把 `past_key_values` 当 sequence_states 携带。
- naive 的 attention mask 增长用现成的 `append_attention_token`(已在 `paged_attention_helpers.py`)。
- naive 不需要 block_size / cache_dtype;`ARAttentionConfig` 这些字段对它是 no-op(或忽略)。
- 每个 family 一个 builder(`build_<family>_torch_naive_backend`),因为 backend 包的是 family 的 trunk(`model._lm_trunk()`),不是纯 kernel——这是与 SGLang(model-agnostic backend)的**结构差异**,接受它。

**风险**:动热解码循环;naive 的 KV-lane 模型(引擎 `cache_lanes`)和 paged 的(state 内 `sequence_states`)不同,统一时要保证引擎调度仍正确。

### T2 [接 T1] 清理 runner —— 删掉所有 `None` 分支
- `init_ar` / `step_ar` / `_advance_*` 里 `if self.attention_backend is None:` 全删,一律 `self.attention_backend.prefill()/.step()`。
- `require_attention_backend` 守卫不再需要(backend 永远在)→ 删掉它(`paged_attention_helpers.py` + 调用点)。
- runner 变成 backend-agnostic,不再有 naive-vs-paged 的特判。

### T3 [接 T2] 按名字选择 backend
选择逻辑放在 `vrl/nn/modules/ar_attention_backends.py`,不要再单独建 `vrl/nn/layers/attention/registry.py`:
```python
# vrl/nn/modules/ar_attention_backends.py
def resolve_attention_backend(family, name, model, **kw) -> ARAttentionBackend: ...
```
- `"vllm_paged"` → shared `build_vllm_attention_backend`。
- `"torch_native"` → shared `build_torch_native_backend`。
- `<family>/runtime.py:_ar_runner` 调
  `resolve_attention_backend(family, sampling["attention_backend"], self.model, block_size=..., cache_dtype=...)`。

### T4 [接 T3] 配置迁移(`--attention-backend` 等价物)
- `use_vllm_paged_attention: bool` → `attention_backend: str`(默认 `"vllm_paged"`)。
- back-compat:短期同时认旧键——`use_vllm_paged_attention: false` 映射成 `attention_backend: "torch_native"`;迁移完所有 config 后删旧键。
- 配置键集中在 sampling;`ar_paged_block_size` / `ar_paged_cache_dtype` 作为 vllm_paged 的 backend-specific 参数透传。

### 可选(等真有第 3 个 backend 再做)
- **T5 能力校验**:照 vLLM `AttentionBackend.validate_configuration()`——每个 backend 声明支持的 dtype/head_size,选择时校验报错。
- **T6 base/impl 文件拆分**:`ARAttentionBackend` 协议从 `paged.py` 挪到 `base.py`(对应 SGLang `base_attn_backend.py`);每个 impl 独立文件(`vllm_paged.py` 已有,`torch_native` 已在 `vrl/nn/modules/torch_attention.py`)。

## 4. 数值不变性 / 测试策略

**核心要求**:T1/T2 完成后,naive 与 paged 输出必须和重构前逐位一致。

现成回归网(直接复用):
- `tests/e2e/test_real_checkpoint_rl.py` —— 用 `use_vllm_paged_attention=false` **专跑 naive 路径**(迁移后改成 `attention_backend=torch_native`)。
- `tests/models/test_janus_kv_decode.py` —— 无 backend 的 naive runner。
- `tests/generation/ar/test_janus_paged_attention_one_step.py` + `test_*_vllm_paged_attention_backend.py` —— paged 路径。
- `tests/nn/layers/test_paged_attention_contract.py` —— 协议契约。

新增:
- `TorchNaiveAttentionBackend` 的 prefill/step 契约测试(对齐 paged backend 的契约测试)。
- **等价性测试**:同一输入,naive backend 的输出 == 重构前 runner naive 分支的输出(可在重构前先抓 golden tensor)。

## 5. 路线图 / 次序

**T1 → T2 → T3 → T4**。只有 **T1 是真重活**(其余是接线)。每步跑完上面的回归网再进下一步。T5/T6 推迟到第 3 个 backend 出现。

里程碑:
- M1(T1+T2):naive 是 backend、runner 无 None 分支、两路数值不变 —— **结构到位,行为不变**。
- M2(T3+T4):按名字选择 + 注册表 + 配置迁移 —— **对外是真 `--attention-backend`**。
- M3(可选):加一个真正的新 backend(如 flashinfer 直连)验证 dispatch —— 只写 impl + 注册一行,runner 零改动 = 成功标准。

## 6. Non-Goals

- 不碰 diffusion SDPA(单实现,不需要 backend)。
- 不搬 SGLang 的 model-agnostic backend 结构(我们的 backend 包 family trunk,保持 per-family builder)。
- 不加 `--prefill-attention-backend` / `--decode-attention-backend` 这类细粒度分离(YAGNI)。
- 不重写 kernel 层(`VllmPagedAttentionKernels` 等不动)。
- 不在没有第 2 个 backend 时做 T5/T6(过度设计)。

## 7. 关键参考文件

vrl(实现时查):
- 协议 + 数据类型:`vrl/nn/layers/attention/paged.py`(`ARAttentionBackend` / `ARAttentionConfig` / `ARAttention{Prefill,Step}{Input,Output}`)
- 现有 impl + kernel:`vrl/nn/modules/ar_decoder.py`(`VllmDecoderPagedAttentionBackend`)、`vrl/nn/kernels/attention/vllm_paged.py`(`VllmPagedAttentionKernels`)
- runner(消费 backend):`vrl/models/ar/janus_pro/runner.py`、`vrl/models/ar/nextstep_1/runner.py`
- 选择点:`vrl/models/ar/<family>/runtime.py`(`_ar_runner`)
- 共享工厂:`vrl/nn/modules/ar_attention_backends.py`(`resolve_attention_backend` / `build_*_attention_backend`)
- 共享 helper:`vrl/models/ar/paged_attention_helpers.py`(`append_attention_token`)
- selector 不单独建 `vrl/nn/layers/attention/registry.py`;放在 `vrl/nn/modules/ar_attention_backends.py`

SGLang / vLLM(对齐参照):
- vLLM `AttentionBackend` 注册表:https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py
- SGLang 抽象基类 `base_attn_backend.py` + impl(`flashinfer_backend.py` / `triton_backend.py` / `torch_native_backend.py`):https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention
- SGLang 服务参数(`--attention-backend` 等一族):https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md

---

## 附录:SGLang/vLLM 的 `*-backend` 命名一族(为什么叫 attention-backend)

"backend" 永远带子系统前缀,且只在"该子系统有多种可换实现"时才有;MLP 是单一 matmul,所以没有 mlp-backend。

| 标志 | back 的是什么 | 可选实现 |
|---|---|---|
| `--attention-backend` | 注意力 | flashinfer / triton / fa3 / fa4 / flashmla / torch_native / flex_attention … |
| `--sampling-backend` | 采样 | flashinfer / pytorch |
| `--grammar-backend` | 受限/语法解码 | outlines / xgrammar / llguidance |
| `--moe-runner-backend` | MoE 的 grouped-GEMM | flashinfer_trtllm / deep_gemm |

我们对齐的是 `--attention-backend`:抽象层叫 `attention backend`(不带 "paged"),具体实现层保留 "paged"(它是 vLLM 分页实现细节)。
