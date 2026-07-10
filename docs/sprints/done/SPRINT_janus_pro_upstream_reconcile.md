# SPRINT: janus_pro family 对账 upstream Janus —— CFG / 并行解码 / VQ 解码

状态：**DONE（Phase 1，2026-06-29）；Phase 2 GPU probe 是可选后续，不阻塞完成**。目标：把自研 `janus_pro` AR 图像生成 family 与 upstream DeepSeek `Janus` 逐项对账(CFG、解码循环、VQ 解码、KV/paged、RL log_prob),把"能借的细节"和"潜在 bug"落实。

> **Phase 1 落地（2026-06-29）**：两条"对账结论"已固化成注释 + 契约测试,防止有人"参考 upstream"改回去:
> - `runner.py` log_prob 处加 RL 正确性注释(sample 自 `guided`、score 自 `cond_logits`,改成 `guided` 会破 old_log_prob 不变量)。
> - `model.py:decode_image_tokens` 处加注释(latent 通道必须动态解析,别学 upstream 硬编码 `8`)。
> - 新测试 `tests/models/ar/janus_pro/test_upstream_reconcile_contracts.py`(4 条,纯 CPU stub):锁 log_prob 来自 cond 不是 guided + 通道动态解析(11≠8)+ override 优先 + 无 quantizer 即 raise。
> - **falsifiability 已实证**:把 `cond_logits`→`guided` 测试红(lp -0.44→-0.05);把动态解析→`return 8` 测试红(`8≠11`);均已还原。janus_pro 目录测试已通过。
> - Phase 2(单次 batched prefill 微优化 probe)需 GPU 实测墙钟占比,未做,见 §2。

## 0. 结论先行（对账已做一轮,基本是 parity 或我们更好）

逐项读过两边代码后:**我们的 `janus_pro` 在正确性上已经 ≥ upstream**,只有一处**值得 probe 的微优化**,没有需要紧急修的 bug。诚实结论:这是一个"验证 + 一个小优化探针"sprint,不是大改。

| 维度 | 我们 (`vrl/models/ar/janus_pro/`) | upstream `Janus` | 结论 |
|---|---|---|---|
| AR 解码循环 | 逐 token + paged-attention 步进 | 逐 token + HF 原生 KV-cache | parity(都是单 token/forward;upstream 的 `parallel_size` 是 batch 维,不是 token 并行) |
| CFG 公式 | `uncond + scale*(cond-uncond)` | 同 | **完全一致** |
| RL log_prob | 从 **cond_logits** 取(采样用 guided,打分用 cond) | 不计算(纯推理脚本) | 我们对、且是 RL 正确做法 |
| VQ 解码 latent 通道 | **动态解析** `_resolve_vq_latent_channels()` | **硬编码 `8`** | **我们更稳**(硬编码在变体上会静默出错) |
| cond/uncond prefill | **两次独立 prefill forward**(cond、uncond) | **一次 batched forward**(交错 even/odd) | ← **唯一可借项**,见 §2 |

## 1. 证据

### 1.1 CFG 公式两边一致

我们 `runner.py:222-238`：

```python
cond_logits, uncond_logits = logits.chunk(2, dim=0)
guided = uncond_logits + state.guidance_scale * (cond_logits - uncond_logits)
probs = F.softmax(guided / state.temperature, dim=-1)
sampled = torch.multinomial(probs, num_samples=1)
log_probs = F.log_softmax(cond_logits / state.temperature, dim=-1)   # 从 cond 取 lp(RL 正确)
```

upstream `generation_inference.py:84-87`：

```python
logit_cond = logits[0::2, :]; logit_uncond = logits[1::2, :]
logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)     # 同一公式
```

→ 公式一致;我们额外正确地**从 cond 分布取 log_prob**(采样自 guided、打分自 policy),这是 importance-sampling 正确性所需,upstream 推理脚本不涉及。

### 1.2 VQ 解码:我们动态、upstream 硬编码（我们更稳）

我们 `model.py:1005-1026` + `_resolve_vq_latent_channels()`(1028-1062):从 live quantizer 的 `embedding.weight` 推 latent 通道,再 `vq_model.decode_code(ids, shape=[B, C, side, side])`。

upstream `generation_inference.py:98`：`decode_code(..., shape=[parallel_size, 8, H, W])` —— **硬编码 8**。

→ upstream 脚本是对单一目标手调的;硬编码通道在不同 quantizer 维度的变体上会静默 reshape 成错形、出垃圾图。**我们的动态解析是正确的**,本 sprint 顺手把这点写成注释/契约,防止有人"参考 upstream"改回硬编码。

### 1.3 唯一可借项:prefill 的单次 vs 两次 forward

我们 `runner.py:76-93`：cond / uncond **各做一次** `_prefill_ar_prompt_paged`,得到两套 paged sequence_states。

upstream `generation_inference.py:69-80`：把 `[cond,uncond]` 交错拼成 `2B` batch,**一次** `language_model.model(...)` forward 完成 prefill。

→ 我们的两次独立 prefill 是 paged-attention 分 lane 的实现选择;**是否能合成一次 batched prefill 省时间,取决于 paged seam 是否支持 cond/uncond 同 batch**。这需要 probe,不是无脑改。

## 2. 应该做什么

### Phase 1 — 固化对账结论（低风险,先做）

- 在 `model.py` VQ 解码处加注释:latent 通道**必须**动态解析,显式标注"不要学 upstream 硬编码 8(变体会静默坏)"。
- 在 `runner.py` log_prob 处加注释:`log_prob 取自 cond_logits 而非 guided`——RL 正确性契约,防止有人"修"成 guided。
- 若现有 `tests/models/ar/janus_pro/*` 未锁这两条,补**逻辑契约测试**(动态通道解析在 mock quantizer 维度变化下仍对;CFG log_prob 来自 cond 分布)。

### Phase 2 — prefill 单次 forward 微优化 probe（MFU-gated,可选）

- 一次性 `*_probe`：测当前"两次独立 prefill"在目标 batch/分辨率下的 prefill 墙钟占比。
- 若 prefill 在 rollout 里占比可观,试把 cond/uncond 合成**一次 batched paged prefill**,实测加速。
- **MFU 北极星门**：只有当 probe 证明 prefill 是真实开销、且合并不引入 critical-path 数据搬运时才落地;否则记 negative,**不改 working 架构**。

## 3. 非目标

- 不改 CFG 公式、不改采样/温度、不动 log_prob 来源(这些是正确的,改了会破 RL)。
- 不把 paged-attention 退回 HF 原生 KV-cache(upstream 那套对单样本推理简单,但我们多样本 rollout 需要 paged 的显存效率)。
- 不为"代码长得像 upstream"而重构;parity 已确认,无需对齐写法。

## 4. 验收

- [x] VQ 动态通道、CFG-log_prob-from-cond 两条契约有注释 + 逻辑测试守护(改坏源码→测试红)。**已落地 2026-06-29**:注释在 `runner.py` log_prob 处 + `model.py:decode_image_tokens` 处;测试 `tests/models/ar/janus_pro/test_upstream_reconcile_contracts.py`;两条 falsifiability 都实证过(见状态块)。
- [ ] Phase-2 probe 数据落本文件:prefill 墙钟占比 + 合并 forward 的实测加速(或 negative)。**GPU-gated,未做。**
- [ ] 一句话结论:是否采纳单次 prefill,依据 MFU 北极星。**待 Phase 2。**

**参考**：`vrl/models/ar/janus_pro/{model.py:1005-1062,runner.py:60-124,222-296}`;`~/Desktop/deep-research/Janus/{generation_inference.py:55-110,janus/models/vq_model.py:284-299,505-508}`。
