# SPRINT: AR 同-prompt 共享前缀 prefill（GRPO group 的前缀复用）

Status: **PARKED / profiling-triggered (2026-07-18)**. Start only when a real
causal-token workload shows prefill at 15–20% or more of rollout wall time. This
is a conditional optimization: only when prefill
（prompt 段）在 AR rollout 里占到有意义的 wall-clock 比例时才做。对当前的**图像生成 AR**
（janus_pro / nextstep_1，短 prompt → 长 image-token decode）ROI 低；对**长-prompt AR**
（文本推理 / 长 system prompt / janus_pro_r1 多段）ROI 才显著。先量再做。

> The current paths are `vrl/models/families/{janus_pro,nextstep_1}/{runtime,runner}.py`,
> `vrl/models/steps/token/paged_attention_helpers.py`, and `vrl/families/registry.py`.
> The audit also followed the rollout-side
> 同-prompt 分组（`SampleChunk` + `n_samples_per_prompt`）。

---

## 0. 一句话

GRPO 要求同一个 prompt 出 K 个 variant 组成一个 group（组内相对优势）。AR rollout 现在把这 K 个
variant 复制成 **K 行相同 prompt** 一起 prefill —— **同一个 prompt 的 KV 被冗余计算了 K 遍**。理想
做法是 **prefill 一次、把 prompt 的 KV 共享/广播给 K 个 decode lane**（vLLM `n` 采样那套）。这个改动
**算法无关**（rollout 层做，GRPO 不参与），但收益随 **prompt 长度 / decode 长度之比** 变化，所以
profiling-gated。

---

## 1. 现状（带证据）

### 1.1 同-prompt K 个 variant 被复制成 K 行

`vrl/models/families/janus_pro/runtime.py`（chunk 路径，nextstep_1 同形）：

```python
repeated_prompts = [chunk.prompt] * chunk.sample_count   # 同一个 prompt 复制 K 份
cond_embeds, ... = encode(repeated_prompts, ...)         # 编成 [K, seq, dim]
init_ar(cond_embeds, uncond_embeds, ...)                 # K 行一起 prefill
```

`runner.init_ar`（`janus_pro/runner.py:74-93`）`batch_size = cond_inputs_embeds.shape[0]`（= K），
`_prefill_ar_prompt_paged(cond_inputs_embeds, ...)` 对**整批 K 行**做 prefill，每行都是**同一个
prompt**。当前 prefill 仍执行 cond+uncond 两次 branch forward，**每次前向内部仍在为 K 个相同行
各算一遍 prompt KV**。

### 1.2 paged backend 没有前缀去重

`vrl/models/steps/token/paged_attention_helpers.py` 只是「shared paged-attention helpers reused by runners」
——共享的是**代码**，不是**KV**。没有 radix/prefix-hash/cache-hit 逻辑，所以相同 prompt 前缀不会被
自动复用。实际 paged KV cache 由每个 decode lane 独立持有；代码中没有 cache-kind registry 或
跨样本前缀共享机制。

### 1.3 已经做对的部分（别重复造）

- **decode 已 K-batched**：K 个 lane 逐 token 一起解（paged attention），decode 不冗余。

所以**唯一的冗余是 prompt 段的 attention prefill 被算了 K 遍**，decode（贵的部分）不冗余。

---

## 2. ROI 分析（为什么 gated）

设 prompt 长 `P`、生成长 `G`（janus `image_token_num`，通常几百）。prefill ~ O(P)，decode ~ O(G)，
都按 K 批。共享前缀把 prefill 从 K×O(P) 降到 1×O(P)：

| 场景 | P/G | prefill 占比 | 共享前缀收益 |
|---|---|---|---|
| **图像 AR（janus/nextstep 现状）** | 短 prompt / 长 image-token（P≪G） | 小 | **低**（不值得单独做） |
| **长-prompt AR**（文本推理、长 system prompt、R1 多段） | P 与 G 同量级或更大 | 大 | **显著**（K×长前缀 → 1×） |

**门槛**：P0 先量 prefill / 总 rollout 的 wall-clock 占比；只有 ≳ 15-20% 才进 P1。

---

## 3. 设计（rollout 层，算法无关）

把「K 行相同 prompt prefill」换成「prefill 一次 + KV 给 K 个 decode lane」。两档实现，按 ROI/复杂度选：

**A. 复制 KV（简单，省 compute 不省显存）**
- prefill 用 batch=1（cond）+ batch=1（uncond）→ 得 prompt 的 paged KV（`sequence_states`）。
- 把这份 KV **复制**到 K 个 decode lane（K 份 KV 内存，但 prefill 只算一次）。
- decode 照旧 K-batched。
- 改动点：`runner.init_ar` 接收 `num_samples`，prefill 单行后 fan-out KV 到 K lane；runtime 不再
  `[chunk.prompt] * sample_count`，只传 1 行 prompt + `sample_count`。

**B. 共享 KV 页（高级，省 compute 也省显存）**
- prefill 单行 → K 个 decode lane **共享只读** prompt KV 页（copy-on-write 仅对 decode 段），即 vLLM
  automatic-prefix-caching 的形态。
- 改动点：`paged_attention_helpers` 支持多 sequence 指向同一前缀页 + decode 段各自分页。
- 复杂度高，留作 P2（仅当显存也成瓶颈时）。

**P0 baseline 选 A**（compute-once，最小改动；显存按 K 不变,因为 decode 段本来就 K 份）。

---

## 4. 实现计划

### P0 — profiling gate（先量，别盲做）
- 用 stage `engine.prefill` 计时，在目标 AR recipe 上量 **prefill 占总 rollout wall-clock 的比例**。
- 验收：占比 < 15% → **不做**（记录结论，关闭）；≥ 15% → 进 P1。

### P1 — 共享前缀 prefill（方案 A，compute-once）
- `runtime`：chunk 路径不再 `[chunk.prompt] * sample_count`；传单 prompt + `sample_count`。
- `runner.init_ar`：prefill 单行 cond/uncond，KV fan-out 到 K decode lane；cond/uncond 两次
  branch forward 不变，但**前向的 batch 维从 K 降到 1**，验收直接比较 `engine.prefill` 墙钟。
- 正确性红线：K 个 lane 的 decode 结果**逐位等于**现状（同 noise/seed/temperature），logprob 不变
  （rollout logprob 是训练的 old_log_prob，必须 bit-parity，否则 GRPO ratio 漂）。
- CPU 单测：1 行 prefill + K lane decode 的 token_ids/logprobs == K 行 prefill 的现状输出。
- 计时：prefill wall-clock 应 ~÷K（在长 prompt 上才看得出）。

### P2 — 共享 KV 页（方案 B，仅当显存瓶颈）
- 仅当 P1 后显存仍是 K×prompt-KV 的瓶颈时做（长 prompt + 大 K）。

---

## 5. 风险 / 非目标

- **红线：rollout logprob bit-parity**。prefill 路径改了不能让 old_log_prob 漂 —— GRPO 的 IS ratio
  靠 old_log_prob == 采样时 logprob。P1 单测必须断言逐位一致。
- **非目标**：不动 GRPO / 算法层（rollout 改动对算法透明，group_ids 契约不变）。
- **非目标**：不做 diffusion 的「共享前缀」——diffusion 没有 KV decode，它的同-prompt 复用已经是
  `repeat_encoded_batch` + `sample_batch_size` 批处理（见 `SPRINT_compile_rollout_lifecycle` / rollout
  efficiency 讨论），不在本 sprint。
- **非目标**：不引入 vLLM/SGLang 依赖；复用现有 `paged_attention_helpers`。

---

## 6. 代码引用（已核实）

- 同-prompt 复制成 K 行：`vrl/models/families/janus_pro/runtime.py`；
  nextstep_1 同形 `vrl/models/families/nextstep_1/runtime.py`。
- prefill 单点：`janus_pro/runner.py:74-93,158-`（`_prefill_ar_prompt_paged`）；
  `nextstep_1/runner.py:93-134,239-247`（`_prefill_paged`）。
- 无前缀去重：`vrl/models/steps/token/paged_attention_helpers.py`。
- AR family/task 分类：`vrl/families/registry.py`；KV-cache 状态与推进：
  `vrl/models/steps/token/paged_attention_helpers.py` 及各 family runner。
- rollout 同-prompt 分组（算法无关，K=group size）：`SampleChunk`（`generation/execution/chunks.py`）
  + `n_samples_per_prompt`（`vrl/config/presets/base/rollout/*.yaml`、`vrl/config/schema.py`）。
