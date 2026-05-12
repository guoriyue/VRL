# SPRINT：AR Rollout KV Cache 与调度优化

这份 sprint 的目标是把自回归视觉模型的 rollout 从“每个 token 重跑完整上下文”升级为“prefill 一次，然后 KV-cache decode”。这不是 Janus 单点优化，而是 AR family 的基础设施：Janus-Pro、Janus-Pro-R1、NextStep-1，以及后续任何长序列 autoregressive image/video model 都应该走同一套能力边界。

## 1. 背景结论

当前 Janus-Pro-R1 Codex-QA profiling 已经说明瓶颈在 rollout generation：

```text
collect                         474.328s
collect.engine_generate         435.502s
collect.reward_score             38.826s
backward                          0.404s
optim_step                        0.045s
```

rollout worker 的 profiler 里还有这些信号：

```text
cudaStreamSynchronize            116.536s
cudaLaunchKernel               1,637,254 calls
Command Buffer Full              71.374s
```

Janus 当前采样路径的问题在这里：

```python
outputs = self._lm_trunk()(
    inputs_embeds=torch.cat([cond_embeds, uncond_embeds], dim=0),
    attention_mask=torch.cat([cond_attn, uncond_attn], dim=0),
    use_cache=False,
)
```

这会导致每个 image token 都重新 forward 全部 text prefix 和历史 image tokens。对于 576 个 image tokens、CFG cond/uncond、R1 initial/final image segment，这个复杂度不合理。

已有代码状态：

- `vrl/engine/ar/token_scheduler.py` 已经有 token-level scheduler 雏形。
- `vrl/models/families/nextstep_1/policy.py` 已经有 `_init_kv()` / `_step_llm()` 的 KV cache path。
- `vrl/models/families/janus_pro/policy.py` 还没有真正用 `past_key_values`，并且每步动态拼接 embedding 和 attention mask。
- Ray rollout worker 已经是常驻 actor 形态，但当前 chunk granularity 仍然是 prompt/sample chunk，不是 token decode engine。

## 2. 总原则

每个长 decode 的 autoregressive model 都应该具备这些能力：

- prefill 一次，decode 阶段每步只喂新 token embedding。
- 每个 active sequence 有独立 decode state，至少包括 position、KV cache、last embedding、采样产物。
- 同一 family/task/dtype/tokenizer/position 的 active sequences 可以合并成 batch。
- rollout worker 尽量常驻模型状态，不在每个 rollout request 中重复构造大对象。
- CPU 不在每 token 路径中做 `.item()`、Python list 拼接、动态 `torch.cat` 全上下文重建。
- `torch.compile`、CUDA graph、自定义 kernel 只能放在 KV decode 稳定之后，不作为第一阶段优化。

不是每个 AR 模型都需要完全一样的实现：

- Janus-Pro 需要 categorical image-code sampling、CFG cond/uncond、VQ decode。
- Janus-Pro-R1 还需要 initial image、self-check text、final image 三段状态。
- NextStep-1 是 continuous token + flow head，已经有部分 KV plumbing，重点是接入统一 scheduler/metrics。
- diffusion family 不进入这个 sprint。

## 3. 非目标

本 sprint 不做这些事：

- 不重写 SGLang。
- 不先接 vLLM/SGLang serving backend。
- 不写 Triton/CUDA 自定义 kernel。
- 不默认打开 `torch.compile`。
- 不把 Codex reward 或 VLM judge 的耗时归因到模型 decode。
- 不改变 GRPO advantage/reward 语义。
- 不追 paper number，只优化 rollout infra。

## 4. 设计目标

### 4.1 统一 AR decode contract

目标是在 `vrl/engine/ar` 和 `vrl/models/ar.py` 层表达一个通用能力：

```text
init_ar_state(...) -> family-specific state
step_ar(state, active_sequences) -> ARStepResult
finalize_ar_state(state) -> family-specific tensors
```

这个 contract 已经存在，但需要升级：

- state 必须能保存和更新 KV cache。
- batch step 不能要求每个 row 都保留完整 historical embeddings。
- scheduler 需要按 `position` 合 batch，避免不同 decode length 混在一起。
- step result 需要携带 profile/debug counters，至少能确认是否真的使用 cache。

### 4.2 Janus KV decode path

Janus 需要改成：

1. 对 cond prompt 和 uncond prompt 各做一次 prefill。
2. 保存 `past_key_values` 和 last hidden。
3. 每个 image token 只调用一次 one-token forward：

```text
inputs_embeds = prepared image token embed
past_key_values = previous cache
use_cache = true
```

4. CFG 仍然使用 cond/uncond logits：

```text
guided = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
sampled = multinomial(softmax(guided / temperature))
old_log_prob = log_softmax(cond_logits / temperature)[sampled]
```

5. 采样得到的 token embedding 同时送入 cond/uncond 的下一步 decode。

### 4.3 常驻 rollout worker

Ray actor 现在已经有 `load_policy()` / `release_policy()`，但 sprint 需要明确：

- 多 GPU机器上默认不使用 `release_after_collect`。
- 单 GPU debug 可以继续 release，避免 trainer/rollout 抢显存。
- weight sync 后只更新 trainable state，不重建整个 policy。
- rollout worker 内部 profiler 继续覆盖 `forward_chunk`，用 before/after trace 验收。

### 4.4 批量 decode scheduler

现有 `ARTokenScheduler` 可以复用，但要升级为真正对 KV decode 有意义：

- batch key 包含 family、task、tokenizer、dtype、max_new_tokens、position、decode backend。
- 同一 position 的 rows 才能合 batch。
- Janus cond/uncond 在 batch 维度固定拼成 `[2B, 1, H]`。
- 避免 per-row Python list cache 在热路径中频繁拆/拼；如果必须拆/拼，封装成 helper 并单测。

### 4.5 Profiling 验收

每个阶段都必须用已有 torch profiler 证明：

- `use_cache=False` 的 full-prefix forward 消失或只保留在 train-time replay。
- rollout wall-clock 明显下降。
- `cudaLaunchKernel` call count 下降。
- `cudaStreamSynchronize` 占比下降。
- reward time 与 model generation time 分开统计。

## 5. 需要编辑/新增的文件

### 5.1 AR 基础设施

编辑：

```text
vrl/models/ar.py
vrl/engine/ar/sequence.py
vrl/engine/ar/token_scheduler.py
vrl/engine/ar/spec.py
vrl/engine/ar/__init__.py
```

新增：

```text
vrl/engine/ar/kv_cache.py
```

用途：

- 放通用 KV row split/concat/validate helper。
- 统一处理 HF `DynamicCache` / legacy tuple cache。
- 提供 batch cache gather/scatter 工具，避免每个 family 自己手写。
- 当前 `ar_split_rows()` / `ar_concat_rows()` 可以迁移或保留兼容 re-export。

### 5.2 Janus policy

编辑：

```text
vrl/models/families/janus_pro/policy.py
```

主要改动：

- 新增 Janus KV state 字段，保存 cond/uncond `past_key_values`。
- `init_ar_state()` 先 prefill cond/uncond prompt。
- `_sample_ar_step()` 改为 one-token decode。
- 保留旧 full-prefix path 作为短期 fallback，用 config/flag 控制。
- `sample_image_tokens()` 默认走 KV decode；必要时测试可指定 legacy path。
- R1 `generate_with_refine()` 复用同一个 KV decode path 生成 initial/final image。

### 5.3 Janus executors

编辑：

```text
vrl/models/families/janus_pro/executor.py
vrl/models/families/janus_pro/r1_executor.py
vrl/models/families/janus_pro/builder.py
configs/model/janus_pro/1b.yaml
configs/sampling/janus_384_576tok.yaml
configs/sampling/janus_r1_384_576tok.yaml
```

主要改动：

- `use_ar_scheduler` 默认打开前，先让 direct path 和 scheduler path 都支持 KV decode。
- 新增 sampling/config 字段：

```yaml
sampling:
  ar_decode_backend: kv_cache   # kv_cache | legacy_full_prefix
  ar_scheduler_batch_size: auto
```

- builder 暴露 `supports_kv_decode` / `supports_token_scheduler` debug metadata。

### 5.4 NextStep 对齐

编辑：

```text
vrl/models/families/nextstep_1/policy.py
vrl/models/families/nextstep_1/executor.py
vrl/models/families/nextstep_1/builder.py
configs/sampling/*
```

主要改动：

- 用新的 `vrl/engine/ar/kv_cache.py` helper 替换本地 cache split/concat 逻辑。
- 确认 `use_ar_scheduler=true` 下没有 full-prefix forward。
- 补 metrics：cache prefill count、decode step count、scheduler batch size。

### 5.5 Rollout runtime

编辑：

```text
vrl/distributed/ray/rollout/worker.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/runtime.py
vrl/rollouts/runtime/config.py
vrl/rollouts/runtime/backend.py
vrl/engine/core/worker.py
```

主要改动：

- 确认 rollout actor 常驻 policy，不在 request 之间释放模型。
- 给 generation metrics 增加 AR decode counters。
- 如果后续加 request-level batching，先在 `GenerationWorker._execute_group()` 里走 family executor 的 `forward_batch()`，不要绕过现有 registry。
- `release_after_collect` 保留给单卡 debug，多卡 profiling wrapper 默认关闭。

### 5.6 Tests

编辑/新增：

```text
tests/engine/generation/test_ar_token_scheduler.py
tests/models/test_ar_cache.py
tests/models/test_janus_wrapper.py
tests/models/test_janus_r1_policy.py
tests/models/test_nextstep_1_policy.py
tests/models/test_nextstep_1_executor.py
tests/distributed/ray/test_rollout_worker.py
```

新增：

```text
tests/models/test_janus_kv_decode.py
tests/models/test_janus_executor_kv_decode.py
```

测试重点：

- Janus KV path 与 legacy path 输出 shape、dtype、mask、logprob contract 一致。
- fake LM 记录 `use_cache=True` 和 `past_key_values`，确保 decode 真的走 cache。
- R1 initial/final image segments 都使用 KV image decode。
- scheduler 不混 position、dtype、family、tokenizer。
- Ray worker profile 仍能写 rollout trace。

### 5.7 Profiling 配置与文档

编辑：

```text
configs/profile/janus_pro_r1_codex_qa_1epoch.yaml
configs/profiling/torch_profiler.yaml
README.md
```

新增：

```text
configs/profile/janus_pro_r1_codex_qa_kv_decode_1epoch.yaml
```

用途：

- 固定一个 before/after profile recipe。
- 保留 legacy full-prefix baseline，方便回归比较。
- README 只链接命令和结果位置，不写长配置。

## 6. 实施阶段

### Phase 1：Janus KV decode 最小闭环

完成标准：

- `JanusProPolicy.init_ar_state()` 对 cond/uncond prompt 做 prefill。
- `_sample_ar_step()` 使用 `past_key_values` 和 `use_cache=True`。
- `sample_image_tokens()` 默认走 KV decode。
- 旧 full-prefix path 可通过 `ar_decode_backend=legacy_full_prefix` 保留。
- 单元测试证明每个 decode step 只喂一个 token embedding。

### Phase 2：R1 三段生成接入 KV decode

完成标准：

- `generate_with_refine()` 的 initial/final image generation 使用同一 KV decode path。
- self-check text generation 暂不强行纳入 image KV path，但不能破坏 image segment 的 cache。
- segment payload、token logprob、token mask 与当前训练 packer 兼容。
- `tests/models/test_janus_r1_policy.py` 覆盖 initial/final segment。

### Phase 3：AR scheduler 与 KV helper 收敛

完成标准：

- `vrl/engine/ar/kv_cache.py` 提供 shared cache split/concat helper。
- Janus 和 NextStep 都使用 shared helper。
- `ARTokenScheduler` batch key 能区分 decode backend。
- `use_ar_scheduler=true` 在 Janus/NextStep fake tests 中通过。

### Phase 4：Ray rollout 常驻与 request-level batching

完成标准：

- 多 GPU配置默认 `release_after_collect=false`。
- Ray actor request 之间保持 policy resident。
- `max_inflight_chunks_per_worker` 与 token scheduler 没有语义冲突。
- 如果实现 `forward_batch()`，必须保持 prompt-major sample order 和 deterministic seed contract。

### Phase 5：Profile 对比与默认开关

完成标准：

- 跑 legacy baseline profile。
- 跑 KV decode profile。
- 输出目录中有 trainer trace 和 rollout trace。
- `phase_events.jsonl` 能看到 `collect.engine_generate` 下降。
- rollout profiler summary 中 `cudaLaunchKernel` call count 下降。
- 没有 regression 时，Janus sampling config 默认切到 `kv_cache`。

## 7. 验收命令

基础单元测试：

```bash
pytest tests/engine/generation/test_ar_token_scheduler.py \
  tests/models/test_ar_cache.py \
  tests/models/test_janus_wrapper.py \
  tests/models/test_janus_kv_decode.py \
  tests/models/test_janus_r1_policy.py \
  tests/models/test_nextstep_1_policy.py \
  tests/models/test_nextstep_1_executor.py
```

Janus profile baseline：

```bash
python -m vrl.scripts.train --config profile/janus_pro_r1_codex_qa_1epoch \
  sampling.ar_decode_backend=legacy_full_prefix
```

Janus KV profile：

```bash
python -m vrl.scripts.train --config profile/janus_pro_r1_codex_qa_kv_decode_1epoch
```

结果检查：

```bash
ls outputs/profile_janus_r1_codex_qa_1epoch/torch_profiler/rollout
ls outputs/profile_janus_r1_codex_qa_kv_decode_1epoch/torch_profiler/rollout
```

## 8. 风险与处理

- HF cache 类型不稳定：用 helper 同时支持 `DynamicCache` 和 legacy tuple cache。
- Janus cond/uncond cache shape 容易错：所有 step tests 都用 fake LM 断言 batch dim。
- R1 self-check text path 可能仍然 full-prefix：本 sprint 先优化 image segments，后续单独优化 text reflection。
- `torch.compile` 可能掩盖首次编译时间：KV profile 通过后再开单独 sprint。
- 单卡 release-after-collect 会让常驻 worker 优势看不出来：真实性能验收以多 GPU或非释放模式为准。

## 9. 最终完成标准

这个 sprint 完成时，应该能清楚回答：

- Janus image-token generation 是否已经从 O(L²) full-prefix decode 变成 O(L) KV decode。
- 每个 AR family 是否都有统一的 scheduled decode contract。
- rollout worker 是否能常驻 policy 并 profile family inference。
- KV decode 相比 legacy baseline 在真实 Janus R1 recipe 上带来多少 wall-clock 和 profiler 改善。
