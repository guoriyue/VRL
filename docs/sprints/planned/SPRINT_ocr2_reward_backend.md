# SPRINT: 用 DeepSeek-OCR-2 替换 OCR reward 的 PaddleOCR 后端

状态：**planned（2026-06-27）**。目标：把 OCR reward 的文本识别后端从 PaddleOCR（CPU、轻量）换成 `DeepSeek-OCR-2`（VLM、更准、可 vLLM 服务），在**不动打分逻辑**的前提下提升识别质量——但先用一次性 probe 证明延迟/显存可接受,否则不落地。

## 0. 结论先行

- 替换面**很干净**：`vrl/rewards/models/ocr.py` 里识别后端是 3 个隔离函数（build / run / extract），打分逻辑（归一化 + Levenshtein + 帧聚合）只吃一个 plain-text 字符串,**完全不用改**。
- **最大风险是延迟**：PaddleOCR 是 CPU、~50–100ms/帧；DeepSeek-OCR-2 是 3B+ VLM,单帧推理 ~200–500ms。OCR reward 在 RL loop 里**逐帧**打分(video 默认 `frame_interval=4`),换上去可能把 reward 阶段拖成新瓶颈。**所以这是一个 probe-gated sprint**：先测吞吐/显存,过门再接。
- 次要风险：`transformers` 版本冲突(OCR-2 钉 `4.46.3`,reward extras 现钉 `>=4.49`);32GB 单卡显存未验证(A100-40G 是官方测试机)。

## 1. 证据

### 1.1 现有后端的三个隔离 seam（都在 `vrl/rewards/models/ocr.py`）

```python
# 195-203  逐帧识别：唯一吃 engine 的地方
def _run_paddle_ocr(engine, frame):           # frame: np.uint8 [H,W,C]
    if hasattr(engine, "predict"): return engine.predict(frame)
    ...
    return engine.ocr(frame, cls=False)
```

```python
# 170-192  懒加载构造 engine
def _build_paddle_ocr() -> Any: ...           # 返回 PaddleOCR 实例

# 206-241  把 engine 的嵌套输出解析成 plain text
def _extract_ocr_text(result) -> str: ...
```

打分主循环 `OCRRewardModel.__call__`（109-136）只依赖 `_normalize_ocr_text(text)` 之后的字符串：

```python
text = _normalize_ocr_text(text_raw)          # replace(" ","").lower()
dist = 0 if single_image and target_text in text else distance(text, target_text)
reward = 1.0 - dist / target_len
```

→ **打分逻辑与后端解耦**;换后端只需替换 build/run/extract 三个函数。

### 1.2 engine 注入点已存在（测试友好）

`ocr.py:45-58`：`self._engine = cfg.get("engine")` —— 后端实例由 `worker_config["engine"]` 注入(测试用 `_FakePaddleOCR`)。新后端沿用同一注入口,零新增 seam。

### 1.3 DeepSeek-OCR-2 推理 API（`~/Desktop/deep-research/DeepSeek-OCR-2`）

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR-2",
            _attn_implementation="flash_attention_2", trust_remote_code=True).eval().cuda().to(torch.bfloat16)
text = model.infer(tokenizer, prompt="<image>\nFree OCR. ", image_file=pil_image, save_results=False)
```

- HF: `deepseek-ai/DeepSeek-OCR-2`;输出是 markdown 文本串(plain OCR prompt 下基本就是识别文本)。
- deps：`transformers==4.46.3`、`tokenizers==0.20.3`、`einops`、`Pillow`;可选 vLLM 路径(`vllm==0.8.5`)做高吞吐异步服务。
- 官方测试机 A100-40G;bf16。

## 2. 应该做什么

### Phase 0 — KILL-RISK probe（先做,过门才继续）

写一次性 `*_probe`（用完即删,结论记本 sprint）：
1. 在目标卡(5090/32GB)加载 OCR-2 bf16,测**单帧识别延迟**与**峰值显存**。
2. 与 PaddleOCR 在同一组帧上比 (a) 单帧延迟 (b) 识别一致性(对几张含文字的真实/合成帧,看 normalized-Levenshtein 打分是否一致或更准)。
3. **过门条件**：单帧延迟在可接受范围(目标 ≤ 现 reward 阶段单帧预算,具体值由 probe 测得的 reward-phase 占比定),且 32GB 能与 rollout 共卡或可用 `execution` 隔离。**不过门 → 停,记录为 negative,不接。**

### Phase 1 — 后端接入（过门后）

- 新增 `_build_deepseek_ocr()` 返回 `(model, tokenizer)`;改 `_run_deepseek_ocr((model,tokenizer), frame)`：`Image.fromarray(frame)` → `model.infer(...)` → 返回 str。
- `_extract_ocr_text` 对 plain-OCR 输出退化为近乎透传(必要时正则剥 `<|ref|>/<|det|>` grounding tag)。
- **保持 PaddleOCR 为默认**;后端用 `worker_config` 里一个 `backend: paddleocr|deepseek_ocr2` 开关选择(de-hardcode,不破坏现有 recipe)。
- deps 进 `pyproject.toml` 的 reward extra,**解决 transformers 版本冲突**(见 §3)。

## 3. 风险与门槛

| 风险 | 处理 |
|---|---|
| **延迟**(VLM 比 CPU OCR 慢 2–10×,在 RL reward 逐帧路径) | Phase-0 probe 硬门;过不了就不接。可选 vLLM 异步 + 批量帧打分摊薄 |
| **transformers 版本**(OCR-2 `4.46.3` vs reward `>=4.49`) | 先验真实兼容区间;能则放宽约束,不能则把 OCR-2 隔到独立 extra / 独立 runtime 进程 |
| **32GB 共卡显存** | probe 实测峰值;必要时 `execution=pool` 隔离或与 rollout 分时 |
| **输出含 grounding tag** | plain-OCR prompt 基本无 tag;保留正则兜底 |

## 4. 非目标

- 不改打分逻辑(归一化/Levenshtein/帧聚合保持不变)——这是本 sprint 的 invariant。
- 不删 PaddleOCR 后端(保留为默认与回退)。
- 不在本 sprint 引入 vLLM 服务化(先证明 transformers 直跑可行;vLLM 作为延迟过不了门时的后续手段)。

## 5. 验收

- [ ] Phase-0 probe 数据落本 sprint：OCR-2 vs PaddleOCR 的单帧延迟、32GB 峰值显存、识别一致性。**过门才进 Phase 1。**
- [ ] 后端开关 `backend=deepseek_ocr2` 跑通,`{"ocr": float}` 契约不变,现有 OCR reward 测试在 `paddleocr` 默认下全绿。
- [ ] transformers 版本冲突有明确结论(放宽 / 隔离),CI clean-install 不破。
- [ ] 一句话决策记入本文件:接 / 不接 + 依据。

**参考**：`vrl/rewards/models/ocr.py:45-58,109-136,170-203,206-241`;`~/Desktop/deep-research/DeepSeek-OCR-2`(`run_dpsk_ocr2.py`、`requirements.txt`);HF `deepseek-ai/DeepSeek-OCR-2`。
