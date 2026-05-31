# SPRINT(auto): vrl/generation/ar/executor.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/ar/executor.py` (159 LOC)
角色判定: facade
结论: improve

## 0. 一句话
`ARPipelineExecutorBase` 把 `ARRequestLayout` 的方法逐个一行转发出来，但调用方（两个 AR runtime）同时又直接用 `self.layout.xxx`，导致同一套 layout 表面在 base 上被复制了一份冗余 wrapper，应收敛转发面、只保留有真实逻辑的方法。

## 1. 现状（读代码得出）
`ARPipelineExecutorBase` 中大部分方法是对 `self.layout` 的纯转发，例如：

```python
def parse_sampling_params(self, request: GenerationRequest) -> ARSamplingParams:
    return self.layout.parse_sampling_params(request)

def expand_prompts(self, request: GenerationRequest) -> list[str]:
    return self.layout.expand_prompts(request)

def validate_chunk(self, request, chunk) -> None:
    self.layout.validate_chunk(request, chunk)

def ordered_chunks(self, request, sample_rows, chunks, *, row_fields=()) -> list[TChunk]:
    return self.layout.ordered_chunks(request, sample_rows, chunks, row_fields=row_fields)

def require_rows(self, name, value, count) -> None:
    self.layout.require_rows(name, value, count)

def chunk_seed_offset(self, request, chunk) -> int:
    return self.layout.chunk_seed_offset(request, chunk)

def chunk_sample_rows(self, request, chunk) -> list[GenerationSampleRow]:
    return self.layout.chunk_sample_rows(request, chunk)

def max_peak_memory_mb(self, chunks) -> float | None:
    return self.layout.max_peak_memory_mb(chunks)

def align_pair(self, a_ids, a_mask, b_ids, b_mask, pad_id=0):
    return self.layout.align_pair(a_ids, a_mask, b_ids, b_mask, pad_id=pad_id)

def peak_memory_mb(self) -> float | None:
    return self.layout.peak_memory_mb()
```
(executor.py:47-154)

而 `layout` 本身也只是用 base 上的三个 `default_*` 字段拼出一个 dataclass：

```python
@property
def layout(self) -> ARRequestLayout:
    return ARRequestLayout(
        default_image_token_num=self.default_image_token_num,
        default_image_size=self.default_image_size,
        default_max_text_length=self.default_max_text_length,
    )
```
(executor.py:39-45)

关键证据：两个 runtime 调用方**并没有统一走 base 的转发方法**，而是 base wrapper 和 `self.layout.xxx` 混用：

- janus_pro/runtime.py 里有 `self.layout.ordered_chunks(...)`（:745, :1102）、`self.layout.max_peak_memory_mb(...)`（:766, :1131）、`self.layout.parse_sampling_params(request).image_token_num`（:767）；同时又用 `self.parse_sampling_params(...)`（:404/:553/:870/:966）、`self.chunk_sample_rows(...)`、`self.align_pair(...)`。
- nextstep_1/runtime.py 同样混用：`self.layout.ordered_chunks(...)`（:653）、`self.layout.max_peak_memory_mb(...)`（:676）与 `self.parse_sampling_params(...)` / `self.chunk_sample_rows(...)` 并存。

也就是说，转发方法没有把 `self.layout` 这个事实隐藏掉——调用方本来就在直接访问 `self.layout`，转发层没有提供边界价值，只是给同一组方法开了第二个入口。

base 上确实有真实逻辑的方法只有两个：
- `forward_batch_plan(...)`（executor.py:67-100）——含 plan/forward_plan 反射校验 + RequestBatch 编排。
- `require_native_ar_engine(...)`（executor.py:50-62）——拒绝 `ar_engine=vllm` 等非法 selector，是 AR 专有策略，不在 layout 里。

## 2. 质疑点 / 改进机会
1. 薄 wrapper 冗余（AGENTS.md thin-function 规则）：`parse_sampling_params / expand_prompts / validate_chunk / ordered_chunks / require_rows / chunk_seed_offset / chunk_sample_rows / max_peak_memory_mb / align_pair / peak_memory_mb` 这 10 个方法都是单行转发，没有 protocol 边界、没有移除复杂度。证据是调用方已经在直接用 `self.layout.ordered_chunks`、`self.layout.max_peak_memory_mb`、`self.layout.parse_sampling_params`（janus_pro/runtime.py:745/766/767/1102/1131；nextstep_1/runtime.py:653/676），说明这层转发并未被当成真正的封装边界。
2. 双入口造成不一致：同一个 `parse_sampling_params` 在两个 runtime 里有时走 `self.parse_sampling_params`、有时走 `self.layout.parse_sampling_params`，纯属转发层制造的二义性，增加 grep/调试噪声。

## 3. 建议动作
两个方向选其一，推荐方向 A：

方向 A（去掉冗余转发，统一走 `self.layout`）：
- 删除 executor.py:47-154 里的 10 个纯转发方法。
- 把 runtime 里所有 `self.parse_sampling_params(...)`、`self.expand_prompts(...)`、`self.validate_chunk(...)`、`self.require_native_ar_engine` 之外的转发调用统一改成 `self.layout.<method>(...)`，与现有 `self.layout.ordered_chunks` 等调用对齐。
- base 仅保留 `layout` property、`forward_batch_plan`、`require_native_ar_engine` 三个有真实价值的成员。
- 注意：`require_native_ar_engine` 留在 base（AR 引擎选择策略，不属于 layout）；`forward_batch_plan` 留在 base（编排逻辑）。

方向 B（彻底统一走 base，禁止直接摸 layout）：
- 如果团队更倾向把 `self.layout` 当内部实现、对外只暴露 base 方法，则反过来：把 runtime 里 6 处 `self.layout.xxx` 改回 `self.xxx`，并把 `layout` property 设为内部（前缀 `_layout`）。这样转发层才真正成为边界。

二者都消除"同一方法两个入口"的不一致。方向 A 改动更小、更贴近现状（runtime 已大量直接用 `self.layout`），优先选 A。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `forward_batch_plan`（executor.py:67-100）：含 plan 反射 + RequestBatch 编排，是真实复杂度；与 diffusion base 的 `forward_batch_plan`（diffusion/executor.py:626-653）形成跨家族一致形状，属于 AGENTS.md "consistency over cleanup" 应保留的对称结构。
- 不动 `require_native_ar_engine`（executor.py:50-62）：AR 专有的引擎 selector 拒绝逻辑，不属于 layout 的请求布局职责。
- 不动 `layout` property 本身：它把 base 的 `default_*` 配置注入 `ARRequestLayout`，是合理的配置组装点。
- 不要把 `ARRequestLayout` 拍平进 base：layout 被 runtime 和 test（test_nextstep_1_kv_decode.py:28/61 直接 `ARRequestLayout().parse_sampling_params`）独立复用，是真正的共享抽象，必须留在 layout.py。

## 5. 验证
- 改完跑 `pytest tests/generation/ar/test_ar_engine_selection.py tests/models/test_nextstep_1_kv_decode.py tests/models/test_janus_kv_decode.py -q`。
- `grep -rn "self\.parse_sampling_params\|self\.layout\.parse_sampling_params" vrl/models/ar/` 确认只剩一种调用风格。
- `grep -rn "ARPipelineExecutorBase" vrl tests` 确认两个 runtime + 测试 fake 仍正常继承。
- `ruff check vrl/generation/ar/executor.py`。
